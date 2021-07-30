import contextlib
from types import MethodType
import traceback

import torch

from fastai.learner import Callback, CancelBatchException
from fastai.callback.fp16 import NativeMixedPrecision
from fastai.torch_core import rank_distrib
from fastai.distributed import DistributedTrainer, rank0_first

from torch.utils.tensorboard import SummaryWriter


def maybe_clip_grad_norm(model, clip):
    if clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip, 2.0)


class GradAccumCallback(Callback):
    count = 0
    run_after = NativeMixedPrecision

    def __init__(self, num_batches, clip):
        self.num_batches = num_batches
        self.clip = clip

    def after_backward(self):
        # increment
        self.count += 1

        # skip opt step if applicable
        if (self.count % self.num_batches) == 0:
            # do optimizer step
            # if (self.count % 200) == 0:
            #     print(torch.cuda.memory_summary())
            maybe_clip_grad_norm(self.model, self.clip)
        else:
            # skip optimizer step
            raise CancelBatchException()


class GradAccumCallbackDDP(Callback):
    count = 0
    run_after = DistributedTrainer  # MixedPrecision

    def __init__(self, num_batches, clip):
        self.num_batches = num_batches
        self.clip = clip

    def before_fit(self):
        self.cm_do_sync = contextlib.nullcontext
        self.cm_no_sync = self.model.no_sync
        self.learn.ddp_cm = self.cm_do_sync

        # patch learner for ddp
        def _do_one_batch(self):
            cm = self.ddp_cm()
            with cm:
                self.pred = self.model(*self.xb)
                self("after_pred")
                if len(self.yb):
                    self.loss = self.loss_func(self.pred, *self.yb)
                self("after_loss")
                if not self.training or not len(self.yb):
                    return
                self("before_backward")
                self._backward()
                self("after_backward")
                self._step()
                self("after_step")
                self.opt.zero_grad()

        self.learn._do_one_batch = MethodType(_do_one_batch, self.learn)

    def after_backward(self):
        # increment
        self.count += 1

        # set ddp_cm for upcoming batch
        if ((self.count + 1) % self.num_batches) == 0:
            self.learn.ddp_cm = self.cm_do_sync
        else:
            self.learn.ddp_cm = self.cm_no_sync

        # skip opt step if applicable
        if (self.count % self.num_batches) == 0:
            # do optimizer step
            maybe_clip_grad_norm(self.model, self.clip)
        else:
            # skip optimizer step
            raise CancelBatchException()


class Rank0Wrapper(Callback):
    """
    - NOT WORKING -
    Execute a given Callback-Instance only on Rank 0.
    """

    def __init__(self, cb):
        self.cb = cb

        # patch in all calls
        from fastai.learner import _loop

        funcs = filter(lambda x: "_" in x, _loop)
        for f in funcs:
            cf = getattr(cb, f, None)
            if cf is not None:

                def wrapped(self, *args, **kwargs):
                    if rank_distrib() == 0:
                        cf(self, *args, **kwargs)

                setattr(self, f, wrapped)


class EvaluatorCallback(Callback):
    run_after = NativeMixedPrecision

    def __init__(self, evaluator, ddp=False, tests_per_epoch=8, enable=True, **kwargs):
        self.evaluator = evaluator
        self.ddp = ddp
        self.tests_per_epoch = tests_per_epoch
        self.kwargs = kwargs
        self.is_fitting = False
        self.train_batch_count = 0
        self._back_to_train = True
        self.enable = enable

    def get_writer(self):
        if hasattr(self.learn, "summary_writer"):
            return self.learn.summary_writer
        return None

    def before_fit(self):
        self.is_fitting = True
        self.train_batch_count = 0

    def before_batch(self):
        if self.is_fitting:
            if self.training:
                self.train_batch_count += 1

    def _pre(self):
        self._back_to_train = self.learn.model.training
        self.learn.model.eval()

    def _post(self):
        if self._back_to_train:
            self.learn.model.train()

    def after_loss(self):
        if self.tests_per_epoch <= 0 or not self.enable:
            return

        def save_fn(fname, **kwargs):
            self.save(fname, **kwargs)

        if self.training:
            a = self.learn.train_iter + 1
            b = int((self.n_iter * (1 / self.tests_per_epoch)))
            if a % b == 0:
                self._pre()
                metrics = []
                try:
                    metrics = self.evaluator.eval(
                        self.learn, ddp=self.ddp, save_fn=save_fn, **self.kwargs
                    )
                except Exception as e:
                    print("failed to evaluate in EvaluatorCallback...")
                    print(e)
                    traceback.print_exc()
                i = self.train_batch_count
                for metric in metrics:
                    w = self.get_writer()
                    if w is not None:
                        for k, v in metric.items():
                            if "metrics" in k:
                                w.add_scalar(k, v, i)
                            else:
                                w.add_text(k, v, i)
                    i += 1
                self._post()

    def after_fit(self):
        self.is_fitting = True


FLUSH_SECS = 120
LOG_EVERY_N_STEPS = 16
LOG_DEBUG_EVERY_N_STEPS = 256


class Tensorboard(Callback):
    toward_end = True

    def __init__(self, name=None, wandb=True, test=True, ddp=False, *args, **kwargs):
        self.tb_name = name
        self.tb_use_wandb = wandb
        self.is_fitting = False
        self.train_batch_count = 0
        self.valid_batch_count = 0
        self.test = test
        self.ddp = ddp
        self.steps = 0
        self.logging_initialized = False

    def before_fit(self):

        # setup logging
        if not self.logging_initialized:

            # setup wandb
            if self.tb_use_wandb:
                import wandb

                wandb.init(
                    project="LibreASR",
                    entity="iceychris",
                    sync_tensorboard=True,
                    config=self.learn.conf,
                )

            # setup tensorboard
            self.writer = SummaryWriter(flush_secs=FLUSH_SECS)

            self.logging_initialized = True

        self.is_fitting = True
        self.train_batch_count = 0
        self.valid_batch_count = 0
        self.writer.add_scalar("epochs", self.learn.n_epoch, self.train_batch_count)
        self.learn.summary_writer = self.writer

    def before_batch(self):
        if self.is_fitting:
            if self.training:
                self.train_batch_count += 1
            else:
                self.valid_batch_count += 1

    def after_backward(self):
        if self.training and self.steps % LOG_EVERY_N_STEPS == 0:
            self.writer.add_scalar(
                "data/seqlen/x", self.learn.xb[0][0].size(1), self.train_batch_count
            )
            self.writer.add_scalar(
                "data/seqlen/y", self.learn.xb[0][1].size(1), self.train_batch_count
            )
            self.writer.add_scalar(
                "data/batch_size",
                self.learn.xb[0][0].size(0),
                self.train_batch_count,
            )

    def _log_debugging(self, func, title="parameters", hist=False):
        for name, param in self.model.named_parameters():

            # prepare tensor
            param = func(param)
            if param is None:
                continue
            m, s = param.mean().item(), param.std().item()

            # save hist
            if hist:
                param = param.detach().cpu().float()
                tag = f"{title}/{name}"
                self.writer.add_histogram(tag, param, self.train_batch_count)

            # save mean & std
            tag = f"debugging-{title}-mean/{name}"
            self.writer.add_scalar(tag, m, self.train_batch_count)
            tag = f"debugging-{title}-std/{name}"
            self.writer.add_scalar(tag, s, self.train_batch_count)

    def after_step(self):
        if self.steps % LOG_EVERY_N_STEPS == 0:
            hyps = self.learn.opt.hypers[0]
            for k, v in hyps.items():
                self.writer.add_scalar("hypers/" + k, v, self.train_batch_count)
            if hasattr(self.learn.opt, "opt"):
                if hasattr(self.learn.opt.opt, "extra"):
                    extra = self.learn.opt.opt.extra
                    for k, v in extra.items():
                        self.writer.add_scalar("extra/" + k, v, self.train_batch_count)
        if self.steps % LOG_DEBUG_EVERY_N_STEPS == 0:
            self._log_debugging(lambda x: x, "parameters")
            self._log_debugging(lambda x: x.grad, "gradients")

        self.steps += 1

    def after_fit(self):
        self.is_fitting = False

    def report_loss_dict(self, loss_dict):
        if self.training:
            batch_cnt = self.train_batch_count
            suffix = "train"
            div = LOG_EVERY_N_STEPS
        else:
            batch_cnt = self.valid_batch_count
            suffix = "valid"
            div = 1
        if self.steps % div == 0:
            for k, v in loss_dict.items():
                v = v.detach().cpu().float().mean().item()
                self.writer.add_scalar(k + "/" + suffix, v, batch_cnt)
