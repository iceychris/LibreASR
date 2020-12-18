"""
Contains Tensorboard Callback which performs testing while training.
"""

from fastai2.fp16_utils import convert_network
from fastai2.torch_core import to_float
from fastai2.learner import Callback

from torch.utils.tensorboard import SummaryWriter

FLUSH_SECS = 30
LOG_EVERY_N_STEPS = 4


class Tensorboard(Callback):
    toward_end = True

    def __init__(
        self,
        name=None,
        wandb=True,
        test=True,
        tests_per_epoch=4,
        mp=False,
        *args,
        **kwargs
    ):
        if name:
            self.writer = SummaryWriter(flush_secs=FLUSH_SECS, log_dir="runs/" + name)
        else:
            self.writer = SummaryWriter(flush_secs=FLUSH_SECS)
        self.tb_name = name
        self.tb_use_wandb = wandb
        self.is_fitting = False
        self.train_batch_count = 0
        self.valid_batch_count = 0
        self.test = test
        self.test_cnt = tests_per_epoch
        self.mp = mp
        self.steps = 0

    def before_fit(self):
        if self.tb_use_wandb:
            import wandb

            wandb.init(project="asr", sync_tensorboard=True)
            wandb.watch(self.learn.model)
        self.is_fitting = True
        self.train_batch_count = 0
        self.valid_batch_count = 0
        self.writer.add_scalar("epochs", self.learn.n_epoch, self.train_batch_count)

    def before_batch(self):
        if self.is_fitting:
            if self.training:
                self.train_batch_count += 1
            else:
                self.valid_batch_count += 1

    def after_pred(self):
        pass

    def after_loss(self):
        if not self.training:
            self.writer.add_scalar(
                "loss/valid", self.learn.loss, self.valid_batch_count
            )

    def after_backward(self):
        if self.training and self.steps % LOG_EVERY_N_STEPS == 0:
            try:
                loss = self.learn.smooth_loss
                self.writer.add_scalar("loss/train", loss, self.train_batch_count)
            except:
                pass
            self.writer.add_scalar(
                "data/seqlen/x", self.learn.xb[0][0].size(1), self.train_batch_count
            )
            self.writer.add_scalar(
                "data/seqlen/y", self.learn.xb[0][1].size(1), self.train_batch_count
            )
            self.writer.add_scalar(
                "data/batch_size", self.learn.xb[0][0].size(0), self.train_batch_count
            )

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
        self.steps += 1

    def after_fit(self):
        self.is_fitting = False

    def after_batch(self):
        try:
            if self.training and self.test:
                a = self.learn.train_iter
                if self.test_cnt <= 0:
                    return
                b = int((self.n_iter * (1 / self.test_cnt)))
                if a % b == 0:
                    metrics = self.learn.test(mp=self.mp)
                    i = self.train_batch_count
                    for metric in metrics:
                        for k, v in metric.items():
                            if "metrics" in k:
                                self.writer.add_scalar(k, v, i)
                            else:
                                self.writer.add_text(k, v, i)
                        i += 1
        except Exception as e:
            print("failed to execute learn.test(...)")
            print(e)
            import traceback

            traceback.print_exc()
