import contextlib
from functools import partial
import math
import gc

import torch
from fastcore.foundation import L, patch
from fastai.learner import *
from fastai.callback.core import Callback
from fastai.callback.tracker import (
    TerminateOnNaNCallback,
    SaveModelCallback,
    ReduceLROnPlateau,
)
from fastai.callback.fp16 import NativeMixedPrecision
from fastai.callback.data import CudaCallback
from fastai.optimizer import Adam, Lamb, Lookahead, ranger
from fastai.torch_core import rank_distrib
from fastai.distributed import DistributedTrainer

from IPython.core.debugger import set_trace

from libreasr.lib.callbacks import (
    Tensorboard,
    GradAccumCallback,
    GradAccumCallbackDDP,
    Rank0Wrapper,
    EvaluatorCallback,
)
from libreasr.lib.optimizer import AdaHessian, Apollo, ranger_adabelief
from libreasr.lib.utils import warn


HESSIAN_EVERY = 1  # 50


def transducer_splitter(m, adahessian=False):
    def ps_of_sub(mod):
        return [p for p in mod.parameters() if p.requires_grad]

    if adahessian:
        # l = L([*m.joint.param_groups(), *m.predictor.param_groups()])
        l = L([*m.param_groups()])
        l = [item for sublist in l for item in sublist]
        l = [p for p in l if p.requires_grad]
        for p in l:
            print(p.shape)
        print(len(l))
        return l
    else:
        ps = L(m.param_groups())
        return ps


def over9000(p, lr=slice(3e-3)):
    return Lookahead(Lamb(p, lr))


class HutchinsonTraceCallback(Callback):
    run_before = NativeMixedPrecision

    def __init__(self, block_length=1):
        self.block_length = block_length
        self.hutchinson_trace = []
        self.zs = []
        self.new_update = False

    def _clip_grad_norm(self, max_norm=0.0, params=None):
        """
        From FairSeq opimizer - Clips gradient norm.
        """
        if max_norm > 0:
            return torch.nn.utils.clip_grad_norm_(params, max_norm)
        else:
            return math.sqrt(
                sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None)
            )

    def before_fit(self):
        self.learn.opt._hutch_iter = 0

    def after_backward(self):
        """
        compute the Hessian vector product with a random vector v, at the current gradient point,
        i.e., compute the gradient of <gradsH,v>.
        :param gradsH: a list of torch variables
        :return: a list of torch tensors
        """

        # Add hutchinson_trace to optimizer state
        def add_hutchinson_to_optim(new_update):
            for i, (_, _, state, _) in enumerate(
                self.learn.opt.all_params(with_grad=True)
            ):
                state["hutchinson_trace"] = self.hutchinson_trace[i]
            self.new_update = new_update

        # only compute every nth step
        if self.learn.opt._hutch_iter % HESSIAN_EVERY != 0:
            add_hutchinson_to_optim(False)
            self.learn.opt._hutch_iter += 1
            return
        else:
            self.learn.opt._hutch_iter += 1

        device = self.learn.dl.device

        params, grads = [], []
        for (p, _, _, _) in self.learn.opt.all_params(with_grad=True):
            if p.requires_grad and p.grad is not None:
                # print(p.shape)
                params.append(p)
                grads.append(p.grad)

        # self._clip_grad_norm(max_norm=0., params=params)   # Not sure if this is needed...

        # cache zs
        if len(self.zs) == 0:
            self.zs = [
                torch.randint_like(p, high=2).to(device) * 2.0 - 1.0 for p in params
            ]
        else:
            for t in self.zs:
                t[:, ...] = torch.randint_like(t, high=2)
                t.mul_(2.0)
                t.sub_(1.0)

        # this line is very expensive memory-wise
        h_zs = torch.autograd.grad(
            grads, params, grad_outputs=self.zs, only_inputs=True, retain_graph=False
        )

        # debug
        # from IPython.core.debugger import set_trace; set_trace()
        # print(len(params), len(grads))

        # @LESSW CODE
        self.hutchinson_trace = []
        for hz, z in zip(h_zs, self.zs):
            param_size = hz.size()
            if len(param_size) <= 3:  # was <= 2 before, for 0/1/2D tensor
                tmp_output = torch.abs(hz * z) + 0.0  # .float()
                self.hutchinson_trace.append(
                    tmp_output
                )  # Hessian diagonal block size is 1 here.
            elif len(param_size) == 4:  # Conv kernel
                tmp_output = (
                    torch.abs(
                        torch.sum(torch.abs(hz * z) + 0.0, dim=[2, 3], keepdim=True)
                    ).float()
                    / z[0, 1].numel()
                )  # Hessian diagonal block size is 9 here: torch.sum() reduces the dim 2/3.
                self.hutchinson_trace.append(tmp_output)

        # apply
        add_hutchinson_to_optim(True)

    def after_step(self):
        pass


def reducer(report_loss_dict_fn, loss_dict, *args, reduction="mean", **kwargs):
    """
    Models are expected to output a
    loss tensor of shape [N].
    This function reduces this tensor
    to a scalar.
    Also perform a NaN check.
    """
    # check for NaN in all sublosses
    #  and bail
    for k, loss in loss_dict.items():
        if torch.isnan(loss).any():
            warn(f"Found NaN values in loss_dict (key {k}), mocking loss...")
            loss = torch.Tensor([0.0])
            loss.requires_grad_(True)
            loss_dict[k] = loss

    # extract main loss
    loss = loss_dict["loss"]

    # report
    report_loss_dict_fn(loss_dict)

    # reduce
    if reduction == "mean":
        return loss.mean()
    else:
        return loss


class LibreASRLearner(Learner):
    @staticmethod
    def from_config(conf, db, m):
        # pull info from config
        acnb = conf["batching"].get("accumulate", 1)
        tests_per_epoch = conf.get("tests_per_epoch", 0)
        mp = conf.get("training", {}).get("mp", {}).get("enable", False)
        lang_name = conf.get("lang", "unknown")
        ddp = conf.get("training", {}).get("ddp", {}).get("enable", False)
        evaluator_kwargs = conf.get("evaluator", {}).get("kwargs", {})
        evaluator_kwargs.update({"lang_name": lang_name})
        use_persistence_cbs = not ddp or rank_distrib() == 0
        clip = conf.get("training", {}).get("clip-grad-norm", 0.0)
        use_tensorboard = conf["tensorboard"]
        device = conf["cuda"]["device"]
        objective = conf["loss"]["type"] or "rnnt"

        # define callbacks
        cbs = [
            CudaCallback(),
            TerminateOnNaNCallback(),
            # ReduceLROnPlateau(patience=1, min_lr=1e-5, factor=1.5),
        ]

        # grab the correct evaluator
        if objective == "contrastive":
            from libreasr.eval import SaveEvaluator

            evaluator = SaveEvaluator()
        elif objective == "rnnt":
            from libreasr.eval import TranscribeEvaluator

            evaluator = TranscribeEvaluator(device=device, lang_name=lang_name)
        else:
            s = f"Evaluator for objective {objective} not implemented."
            raise NotImplementedError(s)

        # add callbacks that
        #  should only get executed once
        #  when running in ddp mode
        if use_persistence_cbs:
            cbs.append(
                EvaluatorCallback(
                    evaluator=evaluator,
                    ddp=ddp,
                    tests_per_epoch=tests_per_epoch,
                    **evaluator_kwargs,
                )
            )
            cbs.append(
                SaveModelCallback(
                    fname=f'{conf["loss"]["type"]}-best-validation-loss', with_opt=True
                )
            )
        optim = conf["training"]["optimizer"].lower()
        if optim == "ranger":
            opt_func = ranger
        elif optim == "ranger_adabelief":
            opt_func = ranger_adabelief
        elif optim == "adam":
            opt_func = Adam
        elif optim == "lamb":
            opt_func = Lamb
        elif optim == "apollo":
            from fastai.optimizer import OptimWrapper

            def of(param_groups, **kwargs):
                lr_init = 1e-4
                lr = 1e-3
                warmup = 10  # 1000
                wd = 4e-4
                apollo = Apollo(param_groups, lr=lr, warmup=warmup)
                new_pgs = []
                for pg in param_groups:
                    new_pgs.append(
                        {
                            "params": pg,
                            "lr": lr,
                            "wd": wd,
                            "mom": 0.99,
                            "eps": 1e-4,
                            "beta": 0.9,
                            "init_lr": lr_init,
                            "base_lr": lr,
                            "warmup": warmup,
                        }
                    )
                apollo.param_groups = new_pgs
                opt = OptimWrapper(apollo)
                return opt

            opt_func = of
        elif optim == "adahessian":
            opt_func = AdaHessian
            cbs.append(HutchinsonTraceCallback())

            @patch
            def _backward(self: Learner):
                if self.opt._hutch_iter % HESSIAN_EVERY == 0:
                    self.loss.backward(create_graph=True)
                else:
                    self.loss.backward()

        else:
            raise Exception("No such optimizer")

        # Gradient Accumulation
        #  & Gradient Clipping
        if not optim == "adahessian":
            if ddp:
                gac = GradAccumCallbackDDP
            else:
                gac = GradAccumCallback
            cbs.append(gac(num_batches=acnb, clip=clip))

        # Tensorboard
        extra_cbs = []
        report_loss_dict_fn = lambda x: None
        if use_tensorboard:
            if use_persistence_cbs:
                _tb = partial(
                    Tensorboard,
                    wandb=conf["wandb"],
                    test=True,
                    tests_per_epoch=conf["tests_per_epoch"],
                    ddp=ddp,
                )()
                report_loss_dict_fn = _tb.report_loss_dict
                extra_cbs.append(_tb)

        # construct learner
        learn = Learner(
            db,
            m,
            loss_func=partial(reducer, report_loss_dict_fn),
            opt_func=opt_func,
            splitter=partial(transducer_splitter, adahessian=(optim == "adahessian")),
            cbs=cbs,
        )
        learn.extra_cbs = extra_cbs
        learn.conf = conf
        if mp:
            learn = learn.to_native_fp16()
            print("NativeMixedPrecision activated.")
        return learn
