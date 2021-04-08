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
from fastai.callback.fp16 import MixedPrecision
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
    EvalSpeechModel,
)
from libreasr.lib.loss import get_loss_func
from libreasr.lib.optimizer import AdaHessian, Apollo, ranger_adabelief


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
    run_before = MixedPrecision

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


class ASRLearner(Learner):
    @staticmethod
    def from_config(conf, db, m):
        # pull info from config
        acnb = conf["batching"].get("accumulate", 1)
        tests_per_epoch = conf.get("tests_per_epoch", 0)
        mp = conf.get("training", {}).get("mp", {}).get("enable", False)
        lang_name = conf.get("lang", "unknown")
        ddp = conf.get("training", {}).get("ddp", {}).get("enable", False)
        espm_kwargs = conf.get("espm_kwargs", {})
        espm_kwargs.update({"lang_name": lang_name})
        use_persistence_cbs = not ddp or rank_distrib() == 0

        # define callbacks
        cbs = [
            CudaCallback(),
            TerminateOnNaNCallback(),
            ReduceLROnPlateau(patience=1, min_lr=1e-5, factor=1.5),
        ]
        if use_persistence_cbs:
            cbs.append(
                EvalSpeechModel(
                    ddp=ddp,
                    tests_per_epoch=tests_per_epoch,
                    espm_kwargs=espm_kwargs,
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
        if acnb > 1 and not optim == "adahessian":
            if ddp:
                gac = GradAccumCallbackDDP
            else:
                gac = GradAccumCallback
            cbs.append(gac(num_batches=acnb))
        extra_cbs = []
        if conf["tensorboard"]:
            if use_persistence_cbs:
                _tb = partial(
                    Tensorboard,
                    wandb=conf["wandb"],
                    test=True,
                    tests_per_epoch=conf["tests_per_epoch"],
                    ddp=ddp,
                )()
                extra_cbs.append(_tb)
        learn = Learner(
            db,
            m,
            loss_func=get_loss_func(
                conf["loss"]["type"],
                conf["cuda"]["device"],
                conf["loss"]["reduction_factor"],
                noisystudent=conf["training"]["noisystudent"],
                debug=False,
                perf=False,
                div_by_len=False,
            ),
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
