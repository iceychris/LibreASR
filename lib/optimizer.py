from functools import partial

import torch
from torch.optim.optimizer import Optimizer

from fastai2.optimizer import *
from fastai2.torch_basics import *
from fastcore.utils import log_args


def average_sqr_diag_hessian(
    p,
    sqr_mom,
    dampening=True,
    sqr_avg_diag_hessian=None,
    hutchinson_trace=None,
    **kwargs
):
    if sqr_avg_diag_hessian is None:
        sqr_avg_diag_hessian = torch.zeros_like(p.grad.data)
    damp = 1 - sqr_mom if dampening else 1.0
    sqr_avg_diag_hessian.mul_(sqr_mom).addcmul_(
        hutchinson_trace, hutchinson_trace, value=damp
    )
    return {"sqr_avg_diag_hessian": sqr_avg_diag_hessian}


def adahessian_step(
    p,
    lr,
    mom,
    step,
    sqr_mom,
    grad_avg,
    sqr_avg_diag_hessian,
    hessian_power,
    eps,
    **kwargs
):
    "Step for Adam with `lr` on `p`"
    debias1 = debias(mom, 1 - mom, step)
    debias2 = debias(sqr_mom, 1 - sqr_mom, step)
    p.data.addcdiv_(
        grad_avg,
        ((sqr_avg_diag_hessian / debias2).sqrt() ** hessian_power) + eps,
        value=-lr / debias1,
    )
    return p


@log_args(to_return=True, but_as=Optimizer.__init__)
def AdaHessian(
    params,
    lr=0.1,
    hessian_power=1,
    hutchinson_trace=None,
    mom=0.9,
    sqr_mom=0.98,
    eps=1e-4,
    wd=0.0,
    decouple_wd=True,
):
    "A `Optimizer` for Adam with `lr`, `mom`, `sqr_mom`, `eps` and `params`"
    cbs = [weight_decay] if decouple_wd else [l2_reg]
    cbs += [
        partial(average_grad, dampening=True),
        average_sqr_diag_hessian,
        step_stat,
        adahessian_step,
    ]
    return Optimizer(
        params,
        cbs,
        lr=lr,
        mom=mom,
        sqr_mom=sqr_mom,
        hessian_power=hessian_power,
        eps=eps,
        wd=wd,
    )


class Apollo(Optimizer):
    r"""Implements Atom algorithm.
        Arguments:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float): learning rate
            beta (float, optional): coefficient used for computing
                running averages of gradient (default: 0.9)
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-4)
            warmup (int, optional): number of warmup steps (default: 0)
            init_lr (float, optional): initial learning rate for warmup (default: 0.01)
            wd (float, optional): weight decay coefficient (default: 0)
        """

    def __init__(self, params, lr, beta=0.9, eps=1e-4, warmup=100, init_lr=0.01, wd=0):
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate value: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta))
        if not 0.0 <= wd:
            raise ValueError("Invalid wd value: {}".format(wd))
        if not 0.0 <= warmup:
            raise ValueError("Invalid warmup updates: {}".format(warmup))
        if not 0.0 <= init_lr <= 1.0:
            raise ValueError("Invalid initial learning rate: {}".format(init_lr))

        defaults = dict(
            lr=lr, beta=beta, eps=eps, warmup=warmup, init_lr=init_lr, base_lr=lr, wd=wd
        )
        super(Apollo, self).__init__(params, defaults)

    """
    def __setstate__(self, state):
        super(Apollo, self).__setstate__(state)
    """

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        printed = False
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg_grad"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["approx_hessian"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Previous update direction
                    state["update"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                # Calculate current lr
                if state["step"] < group["warmup"]:
                    curr_lr = (group["base_lr"] - group["init_lr"]) * state[
                        "step"
                    ] / group["warmup"] + group["init_lr"]
                else:
                    curr_lr = group["lr"]
                # if not printed:
                #     print("lr", curr_lr)

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Atom does not support sparse gradients.")

                # Perform step weight decay
                if group["wd"] != 0:
                    grad = grad.add(p, alpha=group["wd"])

                beta = group["beta"]
                exp_avg_grad = state["exp_avg_grad"]
                B = state["approx_hessian"]
                d_p = state["update"]

                state["step"] += 1
                bias_correction = 1 - beta ** state["step"]
                alpha = (1 - beta) / bias_correction

                # Update the running average grad
                delta_grad = grad - exp_avg_grad
                exp_avg_grad.add_(delta_grad, alpha=alpha)

                denom = d_p.norm(p=4).add(group["eps"])
                d_p.div_(denom)
                v_sq = d_p.mul(d_p)
                delta = (
                    delta_grad.div_(denom).mul_(d_p).sum().mul(-alpha)
                    - B.mul(v_sq).sum()
                )

                # Update B
                B.addcmul_(v_sq, delta)

                # calc direction of parameter updates
                denom = B.abs().clamp_(min=1)
                d_p.copy_(exp_avg_grad.div(denom))

                if not printed:
                    # print("d_p mean", d_p.abs().mean())
                    printed = True

                # from IPython.core.debugger import set_trace
                # set_trace()
                p.add_(d_p, alpha=-curr_lr)

        return loss


"""
Ranger with AdaBelief
"""


def asqg(p, sqr_mom, grad_avg, dampening=True, sqr_avg=None, **kwargs):
    if sqr_avg is None:
        sqr_avg = torch.zeros_like(p.grad.data)
    damp = 1 - sqr_mom if dampening else 1.0
    # new
    sqr_avg.mul_(sqr_mom).addcmul_(
        p.grad.data - grad_avg, p.grad.data - grad_avg, value=damp
    )
    return {"sqr_avg": sqr_avg}


asqg.defaults = dict(sqr_mom=0.99)


def radam_adabelief_step(
    p, lr, mom, step, sqr_mom, grad_avg, sqr_avg, eps, beta, **kwargs
):
    "Step for RAdam with `lr` on `p`"
    debias1 = debias(mom, 1 - mom, step)
    debias2 = debias(sqr_mom, 1 - sqr_mom, step)
    r_inf = 2 / (1 - sqr_mom) - 1
    r = r_inf - 2 * step * sqr_mom ** step / (1 - sqr_mom ** step)
    if r > 5:
        v = math.sqrt(((r - 4) * (r - 2) * r_inf) / ((r_inf - 4) * (r_inf - 2) * r))
        # old
        # denom = (sqr_avg/debias2).sqrt()
        # new
        denom = (sqr_avg / debias2 + eps).sqrt()
        if eps:
            denom += eps
        if beta:
            denom = F.softplus(denom, beta)
        p.data.addcdiv_(grad_avg, denom, value=-lr * v / debias1)
    else:
        p.data.add_(grad_avg, alpha=-lr / debias1)
    return p


radam_adabelief_step._defaults = dict(eps=1e-5)


@log_args(to_return=True, but_as=Optimizer.__init__)
def RAdamAdabelief(
    params, lr, mom=0.9, sqr_mom=0.99, eps=1e-5, wd=0.0, beta=0.0, decouple_wd=True
):
    "A `Optimizer` for Adam with `lr`, `mom`, `sqr_mom`, `eps` and `params`"
    cbs = [weight_decay] if decouple_wd else [l2_reg]
    cbs += [
        partial(average_grad, dampening=True),
        asqg,
        step_stat,
        radam_adabelief_step,
    ]
    return Optimizer(
        params, cbs, lr=lr, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd, beta=beta
    )


@delegates(RAdamAdabelief)
def ranger_adabelief(p, lr, mom=0.95, wd=0.01, eps=1e-9, **kwargs):  # eps=1e-6
    "Convenience method for `Lookahead` with `RAdam`"
    return Lookahead(RAdamAdabelief(p, lr=lr, mom=mom, wd=wd, eps=eps, **kwargs))
