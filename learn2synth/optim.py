"""
This module contains reimplementations of PyTorch's optimizers
that handle backpropagating though the update step.

Note that recent versions of PyTorch seem to add a `differentiable`
argument to these optimizers that may have a similar effect. I am not
sure when it got introduced and haven't tried it yet.
"""
from torch.optim import Optimizer as OptimizerBase
import torch
import math


required = getattr(torch.optim, 'required', None)
if not required:
    class required:
        """Singleton to specify that an input is required"""
        pass


class Optimizer(OptimizerBase):
    def parameters(self):
        for group in self.param_groups:
            for param in group['params']:
                yield param


class SGD(Optimizer):
    def __init__(self, parameters, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):

        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(parameters, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lam = group.get('weight_decay', 0)
            mu = group.get('momentum', 0)
            tau = group.get('dampening', 0)
            gamma = group['lr']
            nes = group.get('nesterov', False)
            for param in group['params']:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]
                momentum = state.get('momentum_buffer', None)

                if lam:
                    grad = grad.add(param, alpha=lam)
                if mu:
                    if momentum is None:
                        momentum = grad.clone()
                    else:
                        momentum.detach_().copy_(momentum.mul_(mu).add_(grad, alpha=1-tau))
                    if nes:
                        grad = torch.add(grad, momentum, alpha=mu)
                    else:
                        grad = momentum

                param.detach_().sub_(grad, alpha=gamma)

                state['momentum_buffer'] = momentum

        return loss

    def zero_grad(self):
        for group in self.param_groups:
            for param in group['params']:
                param.grad = None
                param.detach_().requires_grad_()


class Adam(Optimizer):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):

        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super().__init__(parameters, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group.get('betas', (0.9, 0.999))
            amsgrad = group.get('amsgrad', False)
            lam = group.get('weight_decay', 0)
            gamma = group.get('lr', 0)
            eps = group.get('eps', 1e-8)
            for param in group['params']:
                grad = param.grad
                if grad is None:
                    continue
                state = self.state[param]

                # Lazy state initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(param)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(param)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(param)

                step = state['step'] + 1
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                max_exp_avg_sq = state.get('max_exp_avg_sq', None)

                if lam:
                    grad = grad.add(param, alpha=lam)

                exp_avg.detach_().copy_(exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1))
                exp_avg_sq.detach_().copy_(exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2))

                bias_correction1 = 1 - math.pow(beta1, step)
                bias_correction2 = 1 - math.pow(beta2, step)
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                if amsgrad:
                    max_exp_avg_sq = torch.maximum(max_exp_avg_sq.clone(), exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().div(bias_correction2_sqrt).add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().div(bias_correction2_sqrt).add_(eps)

                step_size = -gamma / bias_correction1
                param.detach_().addcdiv_(exp_avg, denom, value=step_size)

                state['step'] = step
                state['exp_avg'] = exp_avg
                state['exp_avg_sq'] = exp_avg_sq
                state['max_exp_avg_sq'] = max_exp_avg_sq

        return loss

    def zero_grad(self):
        for group in self.param_groups:
            for param in group['params']:
                param.grad = None
                param.detach_().requires_grad_()
