import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required, _use_grad_for_differentiable
from typing import List, Optional
import time

__all__ = ['NISGD', 'sgd']

class NISGD(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False, foreach: Optional[bool] = None,
                 differentiable=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach,
                        differentiable=differentiable)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(NISGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            #params_name = []
            d_p_list = []
            momentum_buffer_list = []
            momentum_input_list = []
            state_steps = []
            acc_deltas = []
            has_sparse_grad = False

            for i,p in enumerate(group['params']):
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    #params_name.append(group['names'][i])
                    if p.grad.is_sparse:
                        has_sparse_grad = True

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])
                    if 'momentum_input' not in state:
                        momentum_input_list.append(None)
                    else:
                        momentum_input_list.append(state['momentum_input'])
                    

            sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                momentum_input_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'])

            # update momentum_buffers in state
            for p, momentum_buffer, momentum_input in zip(params_with_grad, momentum_buffer_list, momentum_input_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer
                state['momentum_input'] = momentum_input

        return loss

def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        momentum_input_list,
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        has_sparse_grad: bool = None,
        foreach: bool = None,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgd
    else:
        func = _single_tensor_sgd

    func(params,
         d_p_list,
         momentum_buffer_list,
         momentum_input_list,
         weight_decay=weight_decay,
         momentum=momentum,
         lr=lr,
         dampening=dampening,
         nesterov=nesterov,
         has_sparse_grad=has_sparse_grad,
         maximize=maximize)
    
def lp_norm(inp, p, sz, keepdim=True):
    if sz:
        if p == 1:
            return torch.sum(torch.abs(inp), dim=1,keepdim=keepdim)
        else:
            return torch.pow(torch.sum(torch.pow(inp,p),1, keepdim=keepdim),1)
            #return torch.sum(torch.pow(inp,p),1, keepdim=True)
    else:
        if p == 1:
            return torch.sum(torch.abs(inp), dim=(-1,-2),keepdim=keepdim)
        else:
            return torch.pow(torch.sum(torch.pow(inp,p),(-1,-2), keepdim=keepdim),1)
            #return torch.sum(torch.pow(inp,p),(-1,-2), keepdim=True)

def mean_std_calc(inp, sz, keepdim=True):
    if sz:
        mean_inp = torch.mean(inp, dim=1, keepdim=keepdim)
        std_inp = torch.nan_to_num(torch.std(inp, dim=1, keepdim=keepdim), nan=0.99) 
        return [mean_inp, std_inp]
    else:
        mean_inp = torch.mean(inp, dim=(-1,-2), keepdim=keepdim)
        std_inp = torch.nan_to_num(torch.std(inp, dim=(-1,-2), keepdim=keepdim), nan=0.99) 
        return [mean_inp, std_inp]

def center_norm(norm_inp, mean_inp, std_inp):
    return torch.div(torch.sub(norm_inp, mean_inp), torch.clamp(std_inp,min=0.01))

def _single_tensor_sgd(params: List[Tensor],
                       d_p_list: List[Tensor],
                       momentum_buffer_list: List[Optional[Tensor]],
                       momentum_input_list,
                       *,
                       weight_decay: float,
                       momentum: float,
                       lr: float,
                       dampening: float,
                       nesterov: bool,
                       maximize: bool,
                       has_sparse_grad: bool):
    
    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]
        
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)
        
        if param.input_to_norm is not None:
            inp = param.input_to_norm

            if isinstance(inp, list):
                group_flag = inp[1]
                inp_c = inp[0]

                [mean_inp, std_inp] = mean_std_calc(inp_c, False)
                inp_c = center_norm(inp_c, mean_inp, std_inp)
                norm_input = lp_norm(inp_c, 2, False)
                norm_input = torch.mean(norm_input, dim=0, keepdim=True)

                buf_input = momentum_input_list[i]
                if buf_input is None:
                    buf_input = torch.zeros_like(norm_input, memory_format=torch.preserve_format)
                buf_input.mul_(momentum).add_(norm_input, alpha=1-dampening)
                norm_input = buf_input
                norm_inp = torch.clamp(torch.log(norm_input),min=0.01)
                if group_flag:
                    norm_inp = torch.permute(norm_inp, [1,0,2,3])
                    
                if momentum != 0:
                    buf = momentum_buffer_list[i]
                
                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        momentum_buffer_list[i] = buf
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                d_p = d_p.div_(norm_inp)

            else:
                [mean_inp, std_inp] = mean_std_calc(inp, True)
                inp = center_norm(inp, mean_inp, std_inp)
                norm_input = lp_norm(inp, 2, True)
                norm_input = torch.mean(norm_input, dim=0, keepdim=True)

                buf_input = momentum_input_list[i]
                if buf_input is None:
                    buf_input = torch.zeros_like(norm_input, memory_format=torch.preserve_format)
                buf_input.mul_(momentum).add_(norm_input, alpha=1-dampening)
                norm_input = buf_input
                norm_inp = torch.clamp(torch.log(norm_input),min=0.01)
                
                if momentum != 0:
                    buf = momentum_buffer_list[i]
                
                    if buf is None:
                        buf = torch.clone(d_p).detach()
                        momentum_buffer_list[i] = buf
                    else:
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                d_p = d_p.div_(norm_inp)

                
        else:
        
            if momentum != 0:
                buf = momentum_buffer_list[i]
            
                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
        
                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf
        
        param.add_(d_p, alpha=-lr)


def _multi_tensor_sgd(params: List[Tensor],
                      grads: List[Tensor],
                      momentum_buffer_list: List[Optional[Tensor]],
                      *,
                      weight_decay: float,
                      momentum: float,
                      lr: float,
                      dampening: float,
                      nesterov: bool,
                      maximize: bool,
                      has_sparse_grad: bool):

    if len(params) == 0:
        return
    
    if has_sparse_grad is None:
        has_sparse_grad = any(grad.is_sparse for grad in grads)

    if maximize:
        grads = torch._foreach_neg(tuple(grads))  # type: ignore[assignment]

    if weight_decay != 0:
        grads = torch._foreach_add(grads, params, alpha=weight_decay)

    if momentum != 0:
        bufs = []

        all_states_with_momentum_buffer = True
        for i in range(len(momentum_buffer_list)):
            if momentum_buffer_list[i] is None:
                all_states_with_momentum_buffer = False
                break
            else:
                bufs.append(momentum_buffer_list[i])

        if all_states_with_momentum_buffer:
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, grads, alpha=1 - dampening)
        else:
            bufs = []
            for i in range(len(momentum_buffer_list)):
                if momentum_buffer_list[i] is None:
                    buf = momentum_buffer_list[i] = torch.clone(grads[i]).detach()
                else:
                    buf = momentum_buffer_list[i]
                    buf.mul_(momentum).add_(grads[i], alpha=1 - dampening)

                bufs.append(buf)

        if nesterov:
            torch._foreach_add_(grads, bufs, alpha=momentum)
        else:
            grads = bufs

    if not has_sparse_grad:
        torch._foreach_add_(params, grads, alpha=-lr)
    else:
        # foreach APIs dont support sparse
        for i in range(len(params)):
            params[i].add_(grads[i], alpha=-lr)