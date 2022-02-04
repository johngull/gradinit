"""
Copyright (c) 2022 Vitaly Bondar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
"""

import torch
import typing


class GradInitWrapper:
    """
    Wrapper for the PyTorch model for gradient initialization.
    This wrapper change model parameters and recover model structure after wrapper detachment (including wrapper deleting).
    For later optimization of the model, it is important to create an optimizer after detaching GradInitWrapper.

    :param module: (torch.nn.Module) torch module to initialize
    :param reinit_zeros: (bool) if True, fully zero parameters, like bias will be reinitialized with the normal distribution
    :param reinit_std: std value for zero parameters re-initializations, used only if reinit_zeros=True
    """

    _postfix_scale = "_gradinit_scale_factor"
    _postfix_data = "_gradinit_data"

    def __init__(
        self,
        module: torch.nn.Module,
        reinit_zeros: bool = False,
        reinit_std: float = 1e-2,
    ):
        self.hooks = []
        self.modules_parameter_names = []
        self.old_parameters = []
        self._register_modules(module, reinit_zeros, reinit_std)
        self.module = module

    def __del__(self):
        self.detach()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.detach()

    def _make_hook(
        self, name: str
    ) -> typing.Callable[[torch.nn.Module, typing.Any], None]:
        """
        Generate pre-forward hook function for exact parameter

        :param name: parameter name
        :return: hook function
        """

        def hook(module, _):
            v: torch.Tensor = getattr(module, name + self._postfix_scale) * getattr(
                module, name + self._postfix_data
            )
            setattr(module, name, v)

        return hook

    def _register_modules(
        self, module: torch.nn.Module, reinit_zeros: bool, reinit_std: float
    ):
        """
        Handle modules parameters in recursive ways.
        For each parameter additional scale parameter registered, hooks registered.
        Also, optionally fully zero parameters, like bias will be reinitialized with the normal distribution.

        :param module: (torch.nn.Module) torch module to handle
        :param reinit_zeros: (bool) if True, fully zero parameters, like bias will be reinitialized with the normal distribution
        :param reinit_std: std value for zero parameters re-initializations, used only if reinit_zeros=True
        """

        for submodule in module.children():
            self._register_modules(submodule, reinit_zeros, reinit_std)

        named_parameters = list(module.named_parameters(recurse=False))
        for name, p in named_parameters:
            self.modules_parameter_names.append((module, name))
            t = p
            delattr(module, name)
            setattr(module, name, t.data)
            data = t.data.clone()
            if reinit_zeros and torch.allclose(
                data, torch.tensor([0.0], device=data.device, dtype=data.dtype)
            ):
                data = torch.randn_like(data) * reinit_std
            data.requires_grad = True
            setattr(module, name + self._postfix_data, data)
            self.old_parameters.append(data)
            setattr(
                module,
                name + self._postfix_scale,
                torch.nn.Parameter(torch.tensor(1.0, device=t.device, dtype=t.dtype)),
            )
            hook = module.register_forward_pre_hook(self._make_hook(name))
            self.hooks.append(hook)

    def _combined_parameters(self) -> typing.List[torch.Tensor]:
        """
        Collect a list of the rescaled parameters tensors. Used for calculating gradinit loss

        :return: (List[torch.Tensor]) list of the rescaled parameters tensors
        """

        return [
            getattr(module, name)
            for module, name in self.modules_parameter_names
            if getattr(module, name).requires_grad
        ]

    def grad_loss(
        self,
        loss: torch.Tensor,
        norm: int = 1,
        gamma: float = 1.0,
        scaler: typing.Optional[torch.cuda.amp.GradScaler] = None,
    ) -> torch.Tensor:
        """
        Calculate loss for the parameters scaling optimization.

        :param loss: (torch.Tensor) original task loss
        :param norm: (int) order of norm to use for loss calculation. Use 1 for Adam and 2 for SGD
        :param gamma: (float) minimal threshold to use special init loss
        :param scaler: torch amp GradScaler used in your train loop for mixed-precision
        :return: (torch.Tensor) loss value
        """

        if scaler:
            all_grads = torch.autograd.grad(
                scaler.scale(loss),
                self._combined_parameters(),
                create_graph=True,
                allow_unused=True,
            )
            all_grads = list(
                filter(
                    lambda x: x is not None and torch.all(torch.isfinite(x)), all_grads
                )
            )
            inv_scale = 1.0 / scaler.get_scale()
            all_grads = [p * inv_scale for p in all_grads]
            with torch.cuda.amp.autocast():
                gnorm = (
                    torch.stack([g.abs().pow(norm).sum() for g in all_grads])
                    .sum()
                    .pow(1.0 / norm)
                )
        else:
            all_grads = torch.autograd.grad(
                loss, self._combined_parameters(), create_graph=True, allow_unused=True
            )
            all_grads = list(filter(lambda x: x is not None, all_grads))
            gnorm = (
                torch.stack([g.abs().pow(norm).sum() for g in all_grads])
                .sum()
                .pow(1.0 / norm)
            )

        if gnorm > gamma:
            return gnorm
        else:
            return loss

    def clamp_scales(self, min_scale: float = 0.01):
        """
        Clamp parameters scale to prevent zeroing of the parameters

        :param min_scale: lower threshold for the scaling factors
        """
        for module, name in self.modules_parameter_names:
            p = getattr(module, name + self._postfix_scale)
            p.data = torch.clamp(p.data, min=min_scale)

    def detach(self):
        """
        Recover initial state of the model
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

        self.old_parameters.clear()
        for module, name in self.modules_parameter_names:
            t = getattr(module, name + self._postfix_scale).data * getattr(
                module, name + self._postfix_data
            )
            delattr(module, name)
            delattr(module, name + self._postfix_scale)
            delattr(module, name + self._postfix_data)
            setattr(module, name, torch.nn.Parameter(t))
        self.modules_parameter_names.clear()
