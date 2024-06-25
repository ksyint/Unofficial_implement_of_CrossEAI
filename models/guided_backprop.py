import torch
import torch.nn as nn
import numpy as np

class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.gradients = None

        def hook_function(module, grad_in, grad_out):
            if isinstance(module, nn.ReLU):
                return torch.clamp(grad_in[0], min=0.0),

        for module in self.model.modules():
            module.register_backward_hook(hook_function)

    def __call__(self, input, index=None):
        input.requires_grad = True
        output = self.model(input)
        if index is None:
            index = torch.argmax(output)
        self.model.zero_grad()
        target = output[0][index]
        target.backward()

        return input.grad[0].detach().cpu().numpy()
