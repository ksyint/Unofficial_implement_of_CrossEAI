import torch
import torch.nn.functional as F
from torchvision import models

class GradCAM:
    def __init__(self, model, target_layer_names):
        self.model = model
        self.target_layer_names = target_layer_names
        self.gradients = []

        self._register_hooks()

    def _register_hooks(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients.append(grad_out[0])

        for name, module in self.model.named_modules():
            if name in self.target_layer_names:
                module.register_backward_hook(hook_function)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        output = self.forward(input)
        if index is None:
            index = torch.argmax(output)
        self.model.zero_grad()
        target = output[0][index]
        target.backward()

        gradients = self.gradients[-1].detach().cpu().numpy()[0]
        return gradients
