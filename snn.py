"""Post-add LIF Spiking ResNet-18 with direct static input over T steps."""
import importlib.metadata

import torch
from torch import nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate
from spikingjelly.activation_based.model import spiking_resnet


class LIFResNet18(nn.Module):
    def __init__(self, initial_ann, time_steps=4, tau=2.0):
        super().__init__()
        if time_steps < 1 or tau <= 1:
            raise ValueError('time_steps >= 1 and tau > 1 are required')
        self.time_steps = time_steps
        # Preserve the ANN initialization and following DataLoader RNG sequence.
        rng_state = torch.get_rng_state()
        self.backbone = spiking_resnet.spiking_resnet18(
            pretrained=False, num_classes=10, spiking_neuron=neuron.LIFNode,
            tau=tau, decay_input=True, v_threshold=1.0, v_reset=0.0,
            surrogate_function=surrogate.ATan(alpha=2.0), detach_reset=True,
        )
        self.backbone.conv1 = layer.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        functional.set_step_mode(self.backbone, 'm')
        functional.set_backend(self.backbone, 'torch', instance=neuron.LIFNode)
        # This is fresh random ANN initialization, never the trained ANN checkpoint.
        self.backbone.load_state_dict(initial_ann.state_dict(), strict=True)
        torch.set_rng_state(rng_state)
        self.spec = {
            'spikingjelly': importlib.metadata.version('spikingjelly'),
            'neuron': 'LIFNode', 'time_steps': time_steps, 'tau': tau,
            'decay_input': True, 'v_threshold': 1.0, 'v_reset': 0.0,
            'detach_reset': True, 'surrogate': 'ATan(alpha=2.0)',
            'step_mode': 'm', 'backend': 'torch',
            'encoding': 'same normalized static image repeated T times',
            'readout': 'mean of linear logits across time',
            'residual': 'conv-bn-lif-conv-bn-add-lif (standard SpikingResNet, not SEW)',
            'lif_nodes': sum(isinstance(m, neuron.LIFNode) for m in self.backbone.modules()),
            'initialization': 'identical to randomly initialized ANN ResNet-18 at same seed',
        }

    def reset_state(self):
        functional.reset_net(self.backbone)

    def forward(self, x):
        self.reset_state()
        sequence = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)
        return self.backbone(sequence).mean(0)
