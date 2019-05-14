import torch as t
import torch.nn as nn
import torch.optim as optim
import yaml

from collections import OrderedDict

opts = yaml.safe_load(open('./tracking/options.yaml','r'))

def append_params(params, module, prefix):
    for child in module.children():
        for k, p in child._parameters.items():
            if p is None:
                continue

            name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError('Duplicated param name: %s' % (name))


def set_optimizer_g(model_g, lr=opts['lr_g'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model_g.get_learnable_params()
    param_list = []
    for k, p in params.items():
        param_list.append(p)

    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


class NetG(nn.Module):
    def __init__(self):
        super(NetG, self).__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('fc1', nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(512 * 3 * 3, 256),
                                  nn.ReLU())),
            ('fc2', nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(256, 1 * 3 * 3)))]))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params

    def forward(self, x):
        #
        # forward model
        for _, module in self.layers.named_children():
            x = module(x)

        return x
