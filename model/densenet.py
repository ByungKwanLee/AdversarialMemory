import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List



class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input):
        # type: (List[Tensor]) -> Tensor
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (List[Tensor]) -> (Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input):
        # type: (Tensor) -> (Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=100, memory_efficient=False, in_channels=3, mean=0.5, std=0.25):

        super(DenseNet, self).__init__()

        # LBK EDIT
        self.mean = mean
        self.std = std
        self.in_channels = in_channels

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(self.in_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.num_features = num_features
        self.classifier = nn.Linear(num_features, num_classes)

        # initialize weights
        self.initialize_weights()


    def initialize_weights(self):
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = (x-self.mean)/self.std
        features = self.features(out)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out



class DenseNet_Plain(DenseNet):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=100, memory_efficient=False, in_channels=3, mean=0.5, std=0.25):
        super(DenseNet_Plain, self).__init__(growth_rate, block_config,
                 num_init_features, bn_size, drop_rate, num_classes, memory_efficient, in_channels, mean, std)
        self.initialize_weights()

    @staticmethod
    def get_epistemic(logit_tensors):
        class_number = logit_tensors.shape[-1]
        max_prob = 1/class_number
        max_norm = -math.log2(max_prob+1e-10)

        if len(logit_tensors.shape)==2:
            # logit_tensors: [batch, class]
            # output: predictive entropy
            soft_logit = F.softmax(logit_tensors, dim=1)
            return 1/max_norm * (-soft_logit*((soft_logit+1e-10).log2())).sum(dim=1).unsqueeze(1)


        # logit_tensors: [num_ensemble, batch, class]
        # output : predictive entropy
        soft_logit = F.softmax(logit_tensors, dim=2).mean(dim=0)
        return 1/max_norm * (-soft_logit*((soft_logit+1e-10).log2())).sum(dim=1).unsqueeze(1)


    def get_inference(self, x, num_ensemble=10, check_uncertainty=False):

        for i in range(num_ensemble):
            z = self(x)
            z_tensors = torch.cat([z_tensors, z.unsqueeze(0)], dim=0) if i else z.unsqueeze(0)
        P_clean = self.get_epistemic(z_tensors)

        if check_uncertainty:
            return z_tensors.mean(dim=0).detach(), P_clean.detach()

        return z_tensors.detach(), P_clean.detach()



class DenseNet_KALMAN(DenseNet):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=100, memory_efficient=False, in_channels=3, mean=0.5, std=0.25):
        super(DenseNet_KALMAN, self).__init__(growth_rate, block_config,
                 num_init_features, bn_size, drop_rate, num_classes, memory_efficient, in_channels, mean, std)
        self.classifier = nn.Linear(self.num_features, self.num_classes)
        self.classifier_v = nn.Linear(self.nChannels, self.num_classes)
        self.classifier_eps = nn.Linear(self.nChannels, self.num_classes)
        self.initialize_weights()

    def forward(self, x, flag=False):
        out = (x-self.mean) / self.std
        out = self.conv1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)

        z_adv = self.classifier(out)
        v = self.classifier_v(out)
        z_tilde = self.classifier_eps(out)

        if not flag:

            P_adv = self.get_epistemic(z_adv)
            P_clean = self.get_epistemic(z_adv+z_tilde+v)
            K = P_adv / (P_adv+v**2+P_clean+1e-10)
            return z_adv+K*(z_tilde+v)

        return z_adv, v, z_tilde


    @staticmethod
    def get_epistemic(logit_tensors):

        class_number = logit_tensors.shape[-1]
        max_prob = 1/class_number
        max_norm = -math.log2(max_prob+1e-10)

        if len(logit_tensors.shape)==2:
            # logit_tensors: [batch, class]
            # output: predictive entropy
            soft_logit = F.softmax(logit_tensors, dim=1)
            return 1/max_norm * (-soft_logit*((soft_logit+1e-10).log2())).sum(dim=1).unsqueeze(1)

        # logit_tensors: [num_ensemble, batch, class]
        # output : predictive entropy
        soft_logit = F.softmax(logit_tensors, dim=2).mean(dim=0)
        return 1/max_norm * (-soft_logit*((soft_logit+1e-10).log2())).sum(dim=1).unsqueeze(1)

    @staticmethod
    def get_cs(input, target, function=False):

        if function:
            return F.cosine_similarity(input, target, dim=len(input.shape)-1).unsqueeze(len(input.shape)-1)

        input_ = input.unsqueeze(len(input.shape)-1)
        target_ = target.unsqueeze(len(target.shape))
        return torch.matmul(input_, target_).squeeze(len(target.shape))


    def get_pre_parameters(self, x, num_ensemble):

        for i in range(num_ensemble):

            z, v, eps = self(x, flag=True)
            # z_tensors : [ensemble, batch, class]
            # M_tensors : [ensemble, batch, class]
            z_tensors = torch.cat([z_tensors, z.unsqueeze(0)], dim=0) if i else z.unsqueeze(0)
            v_tensors = torch.cat([v_tensors, v.unsqueeze(0)], dim=0) if i else v.unsqueeze(0)
            e_tensors = torch.cat([e_tensors, eps.unsqueeze(0)], dim=0) if i else eps.unsqueeze(0)

        # P_adv : [batch, 1]
        P_adv = self.get_epistemic(z_tensors).detach()

        return z_tensors, v_tensors, e_tensors, P_adv



    def get_training(self, optimizer, x_adv, targets, z_clean, P_clean, num_ensemble):

        z_tensors, v_tensors, e_tensors, P_adv = self.get_pre_parameters(x_adv, num_ensemble=num_ensemble)

        # Assumption loss
        P_clean_approx = self.get_epistemic(z_tensors+e_tensors+v_tensors).detach()
        cs = self.get_cs(e_tensors, v_tensors, function=True).mean(dim=0)
        assumption_loss = (1/2*(v_tensors.mean(dim=0)-(z_clean-z_tensors-e_tensors).detach())**2).mean()\
                            +F.mse_loss(e_tensors.mean(dim=0), torch.zeros(e_tensors.mean(dim=0).shape).cuda())

        # F.mse_loss(M_tensors, z_clean-z_tensors.detach()-e_tensors.detach())
        # Get Kalman Gain
        # Linear fusion inference, minimizing uncertainty
        K = P_adv / (P_adv+v_tensors.mean(dim=0)**2+P_clean+1e-10)
        z_update = z_tensors + K*(e_tensors+v_tensors)
        P_update = self.get_epistemic(z_update).detach()
        z_mean_update = z_update.mean(dim=0)

        # Cross entropy loss
        cross_entropy_loss = F.cross_entropy(z_mean_update, targets)

        # Learining procedure
        optimizer.zero_grad()
        total_loss = cross_entropy_loss + assumption_loss
        total_loss.backward()
        optimizer.step()

        return total_loss, [P_adv.mean().item(), K.mean().item(), P_update.mean().item(), P_clean_approx.mean().item(),\
                                                v_tensors.mean(dim=0).mean().item(), cs.mean().item(), e_tensors.mean(dim=0).mean().item()]


    def get_inference(self, x_adv, num_ensemble, check_uncertainty=False, training=False):

        z_tensors, v_tensors, e_tensors, P_adv = self.get_pre_parameters(x_adv, num_ensemble=num_ensemble)

        cs = self.get_cs(v_tensors, e_tensors, function=True).mean(dim=0)
        P_clean = self.get_epistemic(z_tensors+e_tensors+v_tensors).detach()

        K = P_adv/(P_adv+v_tensors.mean(dim=0)**2+P_clean+1e-10)
        z_update = z_tensors + K*(e_tensors+v_tensors)
        P_update = self.get_epistemic(z_update).detach()

        if training:
            return z_update.mean(dim=0), P_update


        if check_uncertainty:
            return z_update.mean(dim=0), z_tensors.mean(dim=0), (z_tensors+e_tensors+v_tensors).mean(dim=0), [P_adv.mean(), K.mean().detach().item(), P_update.mean().detach().item(), P_clean.mean().detach().item(), v_tensors.mean(dim=0).mean().detach().item(), cs.mean().item(), e_tensors.mean(dim=0).mean().detach().item()]
        else:
            return z_update.mean(dim=0)

