'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'VGG8b': [128, 256, 'M', 256, 512, 'M', 512, 'M', 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # 'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # 'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    # 'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


''' [1] NORMAL VGG '''
class VGG(nn.Module):

    def __init__(self, vgg_name, num_classes, img_width, in_channels, mean, std):
        super(VGG, self).__init__()

        self.in_channels = in_channels
        self.mean = mean
        self.std = std

        self.features = self._make_layers(cfg[vgg_name], batch_norm=False)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def forward(self, y):
        y = (y-self.mean)/self.std
        x = self.features(y)
        z = x+0.03*(self.features(y+1e-10)-self.features(y))/1e-10
        print(z.shape)
        exit(0)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


    def get_inference(self, x, num_ensemble=10):
        for i in range(num_ensemble):
            z = self(x)
            z_tensors = torch.cat([z_tensors, z.unsqueeze(0)], dim=0) if i else z.unsqueeze(0)
        z_clean = z_tensors[-1]
        P_clean_epistemic = z_tensors.var(dim=0)
        return z_clean, P_clean_epistemic


    def get_ensemble(self, x, num_ensemble=10):

        for i in range(num_ensemble):
            z = self(x)
            z_tensors = torch.cat([z_tensors, z.unsqueeze(0)], dim=0) if i else z.unsqueeze(0)
        z_clean_mean = z_tensors.mean(dim=0)
        P_clean_epistemic = z_tensors.var(dim=0)
        return z_clean_mean, P_clean_epistemic


    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


''' [2] PROPOSED KALMAN VGG '''
class VGG_KALMAN(nn.Module):
    def __init__(self, vgg_name, num_classes, img_width, in_channels, mean, std):
        super(VGG_KALMAN, self).__init__()

        self.in_channels = in_channels
        self.mean = mean
        self.std = std

        self.features = self._make_layers(cfg[vgg_name], batch_norm=False)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.classifier_M = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.classifier_eps = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x, flag=False):
        x = (x-self.mean)/self.std
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        z = self.classifier(x)
        if not flag:
            return z
        M = self.classifier_M(x)
        eps = self.classifier_eps(x)
        return z, M, eps

    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    @staticmethod
    def get_epistemic(logit_tensors):
        # logit_tensors: [num_ensemble, batch, class]
        # output : predictive entropy
        soft_logit = F.softmax(logit_tensors, dim=2).mean(dim=0)
        return (-soft_logit*((soft_logit+1e-10).log2())).sum(dim=1).unsqueeze(1).detach()

    def get_pre_parameters(self, x, num_ensemble):

        for i in range(num_ensemble):

            z, M, eps = self(x, flag=True)
            # z_tensors : [ensemble, batch, class]
            # M_tensors : [ensemble, batch, class]
            z_tensors = torch.cat([z_tensors, z.unsqueeze(0)], dim=0) if i else z.unsqueeze(0)
            M_tensors = torch.cat([M_tensors, M.unsqueeze(0)], dim=0) if i else M.unsqueeze(0)
            e_tensors = torch.cat([e_tensors, eps.unsqueeze(0)], dim=0) if i else eps.unsqueeze(0)

        # P_adv_epistemic : [batch, 1]
        P_adv_epistemic = self.get_epistemic(z_tensors)

        return z_tensors, M_tensors, e_tensors, P_adv_epistemic



    def get_training(self, optimizer, x_adv, targets, z_clean, P_clean_epistemic, num_ensemble):

        z_tensors, M_tensors, e_tensors, P_adv_epistemic = self.get_pre_parameters(x_adv, num_ensemble=num_ensemble)

        # Assumption loss
        assumption_loss = F.mse_loss(M_tensors, (z_clean-z_tensors-e_tensors).detach()) \
                                            + F.mse_loss(e_tensors.mean(dim=0), torch.zeros(e_tensors.mean(dim=0).shape).cuda())

        # Get Kalman Gain
        # Linear fusion inference, minimizing uncertainty
        K = P_adv_epistemic / (P_adv_epistemic+M_tensors**2+P_clean_epistemic+1e-7)
        z_update = z_tensors + K*(e_tensors+M_tensors)
        P_update_epistemic = self.get_epistemic(z_update)

        # Cross entropy loss
        cross_entropy_loss = F.cross_entropy(z_update.mean(dim=0), targets)

        # Learining procedure
        optimizer.zero_grad()
        total_loss = cross_entropy_loss + assumption_loss
        total_loss.backward()
        optimizer.step()



        return total_loss, [P_adv_epistemic.mean().item(), K.mean().item(), P_update_epistemic.mean().item(), \
                                                M_tensors.mean(dim=0).mean().item(), e_tensors.mean(dim=0).mean().item()]



    def get_inference(self, x_adv, num_ensemble, check_uncertainty=False):

        z_tensors, M_tensors, e_tensors, P_adv_epistemic = self.get_pre_parameters(x_adv, num_ensemble=num_ensemble)

        if check_uncertainty:
            return z_tensors.mean(dim=0), P_adv_epistemic.mean()
        else:
            return z_tensors.mean(dim=0)



