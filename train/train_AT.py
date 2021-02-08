#!/usr/bin/env python

# numpy package
import numpy as np

# torch package
import torch
import torchvision
from torch.nn.functional import cross_entropy

# basic package
import os
import sys
sys.path.append('.')
import argparse
from tqdm import tqdm
from datetime import datetime

# custom package
from loader.argument_print import argument_print
from loader.loader import dataset_loader, network_loader, attack_loader

# cudnn enable
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# argument parser
parser = argparse.ArgumentParser(description='Joint Adversarial Defense')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--steps', default=10, type=int, help='adv. steps')
parser.add_argument('--eps', required=True, type=float, help='max norm')
parser.add_argument('--dataset', required=True, type=str, help='dataset name')
parser.add_argument('--network', required=True, type=str, help='network name')
parser.add_argument('--data_root', required=True, type=str, help='path to dataset')
parser.add_argument('--epoch', default=60, type=int, help='epoch number')
parser.add_argument('--attack', default='pgd', type=str, help='attack type')
parser.add_argument('--save_dir', default='./experiment', type=str, help='save directory')
args = parser.parse_args()

# loading dataset, network, and attack
trainloader, testloader = dataset_loader(args)
net = network_loader(args, mean=args.mean, std=args.std).cuda()
attack = attack_loader(args, net)

# Adam Optimizer with KL divergence, and Scheduling Learning rate
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

# Setting checkpoint date time
date_time = datetime.today().strftime("%m%d%H%M")

# checkpoint_name
checkpoint_name = 'AT_'+args.network+'_'+args.dataset+'_'+date_time+'.pth'

# argument print
argument_print(args, checkpoint_name)


def train():

    # Modeling Adversarial Loss
    for epoch in range(args.epoch):

        # train environment
        net.train()

        print('\n\n[AT/Epoch] : {}'.format(epoch+1))

        total_cross_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):

            # dataloader parsing and generate adversarial examples
            inputs, targets = inputs.cuda(), targets.cuda()

            # learning network parameters
            optimizer.zero_grad()
            adv_x = attack(inputs, targets) if args.eps != 0 else inputs
            cross_entropy_loss = cross_entropy(net(adv_x), targets)
            cross_entropy_loss.backward()
            optimizer.step()

            # validation
            pred = torch.max(net(adv_x).detach(), dim=1)[1]
            correct += torch.sum(pred.eq(targets)).item()
            total += targets.numel()

            # logging two types loss and total loss
            total_cross_loss += cross_entropy_loss.item()

            if batch_idx % 50 == 0 and batch_idx != 0:
                print('[AT/Train] Iter: {}, Acc: {:.3f}, CE: {:.3f}'.format(
                    batch_idx, # Iter
                    100.*correct / total, # Acc
                    total_cross_loss / (batch_idx+1) # CrossEntropy
                    )
                )

        # Scheduling learning rate by stepLR
        scheduler.step()

        # Adversarial validation
        adversarial_test()

        # Save checkpoint file
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_cross_entropy_loss' : total_cross_loss / (batch_idx+1)
            }, os.path.join(args.save_dir,checkpoint_name))

        # argument print
        argument_print(args, checkpoint_name)


def adversarial_test():

    correct = 0
    total = 0

    print('\n\n[AT/Test] Under Testing ... Wait PLZ')
    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):

        # dataloader parsing and generate adversarial examples
        inputs, targets = inputs.cuda(), targets.cuda()
        adv_x = attack(inputs, targets) if args.eps != 0 else inputs

        # Evaluation
        outputs = net(adv_x).detach()

        # Test
        pred = torch.max(outputs, dim=1)[1]
        correct += torch.sum(pred.eq(targets)).item()
        total += targets.numel()

    print('[AT/Test] Acc: {:.3f}'.format(100.*correct / total))


if __name__ == "__main__":
    train()