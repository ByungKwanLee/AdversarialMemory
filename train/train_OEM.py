#!/usr/bin/env python

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
from loader.loader import dataset_loader, OEM_network_loader, attack_loader

# cudnn enable
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# argument parser
parser = argparse.ArgumentParser(description='Joint Adversarial Defense')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--steps', default=10, type=int, help='adv. steps')
parser.add_argument('--eps', default=0.03, type=float, help='max norm')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--network', default='wide', type=str, help='network name')
parser.add_argument('--data_root', default='./datasets', type=str, help='path to dataset')
parser.add_argument('--epoch', default=60, type=int, help='epoch number')
parser.add_argument('--attack', default='pgd', type=str, help='attack type')
parser.add_argument('--save_dir', default='./experiment', type=str, help='save directory')
parser.add_argument('--prior', default=0, type=int, help='load pre-trained prior')
parser.add_argument('--prior_datetime', default='01051206', type=str, help='checkpoint datetime')
args = parser.parse_args()

# loading dataset, network, and attack
trainloader, testloader = dataset_loader(args)
# loading pre-trained prior
if args.prior:
    args.isinitialize=False
    net = OEM_network_loader(args, mean=args.mean, std=args.std).cuda()
    net.load_state_dict(torch.load(os.path.join(args.save_dir,
                            'Plain_'+args.network+'_'+args.dataset+'_'+args.prior_datetime+'.pth'))['model_state_dict'], strict=False)
    net.eval()
    print('[OEM] ' + 'Plain_'+args.network+'_'+args.dataset+'_'+args.prior_datetime+'.pth' +' is Successfully Loaded')

attack = attack_loader(args, net)

# Adam Optimizer with KL divergence, and Scheduling Learning rate
optimizer = torch.optim.SGD(net.OEM.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

# Setting checkpoint date time
date_time = datetime.today().strftime("%m%d%H%M")

# checkpoint_name
checkpoint_name = 'OEM_'+args.network+'_'+args.dataset+'_'+date_time+'.pth'

# argument print
argument_print(args, checkpoint_name)


def train():

    # Modeling Adversarial Loss
    for epoch in range(args.epoch):

        print('\n\n[OEM/Epoch] : {}'.format(epoch+1))

        total_loss = 0
        total_memory_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):

            # dataloader parsing and generate adversarial examples
            inputs, targets = inputs.cuda(), targets.cuda()

            # learning network parameters
            optimizer.zero_grad()
            net.OEM.eval()
            adv_x = attack(inputs, targets) if args.eps != 0 else inputs
            net.OEM.train()
            logit, logit_, memory_loss, memory_loss_ = net.get_training_resource(inputs, adv_x)
            Total_loss = cross_entropy(logit, targets) + cross_entropy(logit_, targets) + memory_loss + memory_loss_
            (0.5*Total_loss).backward()
            optimizer.step()


            # validation
            net.OEM.eval()
            pred = torch.max(net.get_inference(adv_x), dim=1)[1]
            correct += torch.sum(pred.eq(targets)).item()
            total += targets.numel()

            # logging two types loss and total loss
            total_loss += Total_loss.item()
            total_memory_loss += 0.5*(memory_loss+memory_loss_).item()

            if batch_idx % 10 == 0 and batch_idx != 0:
                print('[OEM/Train] Iter: {}, Acc: {:.2f}, Loss: {:.2f}={:.2f}+{:.5f}'.format(
                    batch_idx, # Iter
                    100.*correct / total, # Acc
                    total_loss / (batch_idx+1), # CrossEntropy+MemoryLoss
                    (total_loss-total_memory_loss) / (batch_idx+1), # CrossEntropy
                    total_memory_loss / (batch_idx+1) # MemoryLoss
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
            'total_loss' : total_loss / (batch_idx+1)
            }, os.path.join(args.save_dir,checkpoint_name))

        # argument print
        argument_print(args, checkpoint_name)


def adversarial_test():
    correct = 0
    total = 0
    print('\n[OEM/Test] Under Testing ... Wait PLZ')
    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):

        # dataloader parsing and generate adversarial examples
        inputs, targets = inputs.cuda(), targets.cuda()
        adv_x = attack(inputs, targets) if args.eps != 0 else inputs

        # Evaluation
        outputs = net.get_inference(adv_x)

        # Test
        pred = torch.max(outputs, dim=1)[1]
        correct += torch.sum(pred.eq(targets)).item()
        total += targets.numel()

    print('[OEM/Test] Acc: {:.3f}'.format(100.*correct / total))


if __name__ == "__main__":
    train()