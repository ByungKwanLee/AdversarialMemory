#!/usr/bin/env python

# os package
import os

# math package
import math

# numpy package
import numpy as np

# torch package
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

# Custom package
from model.wideresnet import WideResNet_Plain, WideResNet_OEM

# torchattacks toolbox
import torchattacks

# art toolbox
from art.attacks.evasion import CarliniL2Method, ElasticNet, HopSkipJump, BoundaryAttack, SquareAttack
from art.estimators.classification import PyTorchClassifier


def attack_loader(args, net):

    # Gradient Clamping based Attack
    if args.attack == "pgd":
        return torchattacks.PGD(model=net, eps=args.eps,
                                    alpha=args.eps/args.steps*2.3, steps=args.steps, random_start=True)

    elif args.attack == "auto":
        return torchattacks.APGD(model=net, eps=args.eps)

    elif args.attack == "fab":
        return torchattacks.FAB(model=net, eps=args.eps, n_classes=args.n_classes)

    elif args.attack == "cw":
        return torchattacks.CW(model=net, c=0.1, lr=0.1, steps=200)

    elif args.attack == "fgsm":
        return torchattacks.FGSM(model=net, eps=args.eps)

    elif args.attack == "bim":
        return torchattacks.BIM(model=net, eps=args.eps, alpha=1/255)

    elif args.attack == "deepfool":
        return torchattacks.DeepFool(model=net, steps=10)

    elif args.attack == "sparse":
        return torchattacks.SparseFool(model=net)

    elif args.attack == "gn":
        return torchattacks.GN(model=net, sigma=args.eps)



def network_loader(args, mean, std):
    if args.network == "wide":
        print('Wide Network')
        return WideResNet_Plain(depth=16, in_channels=args.channel, num_classes=args.n_classes, widen_factor=8, dropRate=0.3, mean=mean, std=std)

def OEM_network_loader(args, mean, std):
    if args.network == "wide":
        print('Wide OEM Network')
        print('PyCharm has been successfully uploaded')
        return WideResNet_OEM(depth=16, in_channels=args.channel, num_classes=args.n_classes, isinitialize=args.isinitialize, widen_factor=8, dropRate=0.3, mean=mean, std=std)


def dataset_loader(args):

    args.mean=0.5
    args.std=0.25

    # Setting Dataset Required Parameters
    if args.dataset   == "svhn":
        args.n_classes = 10
        args.img_size  = 32
        args.channel   = 3
    elif args.dataset == "stl":
        args.n_classes = 10
        args.img_size  = 96
        args.channel   = 3
    elif args.dataset == "cifar10":
        args.n_classes = 10
        args.img_size  = 32
        args.channel   = 3
    elif args.dataset == "cifar100":
        args.n_classes = 100
        args.img_size  = 32
        args.channel   = 3
    elif args.dataset == "tiny":
        args.n_classes = 200
        args.img_size  = 64
        args.channel   = 3

    transform_train = transforms.Compose(
        [transforms.Pad(4, padding_mode="reflect"),
         transforms.RandomCrop(args.img_size),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()]
    )


    transform_test = transforms.Compose(
        [
        transforms.ToTensor()]
    )


    args.batch_size = 100

    # Full Trainloader/Testloader
    trainloader = torch.utils.data.DataLoader(dataset(args, True,  transform_train), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    testloader  = torch.utils.data.DataLoader(dataset(args, False, transform_test),  batch_size=args.batch_size, shuffle=True, pin_memory=True)

    return trainloader, testloader


def dataset(args, train, transform):

        if args.dataset == "cifar10":
            return torchvision.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train)

        elif args.dataset == "cifar100":
            return torchvision.datasets.CIFAR100(root=args.data_root, transform=transform, download=True, train=train)

        elif args.dataset == "stl":
            return torchvision.datasets.STL10(root=args.data_root, transform=transform, download=True,
                                    split="train" if train else "test")

        elif args.dataset == "svhn":
            return torchvision.datasets.SVHN(root=args.data_root, transform=transform, download=True,
                                    split="train" if train else "test")
        elif args.dataset == "tiny":
            return torchvision.datasets.ImageFolder(root=args.data_root+'/tiny-imagenet-200/train' if train \
                                    else args.data_root + '/tiny-imagenet-200/val', transform=transform)