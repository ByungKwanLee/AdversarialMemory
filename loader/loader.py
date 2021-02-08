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


class BoundaryAttack_(BoundaryAttack):

    def __init__(self, estimator, targeted=True, delta=0.01, epsilon=0.01, step_adapt=0.667, max_iter=5000,\
            num_trial=25, sample_size=20, init_size=100, batch_size=100, verbose=False):

        super(BoundaryAttack_, self).__init__(estimator=estimator,  targeted=targeted, delta=delta, epsilon=epsilon, step_adapt=step_adapt, max_iter=max_iter,\
                                            num_trial=num_trial, sample_size=sample_size, init_size=init_size, verbose=verbose)
        self.batch_size = batch_size



class HopSkipJump_(HopSkipJump):

    def __init__(self, classifier, targeted=False, norm=2, max_iter=50,\
            max_eval=10000, init_eval=100, init_size=100, batch_size = 100, verbose = True):

        super(HopSkipJump_, self).__init__(classifier=classifier, targeted=targeted, norm=norm, max_iter=max_iter, \
                                            max_eval=max_eval, init_eval=init_eval, init_size=init_size, verbose=verbose)
        self.batch_size = batch_size



def attack_loader(args, net):

    # Gradient Clamping based Attack
    if args.attack == "pgd":
        return torchattacks.PGD(model=net, eps=args.eps,
                                    alpha=args.eps/args.steps*2.3, steps=args.steps, random_start=True)
    elif args.attack == "tpgd":
        return torchattacks.TPGD(model=net, eps=args.eps,
                                    alpha=args.eps/args.steps*2.3, steps=args.steps)
    elif args.attack == "fgsm":
        return torchattacks.FGSM(model=net, eps=args.eps)
    elif args.attack == "bim":
        return torchattacks.BIM(model=net, alpha=args.eps/args.steps, steps=args.steps)
    elif args.attack == "gn":
        return torchattacks.GN(model=net, sigma=args.std)
    elif args.attack == 'eot':
        return torchattacks.APGD(model=net, eps=args.eps,
                                    alpha=args.eps/args.steps*2.3, steps=args.steps, sampling=5)


    # White-Box Attacks
    elif args.attack == 'cw':
        classifier = PyTorchClassifier(
            model=net,
            clip_values=(0, 1),
            loss=nn.CrossEntropyLoss(),
            input_shape=(args.channel, args.img_size, args.img_size),
            nb_classes=args.n_classes,
        )
        attack = CarliniL2Method(classifier=classifier, binary_search_steps=2, initial_const=100, max_iter=1, batch_size=args.batch_size*3, verbose=False)
        def f_attack(input, target):
            return torch.from_numpy(attack.generate(x=input.cpu(), y=target.cpu())).cuda()
        return f_attack
    elif args.attack == 'ead':
        classifier = PyTorchClassifier(
            model=net,
            clip_values=(0, 1),
            loss=nn.CrossEntropyLoss(),
            input_shape=(args.channel, args.img_size, args.img_size),
            nb_classes=args.n_classes,
        )
        attack = ElasticNet(classifier=classifier, binary_search_steps=2, initial_const=100, max_iter=1, batch_size=args.batch_size*3, verbose=False)
        def f_attack(input, target):
            return torch.from_numpy(attack.generate(x=input.cpu(), y=target.cpu())).cuda()
        return f_attack


    # Black-Box Attacks
    elif args.attack == "boundary":
        classifier = PyTorchClassifier(
            model=net,
            clip_values=(0, 1),
            loss=nn.CrossEntropyLoss(),
            input_shape=(args.channel, args.img_size, args.img_size),
            nb_classes=args.n_classes,
        )
        attack = BoundaryAttack_(estimator=classifier, batch_size=args.batch_size*3, max_iter=1000, verbose=False)
        def f_attack(input, target):
            adv_target = torch.randint(low=0, high=args.n_classes-1, size=target.shape).cuda()
            adv_target = adv_target + (adv_target >= target).float()
            adv_target = adv_target - (adv_target > args.n_classes-1).float()
            return torch.from_numpy(attack.generate(x=input.cpu(), y=adv_target.cpu())).cuda()
        return f_attack

    elif args.attack == "hop":
        classifier = PyTorchClassifier(
            model=net,
            clip_values=(0, 1),
            loss=nn.CrossEntropyLoss(),
            input_shape=(args.channel, args.img_size, args.img_size),
            nb_classes=args.n_classes,
        )
        attack = HopSkipJump_(classifier=classifier, max_iter=10, batch_size=args.batch_size*3, verbose=False)
        def f_attack(input, target):
            adv_target = torch.randint(low=0, high=args.n_classes-1, size=target.shape).cuda()
            adv_target = adv_target + (adv_target >= target).float()
            adv_target = adv_target - (adv_target > args.n_classes-1).float()
            return torch.from_numpy(attack.generate(x=input.cpu(), y=adv_target.cpu())).cuda()
        return f_attack

    elif args.attack == "square":
        classifier = PyTorchClassifier(
            model=net,
            clip_values=(0, 1),
            loss=nn.CrossEntropyLoss(),
            input_shape=(args.channel, args.img_size, args.img_size),
            nb_classes=args.n_classes,
        )
        attack = SquareAttack(estimator=classifier, eps=args.eps, nb_restarts=1, max_iter=1000, batch_size=args.batch_size, verbose=False)
        def f_attack(input, target):
            return torch.from_numpy(attack.generate(x=input.cpu(), y=target.cpu())).cuda()
        return f_attack
    elif args.attack == "gen":
        import foolbox as fb
        fmodel = fb.PyTorchModel(net, bounds=(0, 1))
        attack = fb.attacks.GenAttack(steps=100)
        def f_attack(input, target):
            from foolbox.criteria import TargetedMisclassification
            adv_target = torch.randint(low=0, high=args.n_classes-1, size=target.shape).cuda()
            adv_target = adv_target + (adv_target >= target).float()
            adv_target = adv_target - (adv_target > args.n_classes-1).float()
            return attack(fmodel, input, TargetedMisclassification(adv_target.long()), epsilons=args.eps)[1]
        return f_attack



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
    # args.batch_size = 70

    # Full Trainloader/Testloader
    trainloader = torch.utils.data.DataLoader(dataset(args, True,  transform_train), batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    testloader  = torch.utils.data.DataLoader(dataset(args, False, transform_test),  batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

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