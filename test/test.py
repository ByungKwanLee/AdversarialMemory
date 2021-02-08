#!/usr/bin/env python

# torch package
import torch
from torchvision.utils import save_image

# basic package
import os
import sys
sys.path.append('.')
import argparse
from tqdm import tqdm

# custom package
from loader.argument_print import argument_testprint
from loader.loader import dataset_loader, network_loader, OEM_network_loader, attack_loader

# cudnn enable
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# argument parser
parser = argparse.ArgumentParser(description='Joint Adversarial Defense')
parser.add_argument('--steps', default=10, type=int, help='adv. steps')
parser.add_argument('--eps', required=True, type=float, help='max norm')
parser.add_argument('--dataset', required=True, type=str, help='dataset name')
parser.add_argument('--network', required=True, type=str, help='network name')
parser.add_argument('--data_root', required=True, type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./experiment', type=str, help='save directory')
parser.add_argument('--datetime', required=True, type=str, help='checkpoint datetime')
parser.add_argument('--baseline', required=True, type=str, help='baseline')

args = parser.parse_args()

checkpoint_name = args.baseline+'_'+args.network+'_'+args.dataset+'_'+args.datetime+'.pth'


# loading dataset, network, attack
_, testloader = dataset_loader(args)
if "OEM" in args.baseline:
    args.isinitialize=True
    net = OEM_network_loader(args, mean=args.mean, std=args.std).cuda()
    net.load_state_dict(torch.load(os.path.join(args.save_dir, checkpoint_name))['model_state_dict'])
    net.eval()
else:
    if "Plain" in args.baseline:
        args.attack="Plain"
    else:
        args.attack="AT"
    net = network_loader(args, mean=args.mean, std=args.std).cuda()
    net.load_state_dict(torch.load(os.path.join(args.save_dir, checkpoint_name))['model_state_dict'])
print('Total number of Model Parameters is: {}M\n'.format(sum(p.numel() for p in net.parameters() if p.requires_grad)/1000000))


def experiment_clean():

    # test arguemnet test print
    argument_testprint(args, checkpoint_name)


    correct = 0
    total = 0
    print('\n[RFM/Test] Under Testing ... Wait PLZ')
    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):

        # dataloader parsing and generate adversarial examples
        inputs, targets = inputs.cuda(), targets.cuda()

        # Evaluation
        outputs = net.get_inference(inputs)

        # Test
        pred = torch.max(outputs, dim=1)[1]
        correct += torch.sum(pred.eq(targets)).item()
        total += targets.numel()

    print('[RFM/Test] Acc: {:.3f}'.format(100.*correct / total))


def experiment_robustness():

    for steps in [args.steps]:
        args.steps = steps
        attack_score = []

        # test arguemnet test print
        argument_testprint(args, checkpoint_name)

        # stack attack module
        attack_module = {}
        for attack_name in ['fgsm']:
        # for attack_name in ['fgsm', 'pgd', 'cw', 'ead']:
        # for attack_name in ['pgd']:
        # for attack_name in ['cw', 'ead']:
        # for attack_name in ['hop', 'square', 'boundary']:
        # for attack_name in ['square']:
        # for attack_name in ['boundary']:
        # for attack_name in ['adapt']:
            args.attack = attack_name
            attack_module[attack_name]=attack_loader(args, net)

        for key in attack_module:

            correct = 0
            total = 0
            print('\n[RFM/Test] Under Testing ... Wait PLZ')
            for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):

                # dataloader parsing and generate adversarial examples
                inputs, targets = inputs.cuda(), targets.cuda()
                adv_x = attack_module[key](inputs, targets) if args.eps != 0 else inputs

                # Evaluation
                outputs = net.get_inference(adv_x)

                # Test
                pred = torch.max(outputs, dim=1)[1]
                correct += torch.sum(pred.eq(targets)).item()
                total += targets.numel()

            print('[RFM/{}] Acc: {:.3f}'.format(key, 100.*correct / total))
            attack_score.append(100.*correct / total)


        print('\n----------------Summary----------------')
        print(steps, ' steps pgd attack')
        for key, score in zip(attack_module, attack_score):
            print(str(key),' : ', score)
        print('---------------------------------------\n')

def experiment_capturing():
    for steps in [args.steps]:
        args.steps = steps

        # test arguemnet test print
        argument_testprint(args, checkpoint_name)

        # stack attack module
        attack_module = {}
        for attack_name in ['fgsm', 'pgd']:
            # for attack_name in ['fgsm', 'pgd', 'cw', 'ead']:
            # for attack_name in ['pgd']:
            # for attack_name in ['cw', 'ead']:
            # for attack_name in ['hop', 'square', 'boundary']:
            # for attack_name in ['square']:
            # for attack_name in ['boundary']:
            # for attack_name in ['adapt']:
            args.attack = attack_name
            attack_module[attack_name] = attack_loader(args, net)

        for key in attack_module:

            print('\n[RFM/Test] Under Capturing Features ... Wait PLZ')
            for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
                # dataloader parsing and generate adversarial examples
                inputs, targets = inputs.cuda(), targets.cuda()
                adv_x = attack_module[key](inputs, targets) if args.eps != 0 else inputs

                # Capturing Clean Feature and Adversarial Feature
                save_image(inputs[0], os.path.join('FeatureVisualization', args.dataset+key+'_clean_image.png'))
                save_image(adv_x[0], os.path.join('FeatureVisualization', args.dataset+'_'+key+'_adv_image.png'))
                net.get_capturing_feature(inputs, os.path.join('FeatureVisualization', args.dataset+'_'+key+'_clean_feature'))
                net.get_capturing_feature(adv_x, os.path.join('FeatureVisualization', args.dataset+'_'+key+'_adv_feature'))
                break

if __name__ == '__main__':
    # experiment_clean()
    # experiment_robustness()

    if "OEM" in args.baseline:
        experiment_capturing()
