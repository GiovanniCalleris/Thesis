import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
from pruning_functions import Strategy3, Strategy4, Strategy5, Strategy6, Strategy7, Strategy8, Strategy9, \
    OutputAnalyzer, Encoder2Mask
'''
https://github.com/akamaster

Copyright (c) 2018, Yerlan Idelbayev

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----------------------------------------------------------------------------

https://github.com/chenyaofo/pytorch-cifar-models/blob/master/LICENSE
BSD 3-Clause License

Copyright (c) 2021, chenyaofo
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
# parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
#                     choices=model_names,
#                     help='model architecture: ' + ' | '.join(model_names) +
#                     ' (default: resnet32)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet110',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet110)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run') # 200
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')  # 1e-4
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
# parser.add_argument('--resume', default='./save_temp/model.th',
#                     type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default='resnet_cifar10/pretrained_models/resnet110-1d1ed7c2.th',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# 'resnet_cifar10/pretrained_models/resnet110-1d1ed7c2.th'
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set', default=False)
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=40)
parser.add_argument('--strategy', default=11,
                    help='strategy to use (default: None)'),
parser.add_argument('--pruning_rate', default='ACTUAL_PRUNING_1.15',
                    help='pruning_rate to use (default: [.7, .6, .5])')  # [.5, .4, .3]
parser.add_argument('--modules_to_prune', default=["layer1", "layer2", "layer3"],
                    help='modules_to_prune to use (default: ["layer1", "layer2", "layer3"])')
best_prec1 = 0


def main():
    wan = True
    global args, best_prec1
    args = parser.parse_args()


    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()

    # if args.strategy is '6_test':
    #     print('MixedStrategy3&5')
    #     model = Strategy3(model, all_at_once=True, pruning_remove=0)
    #     model = Strategy5(model.model_teacher, test=True)

    # optionally resume from a checkpoint
    if args.resume and args.resume.split('/')[1] == 'save_temp':
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            pruned = [(n, p) for n, p in checkpoint['state_dict'].items()]
            # for n, p in original:
            for n_dict, p_dict in pruned:
                modules = n_dict.split('.')
                nn_module = None
                for i, module in enumerate(modules):
                    # p_dict.to(device)
                    if 'model_teacher_fake' not in n_dict:
                        if nn_module is not None and isinstance(nn_module, nn.DataParallel):
                            nn_module = getattr(nn_module, 'module')

                        if i == 0:
                            if isinstance(model, nn.DataParallel):
                                nn_module = getattr(model, 'module')
                                nn_module = getattr(nn_module, module)
                            else:
                                nn_module = getattr(model, module)
                        elif i < len(modules) - 1:
                            if hasattr(nn_module, module):
                                nn_module = getattr(nn_module, module)
                        else:
                            if (not isinstance(p_dict, torch.FloatTensor) or 'running' in module) \
                                    and ('weight' not in n_dict and 'bias' not in n_dict):
                                setattr(nn_module, module, p_dict)
                            else:
                                # delattr(nn_module, module)
                                setattr(nn_module, module, torch.nn.Parameter(p_dict))
                                if 'weight' in module and isinstance(nn_module, torch.nn.Conv2d):
                                    setattr(nn_module, 'in_channels', p_dict.shape[1])
                                    setattr(nn_module, 'out_channels', p_dict.shape[0])


            args.start_epoch = 0 if 'epoch' not in checkpoint else checkpoint['epoch']
            best_prec1 = 0 if 'best_prec1' not in checkpoint else checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, args.start_epoch))
    elif os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)

        args.start_epoch = 0 if 'epoch' not in checkpoint else checkpoint['epoch']
        best_prec1 = 0 if 'best_prec1' not in checkpoint else checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.evaluate, args.start_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        print('LOADING BASE MODEL')
        model.load_state_dict(torch.load('pretrained_models/resnet110-1d1ed7c2.th')['state_dict'])

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=False),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()
    if wan:
        import wandb
        wandb.init(project="CIFAR_PRUNING_AGAIN", sync_tensorboard=True,
                   name=f'Pruning strategy {args.strategy} with pr {args.pruning_rate} and lr {args.lr}')
    else:
        wandb = None
    if args.strategy == 3:
        print('Strategy3')
        model = Strategy3(model, amount=args.pruning_rate, modules_to_prune=args.modules_to_prune,
                          all_at_once=True)
        wandb.log({
        'flops_count': model.flops_count,
        'params_count': model.params_count,
        'active_parameters': model.active_parameters,
        })

    elif args.strategy == 4:
        print('Strategy4')
        model = Strategy4(model)

    elif args.strategy == 5:
        print('Strategy5')
        model = Strategy5(model)
    elif args.strategy == 6:
        print('Strategy6')
        model = Strategy6(model, amount=args.pruning_rate, modules_to_prune=args.modules_to_prune,
                          all_modules=False)

        tmp_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./resnet_cifar10/data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=False),
        batch_size=500, shuffle=True,
        num_workers=0, pin_memory=True)
        model.cuda()
        model.strategy(next(iter(tmp_loader))[0].cuda())
        del tmp_loader
        wandb.log({
            'flops_count': model.flops_count,
            'params_count': model.params_count,
            'active_parameters': model.active_parameters,
        })
    # elif args.strategy == 6:
    #     print('MixedStrategy3&5')
    #     model = Strategy3(model, all_at_once=True)
    #     model = Strategy5(model.model_teacher)
    # elif args.strategy == '6_test':
    #     print('MixedStrategy3&5')
    #     model = Strategy3(model, all_at_once=True, pruning_remove=0)
    #     model = Strategy5(model.model_teacher, test=True)

    elif args.strategy == 7:
        print('Strategy7')
        model = Strategy7(model, amount=args.pruning_rate, modules_to_prune=args.modules_to_prune,
                          all_modules=False, pad_with_zeros=[True, True, True])
        # wandb.log({
        #     'flops_count': model.flops_count,
        #     'params_count': model.params_count,
        #     'active_parameters': model.active_parameters,
        # })
    elif args.strategy == 8:
        print('Strategy8')
        model = Strategy8(model, amount=args.pruning_rate, modules_to_prune=args.modules_to_prune,
                          all_modules=False, pad_with_zeros=[True, True, True])

    elif args.strategy == 9:
        print('Strategy9')
        model = Strategy9(model, amount=args.pruning_rate, modules_to_prune=args.modules_to_prune,
                          all_modules=False, pad_with_zeros=[True, True, True])

    elif args.strategy in [10]:
        global e2m
        e2m = Encoder2Mask(model, device='cuda')
        e2m.to('cuda')
        e2m()
        model = e2m.model
        global optimizer_pruning_mode_10
        optimizer_pruning_mode_10 = torch.optim.AdamW(list(e2m.parameters()), lr=5e-3)

    elif args.strategy in [11]:
        global oa
        oa = OutputAnalyzer(model)
        oa.to('cuda')
        # global optimizer_pruning_mode_11
        oa.optimizer_pruning_mode_11 = torch.optim.SGD(oa.parameters(), lr=1e-2)

        if wan:
            wandb.log({
                'flops_count': oa.flops_count,
                'params_count': oa.params_count,
            })
    else:
        print('NO STRATEGY')

    # else:
    #     model.init_strategy()

    global optimizer_pruning
    optimizer_pruning = None

    if args.strategy == 7:
        optimizer_pruning = torch.optim.AdamW([p for n, p in model.named_parameters() if 'conv_pruner' in n], lr=4e-4,
                                    weight_decay=args.weight_decay)
        optimizer = torch.optim.SGD([p for n, p in model.named_parameters() if 'conv_pruner' not in n], args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    if args.strategy == 8:

        class OptimSSS(nn.Module):

            def __init__(self, weights, lr=0.1, gamma=0.01):
                super().__init__()
                self.weights = {n: weight for n, weight in weights}
                self.lr = lr
                self.gamma = gamma
                self.mom = {n: torch.zeros_like(weight) for n, weight in weights}


            def soft_thresholding(self, x, gamma):
                y = torch.maximum(torch.zeros(1).to(x.device), torch.abs(x) - gamma)

                return torch.sign(x) * y

            def apg_updater(self, weight, lr, grad, mom, gamma, name):
                z = weight - lr * grad

                z = self.soft_thresholding(z, lr * gamma)
                mom[name][:] = z - weight + 0.9 * mom[name]
                weight.data[:] = z + 0.9 * mom[name]

            def step(self):
                for n, weight in self.weights.items():
                    if weight.grad is not None:
                        self.apg_updater(weight=weight, lr=self.lr, grad=weight.grad,
                                         mom=self.mom, gamma=self.gamma, name=n)
                    else:
                        print('GRAD IS None')



        optimizer_pruning = OptimSSS([(n, p) for n, p in model.named_parameters() if 'conv_pruner' in n])

        optimizer = torch.optim.SGD([p for n, p in model.named_parameters() if 'conv_pruner' not in n], args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    else:
        optimizer = torch.optim.SGD([p for n, p in model.named_parameters() if 'conv_pruner' not in n], args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                     milestones=[100, 150], last_epoch=args.start_epoch - 1)

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        sub_lr = args.lr
        if epoch >= 25 and epoch % 10 == 0:
            sub_lr = args.lr * ((0.941)**(epoch-25))
            print('REDUCING LR TO', sub_lr)
            if wandb is not None:
                wandb.log({'LR': sub_lr})
            # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
            # then switch back. In this setup it will correspond for first epoch.
            for param_group in oa.optimizer_pruning_mode_11.param_groups:
                param_group['lr'] = sub_lr
        if args.strategy in [7, 9] and epoch in [20]:
            model.stop_def_met()
            print('STOPPING PRUNING FROM NOW ON TRAIN ONLY')

        if args.strategy in [8] and epoch in [1]:
            model.stop_def_met()
            print('STOPPING PRUNING FROM NOW ON TRAIN ONLY')

        if epoch == 1 and args.strategy in [10]:
            e2m.stop_pruning()

        if args.strategy in [3, 6, 7] and epoch in [5, 10]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
            print('learning rate has been changed')

        if args.strategy in [8, 9] and epoch in [10, 25]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
            print('learning rate has been changed')

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch, wandb)
        # lr_scheduler.step()
        if args.strategy in [10]:
            e2m()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)
        if wandb is not None:
            wandb.log({'Acc': prec1})


        # remember best acc@1 and save checkpoint
        is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)
        if args.strategy in [11]:
            check_finished = oa.finished
        else:
            check_finished = True

        if check_finished and is_best:
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': oa.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))
            print('SAVED MODEL WITH ACC:', best_prec1)

        # save_checkpoint({
        #     'state_dict': model.state_dict(),
        #     'best_prec1': best_prec1,
        # }, is_best, filename=os.path.join(args.save_dir, 'model.th'))  # 'model.th'


def train(train_loader, model, criterion, optimizer, epoch, wandb=None):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        if i == 5 and args.strategy in [7, 8, 9]:
            wandb.log({
                'flops_count': model.flops_count,
                'params_count': model.params_count,
                'active_parameters': model.active_parameters,
            })
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()


        # compute output
        if args.strategy in [11]:
            output = oa(input_var, wandb)

        else:
            output = model(input_var)

        if args.strategy in [10]:
            optimizer.zero_grad()
            e2m()
            if i % 50 == 0:
                print('TOT PARAMS:', e2m.tot_params, '| TOT ACTIVE PARAMS:', e2m.active_params, '| RATIO: ',
                      e2m.active_params / e2m.tot_params)
        loss = criterion(output, target_var)
        if args.strategy in [10]:
            if e2m.stopped:
                loss.backward()
                e2m.optimizer.step()
                continue
            else:
                loss_final = (1 + loss) * (1 + e2m.loss_to_add / e2m.tot_shape)
                loss_final.backward()
                optimizer_pruning_mode_10.step()
                continue
        if args.strategy in [4, 5]:
            model.backward_and_step(loss)
        else:
        # compute gradient and do SGD step
            if args.strategy in [7, 8]:
                model.backward_and_step()
                optimizer_pruning.zero_grad()
            if args.strategy in [9]:
                model.backward_and_step()
            if args.strategy in [11]:
                oa.optimizer_pruning_mode_11.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            if args.strategy in [7, 8]:
                optimizer_pruning.step()
            if args.strategy in [11]:
                oa.optimizer_pruning_mode_11.step()
            else:
                optimizer.step()

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if wandb is not None:
                wandb.log({'step': args.print_freq,
                           'loss': losses.avg
                           })
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            if args.strategy in [11]:
                output = oa(input_var)
            else:
                output = model(input_var)
            # output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * acc@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()