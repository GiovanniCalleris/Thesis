import copy

import torch
import torch.nn as nn
import torch_pruning as tp
import torchvision
# from mmcv.parallel import MMDataParallel
from tqdm import tqdm
import numpy as np

from resnet import LambdaLayer


class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    def __init__(self):
        super(Binarizer, self).__init__()

    @staticmethod
    def forward(self, inputs):
        outputs = inputs.clone()
        value = 0.
        outputs[inputs.le(value)] = 0.
        outputs[inputs.gt(value)] = 1.

        return outputs

    @staticmethod
    def backward(self, gradOutput):
        return gradOutput

class BinarizerAvg(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    def __init__(self):
        super(BinarizerAvg, self).__init__()

    @staticmethod
    def forward(self, inputs):
        inputs = torch.mean(torch.abs(torch.flatten(inputs, 1)), 1)
        outputs = inputs.clone()

        outputs[inputs.le(0)] = 0
        outputs[inputs.gt(0)] = 1

        return outputs

    @staticmethod
    def backward(self, gradOutput):
        return gradOutput

class Alpha_bin(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    def __init__(self):
        super(Alpha_bin, self).__init__()

    @staticmethod
    def forward(self, inputs):
        outputs = inputs.clone()

        outputs[inputs.le(0)] = 0
        outputs[inputs.gt(1)] = 1

        return outputs

    @staticmethod
    def backward(self, gradOutput):
        return gradOutput

# class BinarizerModule(nn.Module):
#     """Binarizes {0, 1} a real valued tensor."""
#
#     def __init__(self):
#         super(BinarizerModule, self).__init__()
#
#     def forward(self, inputs):
#         outputs = inputs.clone()
#
#         outputs[inputs.le(0)] = 0
#         outputs[inputs.gt(0)] = 1
#
#         return outputs
#
#     def backward(self, gradOutput):
#         return gradOutput


class BinarizerHighThreshold(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    def __init__(self):
        super(BinarizerHighThreshold, self).__init__()

    @staticmethod
    def forward(self, inputs):
        outputs = inputs.clone()

        outputs[inputs.le(0)] = 0
        outputs[inputs.gt(0)] = 1

        return outputs

    @staticmethod
    def backward(self, gradOutput):
        return gradOutput

class MaskingModule(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.moduleToMask = module
        self.mask = []

    def forward(self, x):
        if self.mask:
            x = torch.transpose(x, 0, 1)
            for m in self.mask:
                x = x[m]
            x = torch.transpose(x, 0, 1)
        self.moduleToMask.to(x.device)
        x = self.moduleToMask(x)
        return x


class Strategy4(nn.Module):

    def __init__(self, model, task='classification'):
        super().__init__()
        torch.autograd.set_detect_anomaly(True)
        self.model_teacher = model
        self.device = next(model.parameters()).device
        self.iter = 0
        self.task = task
        self.init_strategy()


    def update_params(self, mode='tensors'):
        names_list = [n for n in self.pruning_dict.keys()]
        for n in names_list:
            p = self.pruning_dict[n].clone()
            modules = n.split('___')
            for i, module in enumerate(modules):
                if i == 0:
                    nn_module = getattr(self.model_teacher, module)

                elif i + 1 < len(modules):
                    nn_module = getattr(nn_module, module)

                else:
                    p = p * torch.ones_like(p)
                    param_bin = self.make_bins.apply(p)
                    # p_to_update = getattr(nn_module, module)
                    # p_updated = (p * p_to_update.transpose(1, -1)).transpose(1, -1)
                    p_updated = torch.transpose((param_bin * torch.transpose(self.original_dict[n], 1, -1)), 1, -1)
                    if mode == 'tensors':
                        delattr(nn_module, module)
                        setattr(nn_module, module, p_updated)

                    else:
                        setattr(nn_module, module, nn.Parameter(p_updated))

    def remove_params(self):
        pruning_list = [(n, p) for n, p in self.pruning_dict.items()]
        for n, p in pruning_list:
            modules = n.split('___')
            mask = p.gt(0)
            with torch.no_grad():
                if mask.ndim == 0 or mask[torch.logical_not(mask)].shape[0] + 3 > mask.shape[0]:
                    # mask[0] = True
                    continue
            if not mask.all():

                for i, module in enumerate(modules):
                    if i == 0:
                        nn_module = getattr(self.model_teacher, module)

                    elif i + 1 < len(modules):
                        nn_module = getattr(nn_module, module)
                        if isinstance(nn_module, MaskingModule):
                            nn_module.mask = mask
                    else:
                        p_to_update = getattr(nn_module, module)
                        self.pruning_dict[n] = nn.Parameter(p[mask])
                        self.original_dict[n] = nn.Parameter(
                            torch.transpose(torch.transpose(self.original_dict[n], 0, 1)[mask], 0, 1))
                        p_updated = torch.transpose(p_to_update, 0, 1)[mask]
                        p_updated = torch.transpose(p_updated, 0, 1)
                        setattr(nn_module, module, p_updated)

                        setattr(nn_module, 'in_channels', p_updated.shape[1])
                        setattr(nn_module, 'out_channels', p_updated.shape[0])

    def print_flops_and_params(self, model):
        example_inputs = torch.randn(1, 3, 1024, 1024)
        self.update_params(mode='parameters')
        flops_count, params_count = tp.utils.count_ops_and_params(model, example_inputs)
        print(f"ACTIVE PARAMETERS ABSOLUTE ORIGINAL: {self.params_count_orig}")
        print(f"ACTIVE PARAMETERS ABSOLUTE PRUNED: {params_count}")
        print(f"ACTIVE PARAMETERS: {100 * (params_count / self.params_count_orig)}")
        print(f'FLOPS PRUNED: {flops_count} | ORIGINAL: {self.flops_count_orig}')

    def init_strategy(self):
        self.avoid_data_parallel()
        self.make_bins = Binarizer()
        self.pruning_dict = nn.ParameterDict()
        self.original_dict = nn.ParameterDict()
        named_param_list = list(self.model_teacher.named_parameters())
        for n, p in named_param_list:
            modules = n.split('.')
            new_modules = []
            for i, module in enumerate(modules):
                if i == 0:
                    nn_module = getattr(self.model_teacher, module)
                    new_modules.append(module)

                elif i + 1 < len(modules):
                    previous_nn_module = nn_module
                    nn_module = getattr(nn_module, module)
                    new_modules.append(module)

                    if isinstance(nn_module, nn.Conv2d) and not isinstance(previous_nn_module, MaskingModule):
                        setattr(previous_nn_module, module, MaskingModule(nn_module))
                        new_modules.append('moduleToMask')
                        if 'weight' in n and 3 not in p.shape[:2] and p.shape[0] == p.shape[1]:
                            new_modules.append('weight')
                            self.pruning_dict['___'.join(new_modules)] = nn.Parameter(
                                torch.mean(torch.abs(p.transpose(0, 1).flatten(1))))
                            self.original_dict['___'.join(new_modules)] = p
                            delattr(nn_module, 'weight')
                            setattr(nn_module, 'weight', p.clone())

                        # nn_module = getattr(previous_nn_module, module)

                # elif isinstance(nn_module, nn.Conv2d):
                #     new_modules.append(module)
                #
                    # if 'weight' in n and 3 not in p.shape:
                    #     self.pruning_dict['___'.join(new_modules)] = nn.Parameter(
                    #         torch.ones_like(p[0, :, 0, 0]) * 1e-4)
                    #     self.original_dict['___'.join(new_modules)] = p
        self.optimizer_pruning = torch.optim.SGD([p for p in self.pruning_dict.parameters()], lr=1e-2)
        self.optimizer_param = torch.optim.SGD([p for p in self.original_dict.parameters()], lr=1e-2)
        self.cpu()
        example_inputs = torch.randn(1, 3, 1024, 1024)
        self.update_params(mode='parameters')
        self.flops_count_orig, self.params_count_orig = tp.utils.count_ops_and_params(self.model_teacher,
                                                                                      example_inputs)

    def zero_grad_and_update(self):
        self.optimizer_pruning.zero_grad()
        self.optimizer_param.zero_grad()
        self.update_params()

    def remove_and_print(self):
        if self.iter % 1000 == 0:
            self.remove_params()
            self.cpu()
            self.print_flops_and_params(self.model_teacher)
            self.cuda()

    def backward_and_step(self, loss):
        loss.backward()
        self.optimizer_pruning.step()
        self.optimizer_param.step()
        self.iter += 1

    def avoid_data_parallel(self):

        class AvoidDataParallel(nn.Module):
            def __init__(self, module, task):
                super().__init__()
                self.module = module
                self.device = next(module.parameters()).device
                self.task = task

            if self.task == 'segmentation':
                def train_step(self, batch):
                    tmp = torch.ones(1, 19, batch.shape[-2], batch.shape[-1])
                    img_metas = [{'ori_shape': batch.shape, 'img_shape': tmp.shape, 'pad_shape': torch.zeros(1),
                                  'gt_semantic_seg': tmp, 'flip': False}]

                    losses = self.module(img=batch, img_metas=img_metas, return_loss=True,
                                         **{'gt_semantic_seg': torch.ones(1, 1,
                                                                          batch.shape[-2] // 8,
                                                                          batch.shape[-1] // 8
                                                                          ).long()})
                    loss, log_vars = self.module._parse_losses(losses)

                    return loss

            elif self.task == 'classification':
                def forward(self, x):
                    return self.module(x)

            def train_step(self, batch, optimizer, **kwargs):
                # batch['img_metas'] = batch['img_metas'].data[0]
                # batch['img'] = batch['img'].data[0].to(self.device)
                # batch['gt_semantic_seg'] = batch['gt_semantic_seg'].data[0].to(self.device)

                return self.module.train_step(batch, optimizer, **kwargs)

        self.model_teacher = AvoidDataParallel(self.model_teacher.module, self.task)

    def forward(self, x):
        self.remove_and_print()
        self.zero_grad_and_update()
        return self.model_teacher(x)

    def train_step(self, batch, optimizer, **kwargs):
        self.remove_and_print()
        self.zero_grad_and_update()
        return self.model_teacher.train_step(batch, optimizer, **kwargs)

    def val_step(self, batch, **kwargs):
        return self.model_teacher.val_step(batch, **kwargs)

class Strategy3(nn.Module):

    """
    L1 method
    """

    def __init__(self, model, modules_to_prune=['layer1', 'layer2', 'layer3'], amount=[.5, .4, .3],
                 not_to_prune='conv2',
                 task='classification', all_at_once=False, iterative_steps=5):
        super().__init__()
        self.model_teacher = model
        self.device = next(model.parameters()).device
        self.iter = 0
        self.amount = amount
        self.all_at_once = all_at_once
        self.task = task
        self.iterative_steps = iterative_steps
        self.modules_to_prune = modules_to_prune
        self.pruners = []
        self.not_to_prune = not_to_prune
        self.init_strategy()

    def pruners_step(self):
        for pruner in self.pruners:
            pruner.step()

    def init_strategy(self):

        class PassModule(nn.Module):
            def __init__(self, module, device, task):
                super().__init__()
                self.model_teacher = module
                self.device = device
                self.task = task

            if self.task == 'segmentation':
                def forward(self, batch):
                    # batch = batch.unsqueeze(0)
                    if hasattr(batch, 'shape') and batch.shape == (1, 3, 1024, 1024):
                        # batch = torch.randn(1, 3, 1024, 1024)
                        tmp = torch.ones(1, 19, batch.shape[-2], batch.shape[-1])
                        img_metas = [{'ori_shape': batch.shape, 'img_shape': tmp.shape, 'pad_shape': None,
                                      'gt_semantic_seg': tmp, 'flip': False}]

                        losses = self.model_teacher(img=batch, img_metas=img_metas, return_loss=True,
                                                    **{'gt_semantic_seg': torch.ones(1, 1, batch.shape[-2] // 8,
                                                                                     batch.shape[-1] // 8
                                                                                     ).long()})
                    else:
                        img = batch['img'].data
                        gt_semantic_seg = torch.cat(batch['gt_semantic_seg'].data, dim=0).cuda()
                        del batch['img']
                        batch['gt_semantic_seg'] = gt_semantic_seg
                        losses = self.model_teacher(img=torch.cat(img, dim=0).cuda(),
                                                    **{k: v.data for k, v in batch.items()})
                    loss, log_vars = self.model_teacher._parse_losses(losses)

                    return loss
            elif self.task == 'classification':
                def forward(self, x):
                    return self.model_teacher(x)

            def train_step(self, batch, optimizer, **kwargs):
                # batch['img_metas'] = batch['img_metas'].data[0]
                # batch['img'] = batch['img'].data[0].to(self.device)
                # batch['gt_semantic_seg'] = batch['gt_semantic_seg'].data[0].to(self.device)

                return self.model_teacher.train_step(batch, optimizer, **kwargs)

            def val_step(self, batch, **kwargs):
                return self.model_teacher.val_step(batch, **kwargs)



        def prune_specific_module(model, amount, divisor):

            model.cpu()

            # modules_list = [m for mn, m in model.named_modules() for n, p in m.named_parameters()
            #                 if isinstance(m, torch.nn.Conv2d) and 'weight' in n and p.shape[0] != p.shape[1] or
            #                 self.not_to_prune in mn or '17.conv1' in mn
            #                 ]
            modules_list = [m for mn, m in model.named_modules()
                             for n, p in m.named_parameters()
                             if isinstance(m, nn.Conv2d) and 19 in p.shape or
                            self.not_to_prune in mn]

            modules_list += [module for name, module in model.named_modules() if
             all([m1 == module for n, m1 in module.named_modules()]) and (
                          isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.LayerNorm))]

            # example_inputs = torch.randn(1, [m.weight.shape[1] for m in model.modules() if isinstance(m, nn.Conv2d)][0],
            #                              1024//(2**divisor), 1024//(2**divisor))
            example_inputs = torch.randn(1, 3, 1024, 1024)
            # example_inputs = torch.randn(1, [p.shape[1] for p in model.parameters() if len(p.shape) > 1][0],
            #                              1024//(2**divisor), 1024//(2**divisor))

            self.pruners.append(tp.pruner.MagnitudePruner(
                model.model,
                example_inputs,
                importance=tp.importance.MagnitudeImportance(p=1),
                iterative_steps=self.iterative_steps,
                ch_sparsity=amount,
                ignored_layers=modules_list,
            ))

        def prune_specific_modules():
            divisor = 0
            if self.modules_to_prune is not None:
                for module, amount in zip(self.modules_to_prune, self.amount):
                    for module_name, nn_module in self.model_teacher.named_modules():
                        if module == module_name.split('.')[-1]:
                            divisor += 1
                            prune_specific_module(model=nn_module, amount=amount, divisor=divisor)
            else:
                prune_specific_module(model=self.model_teacher, amount=self.amount, divisor=divisor)


        # if hasattr(self.model_teacher, 'module'):
        #     self.model_teacher = PassModule(self.model_teacher.module, self.device, self.task)
        # else:
        #     self.model_teacher = PassModule(self.model_teacher, self.device, self.task)

        prune_specific_modules()
        example_inputs = torch.randn(1, 3, 1024, 1024)
        self.cpu()
        self.flops_count_orig, self.params_count_orig = tp.utils.count_ops_and_params(self.model_teacher,
                                                                                  example_inputs)

        if self.all_at_once:
            for _ in range(self.iterative_steps):
                self.pruners_step()
            # self.model_teacher.module = self.model_teacher_fake.model_teacher
            self.flops_count, self.params_count = tp.utils.count_ops_and_params(self.model_teacher, example_inputs)
            self.active_parameters = 100 * (self.params_count / self.params_count_orig)
            print(f"ACTIVE PARAMETERS ABSOLUTE ORIGINAL: {self.params_count_orig}")
            print(f"ACTIVE PARAMETERS ABSOLUTE PRUNED: {self.params_count}")
            print(f"ACTIVE PARAMETERS: {self.active_parameters}")
            print(f'FLOPS PRUNED: {self.flops_count} | ORIGINAL: {self.flops_count_orig}')
        # else:
        #     self.model_teacher_fake = PassModule(self.model_teacher, self.device, self.task)
        #     self.model_teacher = self.model_teacher_fake.model_teacher

        self.cuda()

    def print_zero_step_pruning(self):
        print(f"ACTIVE PARAMETERS ABSOLUTE ORIGINAL: {self.params_count_orig}")
        print(f"ACTIVE PARAMETERS ABSOLUTE PRUNED: {self.params_count}")
        print(f"ACTIVE PARAMETERS: {self.active_parameters}")
        print(f'FLOPS PRUNED: {self.flops_count} | ORIGINAL: {self.flops_count_orig}')

    def strategy(self):
        if self.iter % 1000 == 0 and not self.all_at_once:
            example_inputs = torch.randn(1, 3, 1024, 1024)
            # self.model_teacher = self.model_teacher.module

            self.cpu()
            # flops_count_orig, params_count_orig = tp.utils.count_ops_and_params(self.model_teacher_fake, example_inputs)

            self.pruners_step()

            self.flops_count, self.params_count = tp.utils.count_ops_and_params(self.model_teacher, example_inputs)
            # self.model_teacher.module = self.model_teacher_fake.model_teacher

            print(f"ACTIVE PARAMETERS ABSOLUTE ORIGINAL: {self.params_count_orig}")
            print(f"ACTIVE PARAMETERS ABSOLUTE PRUNED: {self.params_count}")
            print(f"ACTIVE PARAMETERS: {100 * (self.params_count / self.params_count_orig)}")
            print(f'FLOPS PRUNED: {self.flops_count} | ORIGINAL: {self.flops_count_orig}')
                # self.model_teacher.to(self.device)

                # self.model_teacher.module = self.model_teacher_clone
            self.cuda()


    def forward(self, x):
        self.strategy()
        x = self.model_teacher(x)
        self.iter += 1
        return x

    def train_step(self, batch, optimizer, **kwargs):

        self.strategy()
        self.iter += 1
        return self.model_teacher.train_step(batch, optimizer, **kwargs)

    def val_step(self, batch, **kwargs):
        return self.model_teacher.val_step(batch, **kwargs)


class Strategy5(nn.Module):

    def __init__(self, model, task='classification', test=False):
        super().__init__()
        torch.autograd.set_detect_anomaly(True)
        self.model_teacher = model
        self.device = next(model.parameters()).device
        self.iter = 0
        self.task = task
        self.mask = {}
        self.test = test
        self.init_strategy()


    def update_params(self, mode='tensors'):
        names_list = [n for n in self.pruning_dict.keys()]
        for n in names_list:
            p = self.pruning_dict[n].clone()
            modules = n.split('___')
            for i, module in enumerate(modules):
                if i == 0:
                    nn_module = getattr(self.model_teacher, module)

                elif i + 1 < len(modules):
                    nn_module = getattr(nn_module, module)

                else:

                    p = torch.tensordot(p, self.original_dict[n].clone(), dims=[(0, 1, 2), (0, 2, 3)])
                    self.mask[n] = self.make_bins.apply(p)
                    # p_to_update = getattr(nn_module, module)
                    # p_updated = (p * p_to_update.transpose(1, -1)).transpose(1, -1)
                    p_updated = torch.transpose((torch.transpose(self.original_dict[n].clone(), 1, -1) * self.mask[n].clone()), 1, -1)
                    if mode == 'tensors':
                        delattr(nn_module, module)
                        setattr(nn_module, module, p_updated)
                    else:
                        setattr(nn_module, module, nn.Parameter(p_updated))

    def remove_params(self):
        pruning_list = [(n, p) for n, p in self.mask.items()]
        for n, p in pruning_list:
            modules = n.split('___')

            mask = p.gt(0)
            with torch.no_grad():
                if mask.ndim == 0 or mask[torch.logical_not(mask)].shape[0] + 3 > mask.shape[0]:
                # mask[0] = True
                    continue
            if not mask.bool().all():

                for i, module in enumerate(modules):
                    if i == 0:
                        nn_module = getattr(self.model_teacher, module)

                    elif i + 1 < len(modules):
                        nn_module = getattr(nn_module, module)
                        if isinstance(nn_module, MaskingModule):
                            nn_module.mask.append(mask)
                    else:
                        p_to_update = getattr(nn_module, module)
                        # self.pruning_dict[n] = nn.Parameter(self.pruning_dict[n][mask])
                        self.original_dict[n] = nn.Parameter(
                            torch.transpose(torch.transpose(self.original_dict[n], 0, 1)[mask], 0, 1))
                        p_updated = torch.transpose(p_to_update, 0, 1)[mask]
                        p_updated = torch.transpose(p_updated, 0, 1)
                        setattr(nn_module, module, p_updated)

                        setattr(nn_module, 'in_channels', p_updated.shape[1])
                        setattr(nn_module, 'out_channels', p_updated.shape[0])

    def print_flops_and_params(self, model):
        example_inputs = torch.randn(1, 3, 1024, 1024)
        self.update_params(mode='parameters')
        flops_count, params_count = tp.utils.count_ops_and_params(model, example_inputs)
        print(f"ACTIVE PARAMETERS ABSOLUTE ORIGINAL: {self.params_count_orig}")
        print(f"ACTIVE PARAMETERS ABSOLUTE PRUNED: {params_count}")
        print(f"ACTIVE PARAMETERS: {100 * (params_count / self.params_count_orig)}")
        print(f'FLOPS PRUNED: {flops_count} | ORIGINAL: {self.flops_count_orig}')

    def init_strategy(self):
        self.avoid_data_parallel()
        self.make_bins = BinarizerHighThreshold()
        self.pruning_dict = {}
        self.original_dict = {}
        named_param_list = list(self.model_teacher.named_parameters())
        for n, p in named_param_list:
            modules = n.split('.')
            new_modules = []
            for i, module in enumerate(modules):
                if i == 0:
                    nn_module = getattr(self.model_teacher, module)
                    new_modules.append(module)

                elif i + 1 < len(modules):
                    previous_nn_module = nn_module
                    nn_module = getattr(nn_module, module)
                    new_modules.append(module)

                    if isinstance(nn_module, nn.Conv2d) and not isinstance(previous_nn_module, MaskingModule):
                        setattr(previous_nn_module, module, MaskingModule(nn_module))
                        new_modules.append('moduleToMask')
                        if 'weight' in n and 3 not in p.shape[:2]:  # and p.shape[0] == p.shape[1]:
                            new_modules.append('weight')
                            self.pruning_dict['___'.join(new_modules)] = nn.Parameter(p[:, 0].clone() * 1e-4)
                            self.original_dict['___'.join(new_modules)] = p
                            delattr(nn_module, 'weight')
                            setattr(nn_module, 'weight', p.clone())

        self.pruning_dict = nn.ParameterDict(self.pruning_dict)
        self.original_dict = nn.ParameterDict(self.original_dict)
        self.optimizer_pruning = torch.optim.AdamW(self.pruning_dict.parameters(), lr=1e-4)
        self.optimizer_param = torch.optim.AdamW(self.original_dict.parameters(), lr=1e-4)
        self.cpu()
        example_inputs = torch.randn(1, 3, 1024, 1024)
        self.update_params(mode='parameters')
        self.flops_count_orig, self.params_count_orig = tp.utils.count_ops_and_params(self.model_teacher,
                                                                                      example_inputs)
        self.update_params()
        self.cuda()

    def zero_grad_and_update(self):
        if not self.test:
            self.optimizer_pruning.zero_grad()
            self.optimizer_param.zero_grad()
        self.update_params()

    def remove_and_print(self):
        if self.iter % 391 == 0:
            if self.iter not in [0, 391, 782]:
                self.remove_params()
            self.cpu()
            self.print_flops_and_params(self.model_teacher)
            self.cuda()

    def backward_and_step(self, loss):
        loss.backward()
        self.optimizer_pruning.step()
        self.optimizer_param.step()

    def avoid_data_parallel(self):

        class AvoidDataParallel(nn.Module):
            def __init__(self, module, task):
                super().__init__()
                self.module = module
                self.device = next(module.parameters()).device
                self.task = task

            if self.task == 'segmentation':
                def train_step(self, batch):
                    tmp = torch.ones(1, 19, batch.shape[-2], batch.shape[-1])
                    img_metas = [{'ori_shape': batch.shape, 'img_shape': tmp.shape, 'pad_shape': torch.zeros(1),
                                  'gt_semantic_seg': tmp, 'flip': False}]

                    losses = self.module(img=batch, img_metas=img_metas, return_loss=True,
                                         **{'gt_semantic_seg': torch.ones(1, 1,
                                                                          batch.shape[-2] // 8,
                                                                          batch.shape[-1] // 8
                                                                          ).long()})
                    loss, log_vars = self.module._parse_losses(losses)

                    return loss

            elif self.task == 'classification':
                def forward(self, x):
                    return self.module(x)

            def train_step(self, batch, optimizer, **kwargs):
                return self.module.train_step(batch, optimizer, **kwargs)

        if hasattr(self.model_teacher, 'module'):
            self.model_teacher = AvoidDataParallel(self.model_teacher.module, self.task)

    def forward(self, x):

        if self.training:
            if not self.test:
                self.remove_and_print()
            self.zero_grad_and_update()

        self.iter += 1
        self.cuda()
        return self.model_teacher(x)

    def train_step(self, batch, optimizer, **kwargs):
        # batch['img_metas'] = batch['img_metas'].data[0]
        # batch['img'] = batch['img'].data[0].to(self.device)
        # batch['gt_semantic_seg'] = batch['gt_semantic_seg'].data[0].to(self.device)

        if self.training and not self.test:
            self.remove_and_print()
            self.zero_grad_and_update()
            self.iter += 1
        return self.model_teacher.train_step(batch, optimizer, **kwargs)

    def val_step(self, batch, **kwargs):
        return self.model_teacher.val_step(batch, **kwargs)


class Strategy6(nn.Module):

    """
    HRankConv
    """

    def __init__(self, model, modules_to_prune=['layer1', 'layer2', 'layer3'], amount=[.4, .4, .4],
                 not_to_prune='prune_everything',
                 task='classification', all_modules=True):
        super().__init__()

        global progress_bar
        progress_bar = tqdm(range(110))

        self.model_teacher = model
        self.device = next(model.parameters()).device
        self.iter = 0
        self.amount = amount
        self.task = task
        self.modules_to_prune = modules_to_prune
        self.pruners = []
        self.not_to_prune = not_to_prune
        self.all_modules = all_modules


        class HRankConv(nn.Module):

            def __init__(self, module, remove):
                super().__init__()
                self.module = module
                self.prune = False
                self.ranking_function = torch.matrix_rank
                shape_0 = module.weight.shape[0]
                self.remove = int(remove * shape_0)
                self.rank = nn.Parameter(torch.arange(shape_0).float())
                self.output_features = nn.Parameter(torch.tensor(getattr(self.module, 'out_channels')).float())

            def forward(self, x):
                if self.prune:
                    self.prune = False
                    x = self.module(x)
                    self.rank = nn.Parameter(torch.argsort(torch.sum(torch.stack(
                        [torch.stack([self.ranking_function(x_ij) for x_ij in x_i])
                         for x_i in x]), dim=0))[self.remove:].float())
                    self.module.weight = nn.Parameter(self.module.weight[self.rank.clone().long()])

                    setattr(self.module, 'out_channels', self.rank.clone().shape[0])
                    progress_bar.update(1)
                    return x
                else:
                    x = self.module(x)
                    zeros = torch.zeros([x.shape[0], int(self.output_features.item())] + list(x.shape[2:])).to(x.device)
                    zeros[:, self.rank.clone().long()] = x
                    return zeros

        self.HRankConv = HRankConv
        self.init_strategy()


    def init_strategy(self):

        class PassModule(nn.Module):
            def __init__(self, module, device, task):
                super().__init__()
                self.model_teacher = module
                self.device = device
                self.task = task

            if self.task == 'segmentation':
                def forward(self, batch):
                    # batch = batch.unsqueeze(0)
                    if hasattr(batch, 'shape') and batch.shape == (1, 3, 1024, 1024):
                        # batch = torch.randn(1, 3, 1024, 1024)
                        tmp = torch.ones(1, 19, batch.shape[-2], batch.shape[-1])
                        img_metas = [{'ori_shape': batch.shape, 'img_shape': tmp.shape, 'pad_shape': None,
                                      'gt_semantic_seg': tmp, 'flip': False}]

                        losses = self.model_teacher(img=batch, img_metas=img_metas, return_loss=True,
                                                    **{'gt_semantic_seg': torch.ones(1, 1, batch.shape[-2] // 8,
                                                                                     batch.shape[-1] // 8
                                                                                     ).long()})
                    else:
                        img = batch['img'].data
                        gt_semantic_seg = torch.cat(batch['gt_semantic_seg'].data, dim=0).cuda()
                        del batch['img']
                        batch['gt_semantic_seg'] = gt_semantic_seg
                        losses = self.model_teacher(img=torch.cat(img, dim=0).cuda(), **{k: v.data for k, v in batch.items()})
                    loss, log_vars = self.model_teacher._parse_losses(losses)

                    return loss
            elif self.task == 'classification':
                def forward(self, x):
                    return self.model_teacher(x)

            def train_step(self, batch, optimizer, **kwargs):
                # batch['img_metas'] = batch['img_metas'].data[0]
                # batch['img'] = batch['img'].data[0].to(self.device)
                # batch['gt_semantic_seg'] = batch['gt_semantic_seg'].data[0].to(self.device)

                return self.model_teacher.train_step(batch, optimizer, **kwargs)

            def val_step(self, batch, **kwargs):
                return self.model_teacher.val_step(batch, **kwargs)



        def prune_specific_module(model, amount):
            if self.all_modules:
                # modules_list = [m for mn, m in model.named_modules() for n, p in m.named_parameters()
                #                 if self.not_to_prune in mn
                #                 ]
                modules_list = [
                    [m for mn, m in model.named_modules() for n, p in m.named_parameters() if isinstance(m, nn.Conv2d)][
                        -1]]

            else:
                modules_list = [m for mn, m in model.named_modules() for n, p in m.named_parameters()
                                if isinstance(m, nn.Conv2d) and 'weight' in n and p.shape[0] != p.shape[1] or
                                self.not_to_prune in mn or '17.conv1' in mn
                                ]
            named_modules = list(model.named_parameters())
            for n, p in named_modules:
                modules = n.split('.')
                if 'weight' in n:
                    for i, mn in enumerate(modules):
                        if i == 0:
                            if replace_Conv_with_HRankConv(model, mn, amount, modules_list):
                                break
                            m = getattr(model, mn)
                        else:
                            if replace_Conv_with_HRankConv(m, mn, amount, modules_list):
                                break
                            m = getattr(m, mn)

        def replace_Conv_with_HRankConv(m, mn, amount, modules_list):
            sub_module = getattr(m, mn)
            if isinstance(sub_module, nn.Conv2d) and sub_module not in modules_list and not isinstance(m, self.HRankConv):

                setattr(m, mn, self.HRankConv(module=sub_module, remove=amount))
                return True
            else:
                return False


        def prune_specific_modules():
            for module, amount in zip(self.modules_to_prune, self.amount):
                for module_name, nn_module in self.model_teacher.named_modules():
                    if module == module_name.split('.')[-1]:
                        prune_specific_module(model=nn_module, amount=amount)

        prune_specific_modules()
        # example_inputs = torch.randn(1, 3, 1024, 1024)
        # if self.task == 'segmentation':
        #     example_inputs = None
        # else:
        example_inputs = torch.randn(1, 3, 1024, 1024)

        if hasattr(self.model_teacher, 'module'):
            self.model_teacher = PassModule(self.model_teacher.module, self.device, self.task)
        else:
            self.model_teacher = PassModule(self.model_teacher, self.device, self.task)

        self.cpu()
        self.flops_count_orig, self.params_count_orig = tp.utils.count_ops_and_params(self.model_teacher,
                                                                                  example_inputs)

        self.cuda()

    def strategy(self, x):
        with torch.no_grad():
            self.activate_pruning()
            self.forward(x)

        self.cpu()
        example_inputs = torch.randn(1, 3, 1024, 1024)
        self.flops_count, _ = tp.utils.count_ops_and_params(self.model_teacher, example_inputs)
        self.params_count = self.active_params()
        self.active_parameters = 100 * (self.params_count / self.params_count_orig)
        print(f"ACTIVE PARAMETERS ABSOLUTE ORIGINAL: {self.params_count_orig}")
        print(f"ACTIVE PARAMETERS ABSOLUTE PRUNED: {self.params_count}")
        print(f"ACTIVE PARAMETERS: {self.active_parameters}")
        print(f'FLOPS PRUNED: {self.flops_count} | ORIGINAL: {self.flops_count_orig}')
        self.cuda()

    def forward(self, x):
        x = self.model_teacher(x)
        self.iter += 1
        return x

    def train_step(self, batch, optimizer, **kwargs):

        self.iter += 1
        if self.iter == 10:
            print(f"ACTIVE PARAMETERS ABSOLUTE ORIGINAL: {self.params_count_orig}")
            print(f"ACTIVE PARAMETERS ABSOLUTE PRUNED: {self.params_count}")
            print(f"ACTIVE PARAMETERS: {self.active_parameters}")
            print(f'FLOPS PRUNED: {self.flops_count} | ORIGINAL: {self.flops_count_orig}')
        return self.model_teacher.train_step(batch, optimizer, **kwargs)

    def val_step(self, batch, **kwargs):
        return self.model_teacher.val_step(batch, **kwargs)

    def activate_pruning(self):
        for mn, m in self.model_teacher.named_modules():
            if isinstance(m, self.HRankConv):
                setattr(m, 'prune', True)

    def active_params(self):
        active_params = 0
        for n, p in self.named_parameters():
            list_utils_params = ['conv_pruner', 'original_conv', 'output_features', 'indeces', 'tensor_slice', 'original']
            if torch.tensor([name not in n for name in list_utils_params]).all():
                active_params += torch.numel(p)

        return active_params

global make_bins
make_bins = Binarizer()
# make_bins = BinarizerAvg()
polarization_bins = Alpha_bin()
class Strategy7(nn.Module):

    def __init__(self, model, modules_to_prune=['layer1', 'layer2', 'layer3'], amount=[.4, .4, .4],
                 not_to_prune='prune_everything',
                 task='classification', all_modules=True, print_every=391, pad_with_zeros=[True, True, True]):  # not_to_prune='conv2',
        super().__init__()

        self.stop_def = False
        self.model_teacher = model
        self.device = next(model.parameters()).device
        self.iter = 0
        self.print_every = print_every
        self.amount = amount
        self.task = task
        self.modules_to_prune = modules_to_prune
        self.pruners = []
        self.not_to_prune = not_to_prune
        self.all_modules = all_modules
        self.pad_with_zeros = pad_with_zeros

        class PrunedBatchNorm2d(nn.Module):

            def __init__(self, norm_layer, pad_with_zeros=False):
                super().__init__()
                self.norm_layer = norm_layer
                self.weight_original = nn.Parameter(norm_layer.weight)
                self.bias_original = nn.Parameter(norm_layer.bias)
                self.running_mean_original = nn.Parameter(norm_layer.running_mean)
                self.running_var_original = nn.Parameter(norm_layer.running_var)
                self.output_features = nn.Parameter(torch.ones(1) * norm_layer.weight.shape[0])
                self.tensor_to_pass = None
                self.bool_mask = None
                self.pad_with_zeros = pad_with_zeros

            def forward(self, x):
                if self.tensor_to_pass is not None:
                    # self.bool_mask[:] = False
                    self.tensor_to_pass[:] = 0
                    self.tensor_to_pass = self.tensor_to_pass.detach()
                self.train()
                index_slice = torch.tensor([i for i, xi in enumerate(torch.sum(x, dim=(0,-1, -2))) if not xi == 0]).to(x.device)

                if self.tensor_to_pass is None:
                    self.tensor_to_pass = torch.zeros([x.shape[0], int(self.output_features.clone().item())] + list(x.shape[2:]))
                self.bool_mask = torch.tensor([True if i in index_slice.long() else False for i in range(x.shape[1])]).bool()

                self.tensor_to_pass = self.tensor_to_pass.to(x.device)
                self.bool_mask = self.bool_mask.to(x.device)
                # tensor_to_pass = self.tensor_to_pass
                self.bool_mask[index_slice.long()] = True
                if torch.sum(self.bool_mask) == 1:
                    self.bool_mask[self.bool_mask == False][0] = True

                self.norm_layer.weight = nn.Parameter(self.weight_original[self.bool_mask].clone())
                self.norm_layer.bias = nn.Parameter(self.bias_original[self.bool_mask].clone())
                self.norm_layer.running_mean = self.running_mean_original[self.bool_mask].clone().detach()
                self.norm_layer.running_var = self.running_var_original[self.bool_mask].clone().detach()
                # x = x[:, self.bool_mask]
                if self.pad_with_zeros:
                    x = self.norm_layer(x[:, self.bool_mask])
                    if x.shape[0] != self.tensor_to_pass.shape[0]:
                        tmp = torch.zeros([x.shape[0], int(self.output_features.item())] + list(x.shape[2:])).to(x.device)
                        tmp[:, self.bool_mask] = x
                        return tmp
                    self.tensor_to_pass[:, self.bool_mask] = x
                    return self.tensor_to_pass
                else:
                    y = self.norm_layer(x[:, self.bool_mask])
                    x[:, self.bool_mask] = y
                    return x



        class PruningConv(nn.Module):

            def __init__(self, module, remove, name, pad_with_zeros=False):
                super().__init__()
                self.module = module
                self.prune = True
                self.name = name
                shape_0 = module.weight.shape[0]
                self.shape_0 = shape_0
                self.shape_1 = self.module.weight.shape[1]
                self.keep = int((1 - remove) * shape_0)
                self.conv_pruner = nn.Parameter(torch.ones_like(module.weight.clone()[0]) * 1e-2)
                self.original_conv = nn.Parameter(module.weight.clone())
                self.output_features = nn.Parameter(torch.tensor(getattr(self.module, 'out_channels')).float())
                self.indeces = nn.Parameter(torch.ones(shape_0).float())
                delattr(self.module, 'weight')
                setattr(self.module, 'weight', self.original_conv.clone())
                self.reached_threshold = False
                self.wrong_behaviour = False
                self.tensor_slice = nn.Parameter(torch.arange(self.shape_1).float())
                self.tensor_to_pass = None
                self.pad_with_zeros = pad_with_zeros

                class EQ(nn.Module):

                    def forward(self, x):
                        return x

                self.eq = EQ

            def forward(self, x):

                if self.tensor_to_pass is not None:
                    self.tensor_to_pass[:] = 0
                    self.tensor_to_pass = self.tensor_to_pass.detach()
                indeces = None
                if isinstance(self.module, nn.Conv2d):
                    self.tensor_slice = nn.Parameter(torch.tensor(
                        [j for j, xi in enumerate(torch.sum(x, dim=(0, -1, -2))) if not xi == 0]).float())
                    mask_x = torch.tensor(
                        [True if i in self.tensor_slice.long() else False for i in range(x.shape[1])]).bool()

                    mask_w = torch.tensor(
                        [True if i in self.tensor_slice.long() else False for i in range(int(self.output_features.clone().item()))]).bool()

                    # tmp = torch.transpose(torch.transpose(self.original_conv, 0, 1)[tensor_slice], 0, 1)
                    # if isinstance(self.module.weight, nn.Parameter):
                    #     self.module.weight = nn.Parameter(tmp)
                    # else:
                    #     self.module.weight = tmp

                    if self.prune and torch.sum(self.indeces) > self.keep and self.training:
                        original_sliced = self.original_conv.clone()[:, mask_w]
                        self.module.in_channels = original_sliced.shape[1]

                        p = torch.tensordot(self.conv_pruner.clone()[self.tensor_slice.long()], torch.abs(original_sliced)
                                            , dims=[(0, 1, 2), (1, 2, 3)])
                        # self.p = p.clone().detach()
                        indeces = make_bins.apply(p)
                        if torch.sum(indeces) != 0:
                            self.indeces = nn.Parameter(indeces.float())
                            tmp = torch.transpose(torch.transpose(original_sliced, 0, -1
                                                                                 ) * indeces, 0, -1)[self.indeces.bool()]
                            indeces = 'not to use'
                            if isinstance(self.module.weight, nn.Parameter):
                                self.module.weight = nn.Parameter(tmp)

                            else:
                                self.module.weight = tmp
                            # self.module.bias = self.module.bias[self.indeces.bool()]
                        elif not self.wrong_behaviour:
                            self.wrong_behaviour = True
                            indeces = 'not to use'
                            self.stop()
                            print(f'{self.name} tried to remove everything')
                            if self.shape_0 == self.shape_1:
                                print('REPLACING WITH EQ MODULE')
                                self.module = self.eq()
                                return x

                    else:
                        if torch.sum(self.indeces) <= self.keep and not self.reached_threshold:
                            self.reached_threshold = True
                            print(f'{self.name} stopped pruning')
                        self.stop()
                        indeces = 'not to use'
                    if self.pad_with_zeros:
                        x = x[:, mask_x]
                        x = self.module(x)

                        if torch.sum(self.indeces) == self.indeces.shape[0] or self.module.weight.shape[0] == \
                                self.indeces.shape[0]:
                            return x
                        else:
                            if self.tensor_to_pass is None:
                                self.tensor_to_pass = torch.zeros(
                                    [x.shape[0], int(self.output_features.item())] + list(x.shape[2:]))
                            self.tensor_to_pass = self.tensor_to_pass.to(x.device)
                            # tensor_to_pass = self.tensor_to_pass
                            # mask_x_out = torch.tensor(
                            #     [True if i in self.indeces.long() else False for i in range(int(self.output_features.item()))]).bool()
                            if x.shape[0] != self.tensor_to_pass.shape[0]:
                                tmp = torch.zeros([x.shape[0], int(self.output_features.item())] + list(x.shape[2:])).to(
                                    x.device)
                                tmp[:, self.indeces.bool()] = x
                                return tmp
                            self.tensor_to_pass[:, self.indeces.bool()] = x
                            return self.tensor_to_pass
                    else:
                        y = self.module(x[:, mask_x])
                        x[:, self.indeces.bool()] = y
                        return x


                if isinstance(self.module, self.eq):
                    return self.module(x)



            def stop(self):
                if isinstance(self.module, self.eq):
                    self.prune = False
                    return
                tmp = torch.transpose(torch.transpose(self.original_conv, 0, 1)[self.tensor_slice.long()], 0, 1)[self.indeces.bool()]
                # tmp = self.original_conv.clone()[self.indeces.bool()]
                if isinstance(self.module.weight, nn.Parameter):
                    self.module.weight = nn.Parameter(tmp)
                else:
                    self.module.weight = tmp
                # self.module.bias = self.module.bias[self.indeces.bool()]
                setattr(self.module, 'out_channels', int(torch.sum(self.indeces.clone()).item()))
                self.prune = False

            def restart(self):
                self.prune = True

            # def return_mask(self):
            #     return torch.mean(self.p)

        self.PruningConv = PruningConv
        self.PrunedBatchNorm2d = PrunedBatchNorm2d
        self.init_strategy()


    def init_strategy(self):

        class PassModule(nn.Module):
            def __init__(self, module, device, task):
                super().__init__()
                self.model_teacher = module
                self.device = device
                self.task = task

            if self.task == 'segmentation':
                def forward(self, batch):
                    # batch = batch.unsqueeze(0)
                    tmp = torch.ones(1, 19, batch.shape[-2], batch.shape[-1])
                    img_metas = [{'ori_shape': batch.shape, 'img_shape': tmp.shape, 'pad_shape': None,
                                  'gt_semantic_seg': tmp, 'flip': False}]

                    losses = self.model_teacher.model_teacher(img=batch, img_metas=img_metas, return_loss=True,
                                                **{'gt_semantic_seg': torch.ones(1, 1, batch.shape[-2] // 8,
                                                                                 batch.shape[-1] // 8
                                                                                 ).long()})
                    loss, log_vars = self.model_teacher.model_teacher._parse_losses(losses)

                    return loss
            elif self.task == 'classification':
                def forward(self, x):
                    return self.model_teacher(x)

            def train_step(self, batch, optimizer, **kwargs):
                # batch['img_metas'] = batch['img_metas'].data[0]
                # batch['img'] = batch['img'].data[0].to(self.device)
                # batch['gt_semantic_seg'] = batch['gt_semantic_seg'].data[0].to(self.device)

                return self.model_teacher.train_step(batch, optimizer, **kwargs)

            def val_step(self, batch, **kwargs):
                return self.model_teacher.val_step(batch, **kwargs)



        def prune_specific_module(model, amount, name, pad_with_zeros):
            if self.all_modules:
                modules_list = [m for mn, m in model.named_modules() for n, p in m.named_parameters()
                                if self.not_to_prune in mn
                                ]
            else:
                modules_list = [m for mn, m in model.named_modules() for n, p in m.named_parameters()
                                if isinstance(m, nn.Conv2d) and 'weight' in n and p.shape[0] != p.shape[1] or
                                self.not_to_prune in mn or '17.conv1' in mn
                                ]
            named_modules = list(model.named_parameters())
            for n, p in named_modules:
                modules = n.split('.')
                if 'weight' in n:
                    for i, mn in enumerate(modules):
                        if i == 0:
                            if replace_Conv_with_PruningConv(model, mn, amount, modules_list, name=name+'.'+n, pad_with_zeros=pad_with_zeros):
                                break
                            if replace_BatchNorm2d_with_PrunedBatchNorm2d(model, mn, modules_list, pad_with_zeros=pad_with_zeros):
                                break
                            m = getattr(model, mn)
                        else:
                            if replace_Conv_with_PruningConv(m, mn, amount, modules_list, name=name+'.'+n, pad_with_zeros=pad_with_zeros):
                                break
                            if replace_BatchNorm2d_with_PrunedBatchNorm2d(m, mn, modules_list, pad_with_zeros=pad_with_zeros):
                                break
                            m = getattr(m, mn)

        def replace_Conv_with_PruningConv(m, mn, amount, modules_list, name, pad_with_zeros):
            sub_module = getattr(m, mn)
            if isinstance(sub_module, nn.Conv2d) and sub_module not in modules_list and not isinstance(m, self.PruningConv):

                setattr(m, mn, self.PruningConv(module=sub_module, remove=amount, name=name, pad_with_zeros=pad_with_zeros))
                return True
            else:
                return False

        def replace_BatchNorm2d_with_PrunedBatchNorm2d(m, mn, modules_list, pad_with_zeros):
            sub_module = getattr(m, mn)
            if isinstance(sub_module, nn.BatchNorm2d) and sub_module not in modules_list and not isinstance(m, self.PrunedBatchNorm2d):

                setattr(m, mn, self.PrunedBatchNorm2d(norm_layer=sub_module, pad_with_zeros=pad_with_zeros))
                return True
            else:
                return False



        def prune_specific_modules():
            for module, amount, pad_with_zeros in zip(self.modules_to_prune, self.amount, self.pad_with_zeros):
                for module_name, nn_module in self.model_teacher.named_modules():
                    if module == module_name.split('.')[-1]:
                        prune_specific_module(model=nn_module, amount=amount, name=module, pad_with_zeros=pad_with_zeros)

        prune_specific_modules()
        example_inputs = torch.randn(1, 3, 1024, 1024)

        if hasattr(self.model_teacher, 'module'):
            self.model_teacher = PassModule(self.model_teacher.module, self.device, self.task)
        else:
            self.model_teacher = PassModule(self.model_teacher, self.device, self.task)

        self.cpu()
        # self.optimizer = torch.optim.AdamW([p for n, p in self.named_parameters() if 'conv_pruner' in n], lr=1e-5)
        self.weight_to_param()

        self.flops_count_orig, _ = tp.utils.count_ops_and_params(self.model_teacher, example_inputs)
        self.params_count_orig = self.active_params()

        self.param_to_weight()
        self.cuda()

    def strategy(self, x):
        pass

    def forward(self, x):

        if self.iter % self.print_every == 0 and self.training:
            self.stop_pruning()
            if not self.stop_def:
                self.restart_pruning()

        if self.training:
            self.iter += 1
        x = self.model_teacher(x)
        return x

    def train_step(self, batch, optimizer, **kwargs):

        if self.iter % self.print_every == 0 and self.training:
            self.stop_pruning()
            if not self.stop_def:
                self.restart_pruning()

        if self.training:
            self.iter += 1
        return self.model_teacher.train_step(batch, optimizer, **kwargs)

    def val_step(self, batch, **kwargs):
        return self.model_teacher.val_step(batch, **kwargs)

    def stop_pruning(self):
        self.weight_to_param()
        for mn, m in self.model_teacher.named_modules():
            if isinstance(m, self.PruningConv):
               m.stop()
        self.print_flops_and_params()
        self.param_to_weight()

    def weight_to_param(self):
        for mn, m in self.model_teacher.named_modules():
            if isinstance(m, self.PruningConv) and isinstance(m.module, nn.Conv2d):
                setattr(m.module, 'weight', nn.Parameter(getattr(m.module, 'weight')))

    def param_to_weight(self):
        for mn, m in self.model_teacher.named_modules():
            if isinstance(m, self.PruningConv) and isinstance(m.module, nn.Conv2d):
                w = getattr(m.module, 'weight')
                delattr(m.module, 'weight')
                setattr(m.module, 'weight', w.clone())

    def restart_pruning(self):
        for mn, m in self.model_teacher.named_modules():
            if isinstance(m, self.PruningConv):
                m.restart()

    def print_flops_and_params(self):
        self.cpu()
        example_inputs = torch.randn(1, 3, 1024, 1024)
        self.flops_count, _ = tp.utils.count_ops_and_params(self.model_teacher, example_inputs)
        self.params_count = self.active_params()
        self.active_parameters = 100 * (self.params_count / self.params_count_orig)
        print(f"ACTIVE PARAMETERS ABSOLUTE ORIGINAL: {self.params_count_orig}")
        print(f"ACTIVE PARAMETERS ABSOLUTE PRUNED: {self.params_count}")
        print(f"ACTIVE PARAMETERS: {self.active_parameters}")
        print(f'FLOPS PRUNED: {self.flops_count} | ORIGINAL: {self.flops_count_orig}')
        self.cuda()

    def active_params(self):
        active_params = 0
        for n, p in self.named_parameters():
            list_utils_params = ['conv_pruner', 'original_conv', 'output_features', 'indeces', 'tensor_slice', 'original']
            if torch.tensor([name not in n for name in list_utils_params]).all():
                active_params += torch.numel(p)

        return active_params

    def backward_and_step(self):
        # self.optimizer.zero_grad()
        # torch.mean(torch.stack([m.return_mask() for mn, m in self.model_teacher.named_modules()
        #                        if isinstance(m, self.PruningConv)])).backward(retain_graph=True)
        # self.optimizer.step()
        pass

    def stop_def_met(self):
        self.stop_def = True


class Strategy8(nn.Module):

    def __init__(self, model, modules_to_prune=['layer1', 'layer2', 'layer3'], amount=[.4, .4, .4],
                 not_to_prune='prune_everything',
                 task='classification', all_modules=True, print_every=391, pad_with_zeros=[True, True, True]):  # not_to_prune='conv2',
        super().__init__()

        self.stop_def = False
        self.model_teacher = model
        self.device = next(model.parameters()).device
        self.iter = 0
        self.print_every = print_every
        self.amount = amount
        self.task = task
        self.modules_to_prune = modules_to_prune
        self.pruners = []
        self.not_to_prune = not_to_prune
        self.all_modules = all_modules
        self.pad_with_zeros = pad_with_zeros

        class PrunedBatchNorm2d(nn.Module):

            def __init__(self, norm_layer, pad_with_zeros=False):
                super().__init__()
                self.norm_layer = norm_layer
                self.weight_original = nn.Parameter(norm_layer.weight)
                self.bias_original = nn.Parameter(norm_layer.bias)
                self.running_mean_original = nn.Parameter(norm_layer.running_mean)
                self.running_var_original = nn.Parameter(norm_layer.running_var)
                self.output_features = nn.Parameter(torch.ones(1) * norm_layer.weight.shape[0])
                self.tensor_to_pass = None
                self.bool_mask = None
                self.pad_with_zeros = pad_with_zeros

            def forward(self, x):
                if self.tensor_to_pass is not None:
                    # self.bool_mask[:] = False
                    self.tensor_to_pass[:] = 0
                    self.tensor_to_pass = self.tensor_to_pass.detach()
                self.train()
                index_slice = torch.tensor([i for i, xi in enumerate(torch.sum(x, dim=(0,-1, -2))) if not xi == 0]).to(x.device)

                if self.tensor_to_pass is None:
                    self.tensor_to_pass = torch.zeros([x.shape[0], int(self.output_features.clone().item())] + list(x.shape[2:]))
                self.bool_mask = torch.tensor([True if i in index_slice.long() else False for i in range(x.shape[1])]).bool()

                self.tensor_to_pass = self.tensor_to_pass.to(x.device)
                self.bool_mask = self.bool_mask.to(x.device)
                # tensor_to_pass = self.tensor_to_pass
                self.bool_mask[index_slice.long()] = True
                if torch.sum(self.bool_mask) == 1:
                    self.bool_mask[self.bool_mask == False][0] = True

                self.norm_layer.weight = nn.Parameter(self.weight_original[self.bool_mask].clone())
                self.norm_layer.bias = nn.Parameter(self.bias_original[self.bool_mask].clone())
                self.norm_layer.running_mean = self.running_mean_original[self.bool_mask].clone().detach()
                self.norm_layer.running_var = self.running_var_original[self.bool_mask].clone().detach()
                # x = x[:, self.bool_mask]
                if self.pad_with_zeros:
                    x = self.norm_layer(x[:, self.bool_mask])
                    if x.shape[0] != self.tensor_to_pass.shape[0]:
                        tmp = torch.zeros([x.shape[0], int(self.output_features.item())] + list(x.shape[2:])).to(x.device)
                        tmp[:, self.bool_mask] = x
                        return tmp
                    self.tensor_to_pass[:, self.bool_mask] = x
                    return self.tensor_to_pass
                else:
                    y = self.norm_layer(x[:, self.bool_mask])
                    x[:, self.bool_mask] = y
                    return x



        class PruningConv(nn.Module):

            def __init__(self, module, remove, name, pad_with_zeros=True):
                super().__init__()
                self.module = module
                self.prune = True
                self.name = name
                shape_0 = module.weight.shape[0]
                self.shape_0 = shape_0
                self.shape_1 = self.module.weight.shape[1]
                self.keep = int((1 - remove) * shape_0)
                # self.conv_pruner = nn.Parameter(torch.ones_like(module.weight.clone()[0, :, 0, 0]))
                self.conv_pruner = nn.Parameter(torch.mean(torch.abs(module.weight.clone()), dim=(1,2,3)))
                # self.conv_pruner = nn.Parameter(torch.mean(torch.abs(module.weight.clone()), dim=(0,2,3)))
                self.original_conv = nn.Parameter(module.weight.clone())
                self.output_features = nn.Parameter(torch.tensor(getattr(self.module, 'out_channels')).float())
                self.indeces = nn.Parameter(torch.ones(shape_0).float())
                delattr(self.module, 'weight')
                setattr(self.module, 'weight', self.original_conv.clone())
                self.reached_threshold = False
                self.wrong_behaviour = False
                self.tensor_slice = nn.Parameter(torch.arange(self.shape_1).float())
                self.tensor_to_pass = None
                self.pad_with_zeros = pad_with_zeros

                class EQ(nn.Module):

                    def forward(self, x):
                        return x

                self.eq = EQ

            def forward(self, x):

                if self.tensor_to_pass is not None:
                    self.tensor_to_pass[:] = 0
                    self.tensor_to_pass = self.tensor_to_pass.detach()
                indeces = None
                if isinstance(self.module, nn.Conv2d):
                    self.tensor_slice = nn.Parameter(torch.tensor(
                        [j for j, xi in enumerate(torch.sum(x, dim=(0, -1, -2))) if not xi == 0]).float())
                    mask_x = torch.tensor(
                        [True if i in self.tensor_slice.long() else False for i in range(x.shape[1])]).bool()

                    # mask_w = torch.tensor(
                    #     [True if i in self.tensor_slice.long() else False for i in range(int(self.output_features.clone().item()))]).bool()

                    # tmp = torch.transpose(torch.transpose(self.original_conv, 0, 1)[tensor_slice], 0, 1)
                    # if isinstance(self.module.weight, nn.Parameter):
                    #     self.module.weight = nn.Parameter(tmp)
                    # else:
                    #     self.module.weight = tmp

                    self.sss(mask_x)
                    y = self.module(x[:, mask_x])
                    # if self.prune:
                    #     y = y[:, self.indeces.bool()]
                        # y = torch.transpose((torch.transpose(y, 1, -1) * self.conv_pruner[self.indeces.bool()]), 1, -1)


                    if self.pad_with_zeros:
                        x = y

                        # if torch.sum(self.indeces) == self.indeces.shape[0] or self.module.weight.shape[0] == \
                        #         self.indeces.shape[0]:
                        #     return x
                        # else:
                        if self.tensor_to_pass is None and x.shape[0] != 1:
                            self.tensor_to_pass = torch.zeros(
                                [x.shape[0], int(self.output_features.item())] + list(x.shape[2:]))
                        if self.tensor_to_pass is not None:
                            self.tensor_to_pass = self.tensor_to_pass.to(x.device)
                        # tensor_to_pass = self.tensor_to_pass
                        # mask_x_out = torch.tensor(
                        #     [True if i in self.indeces.long() else False for i in range(int(self.output_features.item()))]).bool()
                        if self.tensor_to_pass is None or x.shape[0] != self.tensor_to_pass.shape[0]:
                            tmp = torch.zeros([x.shape[0], int(self.output_features.item())] + list(x.shape[2:])).to(
                                x.device)
                            tmp[:, self.indeces.bool()] = x
                            return tmp
                        self.tensor_to_pass[:, self.indeces.bool()] = x
                        return self.tensor_to_pass
                    else:
                        x[:, self.indeces.bool()] = y
                        return x


                if isinstance(self.module, self.eq):
                    return self.module(x)

            def sss(self, mask_w):
                if self.prune and torch.sum(self.indeces) > self.keep and self.training:
                    original_sliced = self.original_conv.clone()[:, mask_w]
                    self.module.in_channels = original_sliced.shape[1]


                    # p = torch.tensordot(self.conv_pruner.clone()[self.tensor_slice.long()], torch.abs(original_sliced)
                    #                     , dims=[(0, 1, 2), (1, 2, 3)])
                    # self.p = p.clone().detach()
                    indeces = make_bins.apply(self.conv_pruner.clone())
                    if torch.sum(indeces) > 32:
                        self.indeces = nn.Parameter(indeces.float())
                        tmp = torch.transpose(torch.transpose(original_sliced, 0, -1
                                                              ) * indeces, 0, -1)[self.indeces.bool()]
                        indeces = 'not to use'
                        if isinstance(self.module.weight, nn.Parameter):
                            self.module.weight = nn.Parameter(tmp)

                        else:
                            self.module.weight = tmp

                    else:
                        tmp = []
                        counter = 32
                        len_tmp = len(indeces)
                        for c, i in enumerate(indeces):
                            if (counter > 0 and i == 1) or len_tmp - c - 1 < counter:
                                counter -= 1
                                tmp.append(1)
                            else:
                                tmp.append(0)

                        tmp = torch.tensor(tmp)
                        self.indeces = nn.Parameter(tmp.float())
                        original_sliced = original_sliced[self.indeces.bool()]
                        if isinstance(self.module.weight, nn.Parameter):
                            self.module.weight = nn.Parameter(original_sliced)

                        else:
                            self.module.weight = original_sliced
                        # self.module.bias = self.module.bias[self.indeces.bool()]
                    # elif not self.wrong_behaviour:
                    #     self.wrong_behaviour = True
                    #     indeces = 'not to use'
                    #     self.stop()
                    #     print(f'{self.name} tried to remove everything')
                    #     if self.shape_0 == self.shape_1:
                    #         print('REPLACING WITH EQ MODULE')
                    #         self.module = self.eq()

                else:
                    if torch.sum(self.indeces) <= self.keep and not self.reached_threshold:
                        self.reached_threshold = True
                        print(f'{self.name} stopped pruning')
                    self.stop()
                    indeces = 'not to use'

            def stop(self):
                if isinstance(self.module, self.eq):
                    self.prune = False
                    return
                tmp = torch.transpose(torch.transpose(self.original_conv, 0, 1)[self.tensor_slice.long()], 0, 1)[self.indeces.bool()]
                # tmp = self.original_conv.clone()[self.indeces.bool()]
                if isinstance(self.module.weight, nn.Parameter):
                    self.module.weight = nn.Parameter(tmp)
                else:
                    self.module.weight = tmp
                # self.module.bias = self.module.bias[self.indeces.bool()]
                setattr(self.module, 'out_channels', int(torch.sum(self.indeces.clone()).item()))
                self.prune = False

            def restart(self):
                self.prune = True

            # def return_mask(self):
            #     return torch.mean(self.p)

        self.PruningConv = PruningConv
        self.PrunedBatchNorm2d = PrunedBatchNorm2d
        self.init_strategy()


    def init_strategy(self):

        class PassModule(nn.Module):
            def __init__(self, module, device, task):
                super().__init__()
                self.model_teacher = module
                self.device = device
                self.task = task


            if self.task == 'segmentation':
                def forward(self, batch):
                    # batch = batch.unsqueeze(0)
                    if hasattr(batch, 'shape') and batch.shape == (1, 3, 1024, 1024):
                        # batch = torch.randn(1, 3, 1024, 1024)
                        tmp = torch.ones(1, 19, batch.shape[-2], batch.shape[-1])
                        img_metas = [{'ori_shape': batch.shape, 'img_shape': tmp.shape, 'pad_shape': None,
                                      'gt_semantic_seg': tmp, 'flip': False}]

                        losses = self.model_teacher(img=batch, img_metas=img_metas, return_loss=True,
                                                    **{'gt_semantic_seg': torch.ones(1, 1, batch.shape[-2] // 8,
                                                                                     batch.shape[-1] // 8
                                                                                     ).long()})
                    else:
                        img = batch['img'].data
                        gt_semantic_seg = torch.cat(batch['gt_semantic_seg'].data, dim=0).cuda()
                        del batch['img']
                        batch['gt_semantic_seg'] = gt_semantic_seg
                        losses = self.model_teacher(img=torch.cat(img, dim=0).cuda(),
                                                    **{k: v.data for k, v in batch.items()})
                    loss, log_vars = self.model_teacher._parse_losses(losses)

                    return loss
            elif self.task == 'classification':
                def forward(self, x):
                    return self.model_teacher(x)

            def train_step(self, batch, optimizer, **kwargs):
                # batch['img_metas'] = batch['img_metas'].data[0]
                # batch['img'] = batch['img'].data[0].to(self.device)
                # batch['gt_semantic_seg'] = batch['gt_semantic_seg'].data[0].to(self.device)

                return self.model_teacher.train_step(batch, optimizer, **kwargs)

            def val_step(self, batch, **kwargs):
                return self.model_teacher.val_step(batch, **kwargs)



        def prune_specific_module(model, amount, name, pad_with_zeros):
            if self.all_modules:
                # modules_list = [m for mn, m in model.named_modules() for n, p in m.named_parameters()
                #                 if self.not_to_prune in mn
                #                 ]
                modules_list = [
                    [m for mn, m in model.named_modules() for n, p in m.named_parameters() if isinstance(m, nn.Conv2d)][
                        -1]]
            else:
                modules_list = [m for mn, m in model.named_modules() for n, p in m.named_parameters()
                                if isinstance(m, nn.Conv2d) and 'weight' in n and p.shape[0] != p.shape[1] or
                                self.not_to_prune in mn or '17.conv1' in mn
                                ]
            named_modules = list(model.named_parameters())
            for n, p in named_modules:
                modules = n.split('.')
                if 'weight' in n:
                    for i, mn in enumerate(modules):
                        if i == 0:
                            if replace_Conv_with_PruningConv(model, mn, amount, modules_list, name=name+'.'+n, pad_with_zeros=pad_with_zeros):
                                break
                            if replace_BatchNorm2d_with_PrunedBatchNorm2d(model, mn, modules_list, pad_with_zeros=pad_with_zeros):
                                break
                            m = getattr(model, mn)
                        else:
                            if replace_Conv_with_PruningConv(m, mn, amount, modules_list, name=name+'.'+n, pad_with_zeros=pad_with_zeros):
                                break
                            if replace_BatchNorm2d_with_PrunedBatchNorm2d(m, mn, modules_list, pad_with_zeros=pad_with_zeros):
                                break
                            m = getattr(m, mn)

        def replace_Conv_with_PruningConv(m, mn, amount, modules_list, name, pad_with_zeros):
            sub_module = getattr(m, mn)
            if isinstance(sub_module, nn.Conv2d) and sub_module not in modules_list and not isinstance(m, self.PruningConv):

                setattr(m, mn, self.PruningConv(module=sub_module, remove=amount, name=name, pad_with_zeros=pad_with_zeros))
                return True
            else:
                return False

        def replace_BatchNorm2d_with_PrunedBatchNorm2d(m, mn, modules_list, pad_with_zeros):
            sub_module = getattr(m, mn)
            if isinstance(sub_module, nn.BatchNorm2d) and sub_module not in modules_list and not isinstance(m, self.PrunedBatchNorm2d):

                setattr(m, mn, self.PrunedBatchNorm2d(norm_layer=sub_module, pad_with_zeros=pad_with_zeros))
                return True
            else:
                return False



        def prune_specific_modules():
            for module, amount, pad_with_zeros in zip(self.modules_to_prune, self.amount, self.pad_with_zeros):
                for module_name, nn_module in self.model_teacher.named_modules():
                    if module == module_name.split('.')[-1]:
                        prune_specific_module(model=nn_module, amount=amount, name=module, pad_with_zeros=pad_with_zeros)

        prune_specific_modules()
        example_inputs = torch.randn(1, 3, 1024, 1024)

        if hasattr(self.model_teacher, 'module'):
            self.model_teacher = PassModule(self.model_teacher.module, self.device, self.task)
        else:
            self.model_teacher = PassModule(self.model_teacher, self.device, self.task)

        self.cpu()
        # self.optimizer = torch.optim.AdamW([p for n, p in self.named_parameters() if 'conv_pruner' in n], lr=1e-5)
        self.weight_to_param()

        self.flops_count_orig, _ = tp.utils.count_ops_and_params(self.model_teacher, example_inputs)
        self.params_count_orig = self.active_params()

        self.param_to_weight()
        self.cuda()

    def strategy(self, x):
        pass

    def forward(self, x):

        if self.iter % self.print_every == 0 and self.training:
            self.stop_pruning()
            if not self.stop_def:
                self.restart_pruning()

        if self.training:
            self.iter += 1
        x = self.model_teacher(x)
        return x

    def train_step(self, batch, optimizer, **kwargs):

        if self.iter % self.print_every == 0 and self.training:
            self.stop_pruning()
            if not self.stop_def:
                self.restart_pruning()

        if self.training:
            self.iter += 1
        return self.model_teacher.train_step(batch, optimizer, **kwargs)

    def val_step(self, batch, **kwargs):
        return self.model_teacher.val_step(batch, **kwargs)

    def stop_pruning(self):
        self.weight_to_param()
        for mn, m in self.model_teacher.named_modules():
            if isinstance(m, self.PruningConv):
               m.stop()
        self.print_flops_and_params()
        self.param_to_weight()

    def weight_to_param(self):
        for mn, m in self.model_teacher.named_modules():
            if isinstance(m, self.PruningConv) and isinstance(m.module, nn.Conv2d):
                setattr(m.module, 'weight', nn.Parameter(getattr(m.module, 'weight')))

    def weight_detach(self):
        self.parameters_to_reattach = list(self.named_parameters())
        for n, p in self.parameters_to_reattach:
            modules = n.split('.')
            for i, module in enumerate(modules):
                if i == 0:
                    nn_module = getattr(self, module)
                elif i < len(modules) - 1:
                    nn_module = getattr(nn_module, module)
                else:
                    current_param = getattr(nn_module, module)
                    delattr(nn_module, module)
                    setattr(nn_module, module, current_param.clone().detach())


    def weight_attach(self):
        tmp = self.parameters_to_reattach
        for n, p in tmp:
            modules = n.split('.')
            for i, module in enumerate(modules):
                if i == 0:
                    nn_module = getattr(self, module)
                elif i < len(modules) - 1:
                    nn_module = getattr(nn_module, module)
                else:
                    current_param = getattr(nn_module, module)
                    setattr(nn_module, module, nn.Parameter(current_param.clone().detach()))

    def param_to_weight(self):
        for mn, m in self.model_teacher.named_modules():
            if isinstance(m, self.PruningConv) and isinstance(m.module, nn.Conv2d):
                w = getattr(m.module, 'weight')
                delattr(m.module, 'weight')
                setattr(m.module, 'weight', w.clone())

    def restart_pruning(self):
        for mn, m in self.model_teacher.named_modules():
            if isinstance(m, self.PruningConv):
                m.restart()

    def print_flops_and_params(self):
        self.weight_to_param()
        self.cpu()
        # self.weight_detach()
        example_inputs = torch.randn(1, 3, 1024, 1024)
        # self.flops_count, _ = tp.utils.count_ops_and_params(self.model_teacher, example_inputs)
        self.params_count = self.active_params()
        self.active_parameters = 100 * (self.params_count / self.params_count_orig)
        print(f"ACTIVE PARAMETERS ABSOLUTE ORIGINAL: {self.params_count_orig}")
        print(f"ACTIVE PARAMETERS ABSOLUTE PRUNED: {self.params_count}")
        print(f"ACTIVE PARAMETERS: {self.active_parameters}")
        # print(f'FLOPS PRUNED: {self.flops_count} | ORIGINAL: {self.flops_count_orig}')
        # self.weight_attach()
        self.cuda()

    def active_params(self):
        active_params = 0
        for n, p in self.named_parameters():
            list_utils_params = ['conv_pruner', 'original_conv', 'output_features', 'indeces', 'tensor_slice', 'original']
            if torch.tensor([name not in n for name in list_utils_params]).all():
                active_params += torch.numel(p)

        return active_params

    def backward_and_step(self):
        # self.optimizer.zero_grad()
        # torch.mean(torch.stack([m.return_mask() for mn, m in self.model_teacher.named_modules()
        #                        if isinstance(m, self.PruningConv)])).backward(retain_graph=True)
        # self.optimizer.step()
        pass

    def stop_def_met(self):
        # self.weight_detach()
        self.stop_def = True
        self.stop_pruning()


class Strategy9(nn.Module):

    """
    L1 method
    """

    def __init__(self, model, modules_to_prune=['layer1', 'layer2', 'layer3'], amount=[.4, .4, .4],
                 not_to_prune='prune_everything',
                 task='classification', all_modules=True, print_every=391, pad_with_zeros=[True, True, True]):  # not_to_prune='conv2',
        super().__init__()

        self.stop_def = False
        self.model_teacher = model
        self.device = next(model.parameters()).device
        self.iter = 0
        self.print_every = print_every
        self.amount = amount
        self.task = task
        self.modules_to_prune = modules_to_prune
        self.pruners = []
        self.not_to_prune = not_to_prune
        self.all_modules = all_modules
        self.pad_with_zeros = pad_with_zeros

        class PrunedBatchNorm2d(nn.Module):

            def __init__(self, norm_layer, pad_with_zeros=False):
                super().__init__()
                self.norm_layer = norm_layer
                self.weight_original = nn.Parameter(norm_layer.weight)
                self.bias_original = nn.Parameter(norm_layer.bias)
                self.running_mean_original = nn.Parameter(norm_layer.running_mean)
                self.running_var_original = nn.Parameter(norm_layer.running_var)
                self.output_features = nn.Parameter(torch.ones(1) * norm_layer.weight.shape[0])
                self.tensor_to_pass = None
                self.bool_mask = None
                self.pad_with_zeros = pad_with_zeros

            def forward(self, x):
                if self.tensor_to_pass is not None:
                    # self.bool_mask[:] = False
                    self.tensor_to_pass[:] = 0
                    self.tensor_to_pass = self.tensor_to_pass.detach()
                self.train()
                index_slice = torch.tensor([i for i, xi in enumerate(torch.sum(x, dim=(0,-1, -2))) if not xi == 0]).to(x.device)

                if self.tensor_to_pass is None:
                    self.tensor_to_pass = torch.zeros([x.shape[0], int(self.output_features.clone().item())] + list(x.shape[2:]))
                self.bool_mask = torch.tensor([True if i in index_slice.long() else False for i in range(x.shape[1])]).bool()

                self.tensor_to_pass = self.tensor_to_pass.to(x.device)
                self.bool_mask = self.bool_mask.to(x.device)
                # tensor_to_pass = self.tensor_to_pass
                self.bool_mask[index_slice.long()] = True
                if torch.sum(self.bool_mask) == 1:
                    self.bool_mask[self.bool_mask == False][0] = True

                self.norm_layer.weight = nn.Parameter(self.weight_original[self.bool_mask].clone())
                self.norm_layer.bias = nn.Parameter(self.bias_original[self.bool_mask].clone())
                self.norm_layer.running_mean = self.running_mean_original[self.bool_mask].clone().detach()
                self.norm_layer.running_var = self.running_var_original[self.bool_mask].clone().detach()
                # x = x[:, self.bool_mask]
                if self.pad_with_zeros:
                    x = self.norm_layer(x[:, self.bool_mask])
                    if x.shape[0] != self.tensor_to_pass.shape[0]:
                        tmp = torch.zeros([x.shape[0], int(self.output_features.item())] + list(x.shape[2:])).to(x.device)
                        tmp[:, self.bool_mask] = x
                        return tmp
                    self.tensor_to_pass[:, self.bool_mask] = x
                    return self.tensor_to_pass
                else:
                    y = self.norm_layer(x[:, self.bool_mask])
                    x[:, self.bool_mask] = y
                    return x



        class PruningConv(nn.Module):

            def __init__(self, module, remove, name, pad_with_zeros=True):
                super().__init__()
                self.module = module
                self.prune = True
                self.name = name
                shape_0 = module.weight.shape[0]
                self.shape_0 = shape_0
                self.shape_1 = self.module.weight.shape[1]
                self.keep = int((1 - remove) * shape_0)
                # self.conv_pruner = nn.Parameter(torch.ones_like(module.weight.clone()[0, :, 0, 0]) * .5)
                self.conv_pruner = nn.Parameter(torch.mean(torch.abs(module.weight.clone()), dim=(0,2,3)))
                self.original_conv = nn.Parameter(module.weight.clone())
                self.output_features = nn.Parameter(torch.tensor(getattr(self.module, 'out_channels')).float())
                self.indeces = nn.Parameter(torch.ones(shape_0).float())
                delattr(self.module, 'weight')
                setattr(self.module, 'weight', self.original_conv.clone())
                self.reached_threshold = False
                self.wrong_behaviour = False
                self.tensor_slice = nn.Parameter(torch.arange(self.shape_1).float())
                self.tensor_to_pass = None
                self.pad_with_zeros = pad_with_zeros

                class EQ(nn.Module):

                    def forward(self, x):
                        return x

                self.eq = EQ

            def forward(self, x):

                if self.tensor_to_pass is not None:
                    self.tensor_to_pass[:] = 0
                    self.tensor_to_pass = self.tensor_to_pass.detach()
                indeces = None
                if isinstance(self.module, nn.Conv2d):
                    self.tensor_slice = nn.Parameter(torch.tensor(
                        [j for j, xi in enumerate(torch.sum(x, dim=(0, -1, -2))) if not xi == 0]).float())
                    mask_x = torch.tensor(
                        [True if i in self.tensor_slice.long() else False for i in range(x.shape[1])]).bool()

                    # mask_w = torch.tensor(
                    #     [True if i in self.tensor_slice.long() else False for i in range(int(self.output_features.clone().item()))]).bool()

                    # tmp = torch.transpose(torch.transpose(self.original_conv, 0, 1)[tensor_slice], 0, 1)
                    # if isinstance(self.module.weight, nn.Parameter):
                    #     self.module.weight = nn.Parameter(tmp)
                    # else:
                    #     self.module.weight = tmp

                    self.polarization(mask_x)
                    if isinstance(self.module, self.eq):
                        return self.module(x)
                    y = self.module(x[:, mask_x])
                    if self.prune:
                        y = torch.transpose((torch.transpose(y, 1, -1) * self.conv_pruner[self.indeces.bool()]), 1, -1)


                    if self.pad_with_zeros:
                        x = y

                        # if torch.sum(self.indeces) == self.indeces.shape[0] or self.module.weight.shape[0] == \
                        #         self.indeces.shape[0]:
                        #     return x
                        # else:
                        if self.tensor_to_pass is None and x.shape[0] != 1:
                            self.tensor_to_pass = torch.zeros(
                                [x.shape[0], int(self.output_features.item())] + list(x.shape[2:]))
                        if self.tensor_to_pass is not None:
                            self.tensor_to_pass = self.tensor_to_pass.to(x.device)
                        # tensor_to_pass = self.tensor_to_pass
                        # mask_x_out = torch.tensor(
                        #     [True if i in self.indeces.long() else False for i in range(int(self.output_features.item()))]).bool()
                        if self.tensor_to_pass is None or x.shape[0] != self.tensor_to_pass.shape[0]:
                            tmp = torch.zeros([x.shape[0], int(self.output_features.item())] + list(x.shape[2:])).to(
                                x.device)
                            tmp[:, self.indeces.bool()] = x
                            return tmp
                        self.tensor_to_pass[:, self.indeces.bool()] = x
                        return self.tensor_to_pass
                    else:
                        x[:, self.indeces.bool()] = y
                        return x

                if isinstance(self.module, self.eq):
                    return self.module(x)

            def polar_bins(self):
                polar_indeces = polarization_bins.apply(self.conv_pruner.clone())
                return polar_indeces

            def polarization(self, mask_w):
                if self.prune and torch.sum(self.indeces) > self.keep and self.training:
                    original_sliced = self.original_conv.clone()[:, mask_w]
                    self.module.in_channels = original_sliced.shape[1]

                    indeces = make_bins.apply(self.conv_pruner.clone())
                    if torch.sum(indeces) != 0:
                        self.indeces = nn.Parameter(indeces.float())
                        tmp = torch.transpose(torch.transpose(original_sliced, 0, -1
                                                              ) * indeces, 0, -1)[self.indeces.bool()]
                        indeces = 'not to use'
                        if isinstance(self.module.weight, nn.Parameter):
                            self.module.weight.data = tmp

                        else:
                            self.module.weight = tmp

                    elif not self.wrong_behaviour:
                        self.wrong_behaviour = True
                        indeces = 'not to use'
                        self.stop()
                        print(f'{self.name} tried to remove everything')
                        if self.shape_0 == self.shape_1:
                            print('REPLACING WITH EQ MODULE')
                            self.module = self.eq()

                else:
                    if torch.sum(self.indeces) <= self.keep and not self.reached_threshold:
                        self.reached_threshold = True
                        print(f'{self.name} stopped pruning')
                    self.stop()
                    indeces = 'not to use'

            def stop(self):
                if isinstance(self.module, self.eq):
                    self.prune = False
                    return
                tmp = torch.transpose(torch.transpose(self.original_conv, 0, 1)[self.tensor_slice.long()], 0, 1)[self.indeces.bool()]
                # tmp = self.original_conv.clone()[self.indeces.bool()]
                if isinstance(self.module.weight, nn.Parameter):
                    self.module.weight.data = tmp
                else:
                    self.module.weight = tmp
                # self.module.bias = self.module.bias[self.indeces.bool()]
                setattr(self.module, 'out_channels', int(torch.sum(self.indeces.clone()).item()))
                self.prune = False

            def restart(self):
                self.prune = True

            # def return_mask(self):
            #     return torch.mean(self.p)

        self.PruningConv = PruningConv
        self.PrunedBatchNorm2d = PrunedBatchNorm2d
        self.init_strategy()


    def init_strategy(self):

        class PassModule(nn.Module):
            def __init__(self, module, device, task):
                super().__init__()
                self.model_teacher = module
                self.device = device
                self.task = task

            if self.task == 'segmentation':
                def forward(self, batch):
                    # batch = batch.unsqueeze(0)
                    tmp = torch.ones(1, 19, batch.shape[-2], batch.shape[-1])
                    img_metas = [{'ori_shape': batch.shape, 'img_shape': tmp.shape, 'pad_shape': None,
                                  'gt_semantic_seg': tmp, 'flip': False}]

                    losses = self.model_teacher.model_teacher(img=batch, img_metas=img_metas, return_loss=True,
                                                **{'gt_semantic_seg': torch.ones(1, 1, batch.shape[-2] // 8,
                                                                                 batch.shape[-1] // 8
                                                                                 ).long()})
                    loss, log_vars = self.model_teacher.model_teacher._parse_losses(losses)

                    return loss
            elif self.task == 'classification':
                def forward(self, x):
                    return self.model_teacher(x)

            def train_step(self, batch, optimizer, **kwargs):
                # batch['img_metas'] = batch['img_metas'].data[0]
                # batch['img'] = batch['img'].data[0].to(self.device)
                # batch['gt_semantic_seg'] = batch['gt_semantic_seg'].data[0].to(self.device)

                return self.model_teacher.train_step(batch, optimizer, **kwargs)

            def val_step(self, batch, **kwargs):
                return self.model_teacher.val_step(batch, **kwargs)



        def prune_specific_module(model, amount, name, pad_with_zeros):
            if self.all_modules:
                modules_list = [m for mn, m in model.named_modules() for n, p in m.named_parameters()
                                if self.not_to_prune in mn
                                ]
            else:
                modules_list = [m for mn, m in model.named_modules() for n, p in m.named_parameters()
                                if isinstance(m, nn.Conv2d) and 'weight' in n and p.shape[0] != p.shape[1] or
                                self.not_to_prune in mn or '17.conv1' in mn
                                ]
            named_modules = list(model.named_parameters())
            for n, p in named_modules:
                modules = n.split('.')
                if 'weight' in n:
                    for i, mn in enumerate(modules):
                        if i == 0:
                            if replace_Conv_with_PruningConv(model, mn, amount, modules_list, name=name+'.'+n, pad_with_zeros=pad_with_zeros):
                                break
                            if replace_BatchNorm2d_with_PrunedBatchNorm2d(model, mn, modules_list, pad_with_zeros=pad_with_zeros):
                                break
                            m = getattr(model, mn)
                        else:
                            if replace_Conv_with_PruningConv(m, mn, amount, modules_list, name=name+'.'+n, pad_with_zeros=pad_with_zeros):
                                break
                            if replace_BatchNorm2d_with_PrunedBatchNorm2d(m, mn, modules_list, pad_with_zeros=pad_with_zeros):
                                break
                            m = getattr(m, mn)

        def replace_Conv_with_PruningConv(m, mn, amount, modules_list, name, pad_with_zeros):
            sub_module = getattr(m, mn)
            if isinstance(sub_module, nn.Conv2d) and sub_module not in modules_list and not isinstance(m, self.PruningConv):

                setattr(m, mn, self.PruningConv(module=sub_module, remove=amount, name=name, pad_with_zeros=pad_with_zeros))
                return True
            else:
                return False

        def replace_BatchNorm2d_with_PrunedBatchNorm2d(m, mn, modules_list, pad_with_zeros):
            sub_module = getattr(m, mn)
            if isinstance(sub_module, nn.BatchNorm2d) and sub_module not in modules_list and not isinstance(m, self.PrunedBatchNorm2d):

                setattr(m, mn, self.PrunedBatchNorm2d(norm_layer=sub_module, pad_with_zeros=pad_with_zeros))
                return True
            else:
                return False



        def prune_specific_modules():
            for module, amount, pad_with_zeros in zip(self.modules_to_prune, self.amount, self.pad_with_zeros):
                for module_name, nn_module in self.model_teacher.named_modules():
                    if module == module_name.split('.')[-1]:
                        prune_specific_module(model=nn_module, amount=amount, name=module, pad_with_zeros=pad_with_zeros)

        prune_specific_modules()
        example_inputs = torch.randn(1, 3, 1024, 1024)

        if hasattr(self.model_teacher, 'module'):
            self.model_teacher = PassModule(self.model_teacher.module, self.device, self.task)
        else:
            self.model_teacher = PassModule(self.model_teacher, self.device, self.task)

        self.cpu()
        self.optimizer = torch.optim.AdamW([p for n, p in self.named_parameters() if 'conv_pruner' in n], lr=1e-2)
        self.weight_to_param()

        self.flops_count_orig, _ = tp.utils.count_ops_and_params(self.model_teacher, example_inputs)
        self.params_count_orig = self.active_params()

        self.param_to_weight()
        self.cuda()

    def strategy(self, x):
        pass

    def forward(self, x):

        if self.iter % self.print_every == 0 and self.training:
            self.stop_pruning()
            if not self.stop_def:
                self.restart_pruning()

        if self.training:
            self.iter += 1
        x = self.model_teacher(x)
        return x

    def train_step(self, batch, optimizer, **kwargs):

        if self.iter % self.print_every == 0 and self.training:
            self.stop_pruning()
            if not self.stop_def:
                self.restart_pruning()

        if self.training:
            self.iter += 1
        return self.model_teacher.train_step(batch, optimizer, **kwargs)

    def val_step(self, batch, **kwargs):
        return self.model_teacher.val_step(batch, **kwargs)

    def stop_pruning(self):
        self.weight_to_param()
        for mn, m in self.model_teacher.named_modules():
            if isinstance(m, self.PruningConv):
               m.stop()
        self.print_flops_and_params()
        self.param_to_weight()

    def weight_to_param(self):
        for mn, m in self.model_teacher.named_modules():
            if isinstance(m, self.PruningConv) and isinstance(m.module, nn.Conv2d):
                setattr(m.module, 'weight', nn.Parameter(getattr(m.module, 'weight')))

    def param_to_weight(self):
        for mn, m in self.model_teacher.named_modules():
            if isinstance(m, self.PruningConv) and isinstance(m.module, nn.Conv2d):
                w = getattr(m.module, 'weight')
                delattr(m.module, 'weight')
                setattr(m.module, 'weight', w.clone())

    def restart_pruning(self):
        for mn, m in self.model_teacher.named_modules():
            if isinstance(m, self.PruningConv):
                m.restart()

    def print_flops_and_params(self):
        self.cpu()
        example_inputs = torch.randn(1, 3, 1024, 1024)
        self.flops_count, _ = tp.utils.count_ops_and_params(self.model_teacher, example_inputs)
        self.params_count = self.active_params()
        self.active_parameters = 100 * (self.params_count / self.params_count_orig)
        print(f"ACTIVE PARAMETERS ABSOLUTE ORIGINAL: {self.params_count_orig}")
        print(f"ACTIVE PARAMETERS ABSOLUTE PRUNED: {self.params_count}")
        print(f"ACTIVE PARAMETERS: {self.active_parameters}")
        print(f'FLOPS PRUNED: {self.flops_count} | ORIGINAL: {self.flops_count_orig}')
        self.cuda()

    def active_params(self):
        active_params = 0
        for n, p in self.named_parameters():
            list_utils_params = ['conv_pruner', 'original_conv', 'output_features', 'indeces', 'tensor_slice', 'original']
            if torch.tensor([name not in n for name in list_utils_params]).all():
                active_params += torch.numel(p)

        return active_params

    def backward_and_step(self):
        self.optimizer.zero_grad()
        self.loss_polar().backward(retain_graph=True)
        self.optimizer.step()

    def loss_polar(self):
        pb = [p
              for mn, m in self.model_teacher.named_modules()
              if isinstance(m, self.PruningConv)
              for p in m.polar_bins()]
        pb_mean = torch.mean(torch.stack(pb))
        loss_polar = torch.mean(torch.stack([1.2 * p - torch.abs(p - pb_mean) for p in pb]))
        # active_params = []
        # for n, p in self.named_parameters():
        #     if 'original_conv' in n:
        #         active_params.append(torch.sum(torch.abs(p)))
        # loss_l1 = torch.mean(torch.stack(active_params))
        # return loss_polar + loss_l1
        return loss_polar

    def stop_def_met(self):
        self.stop_def = True


from pynput import keyboard

class OutputAnalyzer(nn.Module):

    def __init__(self, model, device='cuda'):
        super().__init__()
        self.device = device
        self.model = model
        self.print_flops_and_params()
        # _features = 128
        # self.analyzer = nn.ModuleList([nn.ModuleList([nn.Conv2d(_features, _features, 3, padding=1, groups=_features),
        #                                               nn.BatchNorm2d(_features)]) for _ in range(_layers)])
        class Pass(nn.Module):
            def __init__(self):
                super().__init__()
                print('Replaced BasicBlock with Pass Module')

            def forward(self, x):
                return x

        self.Pass = Pass

        class PassZero(nn.Module):
            def __init__(self):
                super().__init__()
                print('Replaced Layer with PassZero Module')

            def forward(self, x):
                return 0

        self.PassZero = PassZero
        from resnet import BasicBlock
        self.BasicBlock = BasicBlock
        self.stop_iter = 10 # 4_000
        self.list_non__zeros = []
        self.list_el = []
        self.counter = 0
        self.interact = 0.
        self.activation = nn.ReLU()
        self.done = False
        self.finished = False

        class NewConv2d(nn.Module):

            def __init__(self, m, intro, analyzer, outro, list_non__zeros, list_el, m_name,
                         stop_iter=10_000, previous_mask_bool=False, also_next=False):
                super().__init__()
                self.m = m
                self.previous_mask = None
                self.also_next = also_next
                self.previous_mask_bool = previous_mask_bool
                self.m_name = m_name
                # self.initial_tuning = stop_iter//2
                self.stop_iter = stop_iter # + self.initial_tuning
                if m.weight.shape[0] != m.weight.shape[1]:
                    self.is_first = True
                else:
                    self.is_first = False

                self.m_recover = nn.Parameter(torch.transpose(copy.deepcopy(m.weight), 1, -1))

                self.intro = intro

                if self.intro is not None: del self.m.weight

                self.outro = outro
                self.activation = nn.ReLU()
                self.activation_inner = nn.ReLU()
                self.list_non__zeros = list_non__zeros
                self.list_el = list_el
                self.analyzer = analyzer
                # self.binarizer = Binarizer()
                # self.m.requires_grad = False
                self.counter = 0
                self.start_offset = 1.15 # [1, 1.15, 1.25]
                self.continuous_mask = torch.zeros(1).cuda()
                self.mask = torch.ones(self.m_recover.shape[-1]).cuda()
                self.pos = None
                self.mp = nn.MaxPool2d(2)
                # for p in self.parameters():
                #     p = p * 0 + 1

            def apply_pruning_also_next(self, previous_mask=None, unstrided=True, do_input=False, full_output=False):
                self.m.weight = nn.Parameter(torch.transpose(copy.deepcopy(self.m_recover), 1, -1))

                if full_output:
                    self.previous_mask = torch.ones_like(self.mask)
                    previous_mask = self.previous_mask
                # if previous_mask is None:
                #     previous_mask = self.mask
                # if do_input:
                    # self.m.weight = nn.Parameter(torch.transpose(copy.deepcopy(self.m_recover), 1, -1))
                mask = self.mask
                # else:
                #     mask = previous_mask
                w = self.m.weight
                if previous_mask is not None and not do_input:
                    # if not unstrided:
                    #     previous_mask = torch.concat([previous_mask, previous_mask])
                    # elif do_input and not torch.allclose(mask, torch.ones_like(mask)):
                    #     previous_mask = mask

                    w = w[previous_mask != 0]
                    ch = torch.count_nonzero(previous_mask)
                    self.m.out_channels = ch

                w = torch.transpose(w, 1, 0)
                if unstrided and not do_input:
                    self.mask = mask
                    w = w[self.mask != 0]
                else:
                    w = w[self.mask != 0]
                self.m.in_channels = torch.count_nonzero(self.mask)
                w = torch.transpose(w, 1, 0)
                self.m.weight = nn.Parameter(w)
                return self.mask


            def apply_pruning(self, previous_mask=None):
                self.m.weight = nn.Parameter(torch.transpose(copy.deepcopy(self.m_recover), 1, -1))
                w = self.m.weight
                # if self.previous_mask:
                #     w = w[self.mask != 0]
                #     self.m.out_channels = torch.count_nonzero(self.mask)
                self.previous_mask = previous_mask
                if previous_mask is not None:
                    w = w[previous_mask != 0]
                    self.m.out_channels = torch.count_nonzero(previous_mask)
                w = torch.transpose(w, 1, 0)
                w = w[self.mask != 0]
                w = torch.transpose(w, 1, 0)
                self.m.weight = nn.Parameter(w)

                self.m.in_channels = torch.count_nonzero(self.mask)
                return self.mask



            def forward(self, inp):
                first_statement = self.counter < self.stop_iter # and (self.counter//(self.stop_iter//10)) % 2 == 0
                second_statement = self.training and self.counter < self.stop_iter and (self.counter // (self.stop_iter // 10)) % 2 == 1
                third_statement = False # self.counter // (self.stop_iter // 4) == 0
                # if self.training and self.counter < 10_000 and (self.counter//1_000) % 2 == 0:
                if self.intro is None:
                    x = self.m(inp)
                    return x

                elif self.training and (first_statement or third_statement):
                    # y = torch.mean(inp, dim=(2, 3))
                    # y = self.activation(inp)
                    y = torchvision.transforms.functional.resize(inp.clone(), (16, 16))
                    y = self.intro(y)
                    x = y
                    for idx, (l, b) in enumerate(self.analyzer):
                        # if idx % 2 == 0:
                        #     y_ = y

                        y = l(y)
                        y = b(y)
                        y = self.activation_inner(y)
                        # if (idx + 1) % 2 == 0:
                        #     y = y + x
                        #     if (idx + 1) % 4 == 0:
                        #         y = self.mp(y)
                        #     x = y
                        y = self.mp(y)

                    y = torch.transpose(y, 1, -1)
                    y = torch.reshape(y, (y.shape[0], y.shape[3]))
                    y = self.outro(y)
                    r_mean = torch.mean(torch.abs(y))
                    r_var = torch.var(y)
                    self.continuous_mask = self.activation(self.continuous_mask.clone().detach() * .5 + .5 * (torch.mean(y, dim=(0,)) - r_mean)/torch.pow(r_var, .5) + (self.start_offset - ((self.counter + interact)/(self.stop_iter))))
                    if self.counter % 50 == 0:
                        self.mask = self.activation(self.continuous_mask)
                    else:
                        self.mask = self.activation(self.continuous_mask)

                # elif self.training and self.counter < 10_000 and (self.counter // 1_000) % 2 == 1:
                elif second_statement:
                    self.mask = self.mask.detach()

                elif self.counter == self.stop_iter:
                    print('END_PRUNING_PHASE')
                    self.mask = self.mask.detach()
                    self.mask_backup = nn.Parameter(self.mask.detach())

                if self.training:
                    self.counter += 1

                if self.pos is None:
                    self.pos = len(self.list_non__zeros)
                    self.list_non__zeros.append(torch.count_nonzero(self.mask.clone().detach()))
                    self.list_el.append(torch.numel(self.mask.clone().detach()))
                else:
                    self.list_non__zeros[self.pos] = torch.count_nonzero(self.mask.clone().detach())
                    self.list_el[self.pos] = torch.numel(self.mask.clone().detach())

                if third_statement:
                    w = self.mask * self.m_recover.clone().detach()
                else:
                    w = self.mask * self.m_recover
                if hasattr(self.m, 'weight'): delattr(self.m, 'weight')


                self.m.weight = torch.transpose(w, 1, -1)
                if self.previous_mask is not None:
                    w = self.m.weight
                    w = torch.transpose(w, 0, -1)
                    w = w * self.previous_mask
                    w = torch.transpose(w, 0, -1)
                    self.m.weight = w


                x = self.m(inp)
                return x

        self.NewConv2d_class = NewConv2d

        iter_over = [(n, len(p.shape)) for n, p in self.model.named_parameters() if len(p.shape) == 4] # if np.array([i not in p.shape[:2] for i in [1,3,19]]).all()]
        # skip_next = True
        self.comunicating_module = {}
        also_next = False
        for i, (n, p_shape) in enumerate(iter_over):
            if p_shape == 4:
                modules = n.split('.')
                for j, module in enumerate(modules):
                    if j == 0:
                        nn_module = getattr(self.model, module)
                        # if isinstance(nn_module, nn.Conv2d) and 3 in m.weight.shape[:2]:
                        #     m = getattr(nn_module, module)
                        #     nn_module.add_module(module, NewConv2d(
                        #         m, None, None, None, self.list_non__zeros,
                        #         self.list_el, stop_iter=self.stop_iter, m_name=n,
                        #     ))

                    elif not isinstance(getattr(nn_module, module), NewConv2d) and isinstance(getattr(nn_module, module), nn.Conv2d)\
                            and (module[-1] == '1' or also_next or len(iter_over) - 2 <= i):

                        m = getattr(nn_module, module)
                        if 3 == m.weight.shape[1]:
                            # skip_next = True
                            none_module = NewConv2d(
                                m, None, None, None, self.list_non__zeros,
                                self.list_el, stop_iter=self.stop_iter, m_name=n, also_next=also_next,
                            )
                            nn_module.add_module(module, none_module)

                        else:
                            # skip_next = False
                            intro, analyzer, outro = self.get_pruning_models(m)
                            key_mask = m.weight.shape[1] # f'{m.weight.shape[0]}_{m.weight.shape[1]}'
                            if key_mask not in self.comunicating_module:
                                self.comunicating_module[key_mask] = []
                            self.comunicating_module[key_mask].append(NewConv2d(
                                m, intro, analyzer, outro, self.list_non__zeros,
                                self.list_el, stop_iter=self.stop_iter, m_name=n, also_next=also_next or len(iter_over) - 1 == i,
                            ))
                            nn_module.add_module(module, self.comunicating_module[key_mask][-1])

                        if also_next:
                            also_next = False

                        if m.stride == (2,2):
                            also_next = True
                        break

                    elif not isinstance(getattr(nn_module, module), NewConv2d) and isinstance(getattr(nn_module, module), nn.Conv2d):

                        m = getattr(nn_module, module)
                        intro, analyzer, outro = self.get_pruning_models(m)


                        if 3 not in m.weight.shape[:2]: # and not skip_next:
                            nn_module.add_module(module, NewConv2d(
                                m, intro, analyzer, outro, self.list_non__zeros,
                                self.list_el, stop_iter=self.stop_iter, m_name=n,
                                previous_mask_bool=True,
                            ))
                        else:
                            nn_module.add_module(module, NewConv2d(
                                m, None, None, None, self.list_non__zeros,
                                self.list_el, stop_iter=self.stop_iter, m_name=n,
                            ))
                        break

                    elif j < len(modules) - 1:
                        nn_module = getattr(nn_module, module)

        # self.remove_shortcut()
    def get_pruning_models(self, m):
        _features = m.weight.shape[1]
        channel_multiplier = 2
        intro = nn.Conv2d(_features, _features * channel_multiplier, 1, padding=0, groups=_features, bias=True)

        _layers = 4

        analyzer = nn.ModuleList([nn.ModuleList([nn.Conv2d(_features * channel_multiplier,
                                                           _features * channel_multiplier, 1, stride=1, groups=_features,
                                                           padding=0, bias=True),
                                                 nn.BatchNorm2d(_features * channel_multiplier)]) for _
                                  in range(_layers)])

        outro = nn.Linear(_features * channel_multiplier, _features)
        return intro, analyzer, outro

    # def remove_shortcut(self):
    #     iter_over = []
    #     return
    #     previous_mask = None
    #     for pm in list(self.model.modules())[::-1]:
    #         if isinstance(pm, self.NewConv2d_class):
    #             tmp_name = pm.m_name
    #             # tmp_out = pm.m.out_channels
    #             # if previous_mask is not None:
    #             #     iter_over.append((tmp_name, previous_mask))
    #             # else:
    #             iter_over.append(tmp_name)
    #
    #     for i, n in enumerate(iter_over):
    #
    #         modules = n.split('.')
    #         nn_module = self.model
    #         for j, module in enumerate(modules):
    #
    #
    #             if isinstance(getattr(nn_module, module), self.NewConv2d_class):
    #
    #                 pm = getattr(nn_module, module)
    #
    #                 is_first = pm.is_first
    #                 if hasattr(nn_module, 'shortcut') and (is_first or i == 0):
    #                     setattr(nn_module, 'shortcut', self.PassZero())
    #                 break
    #
    #             elif j < len(modules) - 1:
    #                 nn_module = getattr(nn_module, module)

    def prune(self):
        iter_over = []
        previous_mask = None
        modules_list = []
        counter = 0
        for i, pm in enumerate(list(self.model.modules())[::-1]):
            if isinstance(pm, self.NewConv2d_class):
                tmp_name = pm.m_name
                tmp_out = pm.m.out_channels
                if previous_mask is not None:
                    iter_over.append((tmp_name, previous_mask))
                else:
                    iter_over.append((tmp_name, torch.ones(tmp_out)))
                # if not pm.also_next and not pm.m.stride == (2,2):
                previous_mask = pm.apply_pruning(previous_mask)
                # elif pm.m.stride == (2,2):
                #     previous_mask = pm.mask
                #     pass
                modules_list.append((counter, pm))
                counter += 1

        do_again = False
        previous_mask = None
        for j, ((n, mask), (i, pm)) in enumerate(zip(iter_over, modules_list)):

            if len(modules_list)-2 > j and modules_list[j+1][1].also_next:
                if previous_mask is None:
                    previous_mask = modules_list[j+1][1].mask
                do_again = True
            else:
                False
            # if j == 0:
            #     previous_mask = pm.apply_pruning_also_next(do_input=True, only=True)

            if pm.also_next or do_again and pm.m.stride != (2,2):
                previous_mask = pm.apply_pruning_also_next(previous_mask, do_input=do_again and not pm.also_next)
                # if pm.also_next:
                #     do_again = True
                # else:
                #     do_again = False
                iter_over[i] = (n, previous_mask)

            elif pm.m.stride == (2,2):
                # previous_mask = pm.mask
                iter_over[i] = (n, previous_mask)
                previous_mask = pm.apply_pruning_also_next(previous_mask, unstrided=False)
            else:
                previous_mask = None


        self.comunicating_module_mask_previous()
        for j, ((n, mask), (i, pm)) in enumerate(zip(iter_over[::-1], modules_list[::-1])):
            pm.apply_pruning(pm.previous_mask)
            # iter_over[i] = (n, pm.mask)

        for j, ((n, mask), (i, pm)) in enumerate(zip(iter_over, modules_list)):

            if j == 0:
                pm.apply_pruning_also_next(full_output=True)
                pm.previous_mask = torch.ones_like(pm.previous_mask)

        for i, (n, mask) in enumerate(iter_over):
            # if mask is None:
            #     continue
            modules = n.split('.')
            nn_module = self.model
            for j, module in enumerate(modules):

                if mask is not None and torch.count_nonzero(mask) == 0 and isinstance(getattr(nn_module, module), self.BasicBlock):

                    setattr(nn_module, module, self.Pass())
                    break

                elif isinstance(getattr(nn_module, module), self.NewConv2d_class):

                    pm = getattr(nn_module, module)
                    # mask = pm.mask

                    pruned_module = pm.m

                    setattr(nn_module, module, pruned_module)
                    is_first = pm.is_first
                    if (hasattr(nn_module, 'shortcut') and ((is_first and module[-1] == '1') or i == 1) or i == 0) and  i != 1:
                        # planes = int(torch.count_nonzero(mask != 0))

                        class Simple(nn.Module):
                            
                            def __init__(self, inp, out, half=True, mask=None, original_shape=None, previous_mask=None, last=False):
                                super().__init__()
                                self.last = last
                                self.half = half
                                self.out = out
                                self.inp = inp
                                self.mask = nn.Parameter(mask.cuda())
                                self.original_shape = original_shape

                                self.previous_mask = nn.Parameter(previous_mask)

                                if self.previous_mask is not None:
                                    pm_shape = copy.deepcopy(self.mask.shape[0])
                                    self.pm_shape = pm_shape
                                # place = (self.out - self.inp)
                                # self.lambda_layer = LambdaLayer(lambda x:
                                #             nn.functional.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, place//2, place//2 + place%2), "constant", 0))

                                if not self.last:
                                    self.lambda_layer = LambdaLayer(lambda x:
                                                nn.functional.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.original_shape//4, self.original_shape//4), "constant", 0))

                            def forward(self, x):
                                # if self.out != self.inp:
                                    if self.last:
                                       k = torch.zeros((self.original_shape, x.shape[0], x.shape[2], x.shape[3])).cuda()
                                    else:
                                        k = torch.zeros(
                                            (self.original_shape // 2, x.shape[0], x.shape[2], x.shape[3])).cuda()
                                    x = torch.transpose(x, 0, 1)
                                    k[self.mask != 0] = x
                                    x = torch.transpose(k, 0, 1)
                                    if not self.last:
                                        y = self.lambda_layer(x)
                                    else:
                                        y = x
                                    y = torch.transpose(torch.transpose(y, 0, 1)[self.previous_mask!=0],0,1)
                                    # remove = (y.shape[0]//2)-(self.pm_shape//2)
                                    # if remove != 0:
                                    #     y = y[remove:-remove + self.pm_shape%2]
                                    # y[self.mask != 0] = x
                                    # y = torch.transpose(y, 0, 1)
                                    return y
                                    # y = torch.zeros((self.original_shape, x.shape[0], x.shape[2], x.shape[3])).cuda()
                                    # x = torch.transpose(x, 0, 1)
                                    # y[self.previous_mask != 0] = x
                                    # y = torch.transpose(y, 0, 1)
                                    # y = self.lambda_layer(y)
                                    # place = (self.out - self.inp)
                                    # y[:,place//2 : -(place//2 + place%2)] = x
                                    # return y[:,:,::2,::2] if self.half else y
                                    # return self.lambda_layer(x)
                                # return x
                        if i != 0:
                            prev_mask = modules_list[i - 1][1].previous_mask
                            original_shape = modules_list[i - 1][1].m_recover.shape[0]
                            setattr(nn_module, 'shortcut', Simple(pruned_module.weight.shape[1], pruned_module.weight.shape[0], half=i != 0,
                                                              mask=pm.mask,original_shape=original_shape, previous_mask=prev_mask))
                        else:
                            pm = modules_list[i + 1][1]
                            prev_mask =  modules_list[i][1].previous_mask
                            original_shape = modules_list[i][1].m_recover.shape[0]
                            setattr(nn_module, 'shortcut', Simple(pm.m.weight.shape[1], pm.m.weight.shape[0], half=i != 0,
                                                              mask=pm.mask,original_shape=original_shape, previous_mask=prev_mask, last=True))
                            pm = modules_list[i][1]

                    mask = pm.previous_mask
                    if module[-1] == '1':
                        # if len(bn.bias) != len(mask):
                        #     mask = torch.concat(([mask, mask]))
                        bn = getattr(nn_module, 'bn1')
                        bn.bias = nn.Parameter(bn.bias[mask != 0])
                        bn.weight = nn.Parameter(bn.weight[mask != 0])
                        bn.running_mean = bn.running_mean[mask != 0]
                        bn.running_var = bn.running_var[mask != 0]
                        bn.num_features = int(torch.count_nonzero(mask != 0))
                        setattr(nn_module, 'bn1', bn)

                    elif module[-1] == '2':
                        bn = getattr(nn_module, 'bn2')
                        bn.bias = nn.Parameter(bn.bias[mask != 0])
                        bn.weight = nn.Parameter(bn.weight[mask != 0])
                        bn.running_mean = bn.running_mean[mask != 0]
                        bn.running_var = bn.running_var[mask != 0]
                        bn.num_features = int(torch.count_nonzero(mask != 0))
                        setattr(nn_module, 'bn2', bn)

                    break

                elif j < len(modules) - 1:
                    nn_module = getattr(nn_module, module)


    def forward(self, x, wandb=None):
        if self.stop_iter == self.counter:
            if not self.done:
                self.comunicating_module_mask()
                self.comunicating_module_mask_previous()
                self.done = True
            self.prune()
            self.finished = True
            self.optimizer_pruning_mode_11 = torch.optim.SGD(self.parameters(), lr=1e-2)

            mode = self.training
            self.eval()
            self.print_flops_and_params()

            if mode:
                self.train()

            if wandb is not None:
                wandb.log({
                    'flops_count': self.flops_count,
                    'params_count': self.params_count,
                })

        if (not self.training and not self.done) and self.stop_iter > self.counter:
            self.comunicating_module_mask()
            self.done = True
        if self.training:
            self.done = False
        global interact
        # self.listener.join(.1)
        with keyboard.Events() as events:
            # Block at most one second
            event = events.get(0.01)

            if event is None:
                pass
            elif event.key == keyboard.Key.up:
                self.interact += 50
                print("Prune more")
            elif event.key == keyboard.Key.down:
                self.interact -= 50
                print("Prune less")
        interact = self.interact
        x = self.model(x)
        if self.counter % 50 == 0:
            tmp_non_zeros = torch.tensor(self.list_non__zeros).sum().item()
            tmp_el = torch.tensor(self.list_el).sum().item()
            off_parameters = ((tmp_el-tmp_non_zeros)/tmp_el) * 100
            print(f'off_parameters: {off_parameters} %')
            if wandb is not None:
                wandb.log({
                    'off_parameters': off_parameters,
                })
            if off_parameters>50:

                self.interact -= 50


        self.counter += 1
        return x

    def comunicating_module_mask(self):

        self.used_masks = {k: torch.stack([mod.continuous_mask for mod in modl]) for k, modl in self.comunicating_module.items()}
        for k, modl in self.comunicating_module.items():
            for mod in modl:
                mask_mod = self.used_masks[k]
                _mean = torch.mean(mask_mod)
                mod.mask = self.activation(torch.mean(mask_mod, dim=0) - _mean * .7)
                # print(torch.sum(mod.mask == 0))

    def comunicating_module_mask_previous(self):
        tmp_dict = {}
        for modl in self.comunicating_module.values():
            for mod in modl:
                if mod.previous_mask is not None:
                    if mod.previous_mask.shape[0] not in tmp_dict:
                        tmp_dict[mod.previous_mask.shape[0]] = []
                    tmp_dict[mod.previous_mask.shape[0]].append(mod.previous_mask.cuda())
        # mask = {k: torch.mean(torch.stack([mod for mod in modl if mod is not None]), dim=0) for k, modl in tmp_dict.items()}
        mask = {k: self.used_masks[k] for k, modl in tmp_dict.items()}

        for k, mask_mod in tmp_dict.items():
                mask_mod = mask[k]
                _mean = torch.mean(mask_mod)
                mod.previous_mask = self.activation(torch.mean(mask_mod, dim=0) - _mean * .7)
                # mod.previous_mask = self.activation(torch.mean(mask_mod, dim=0) - torch.mean(mask_mod) + .5)
    def print_flops_and_params(self):
        example_inputs = torch.randn(1, 3, 1024, 1024)
        # self.model.cpu()
        self.model.eval()
        self.flops_count, self.params_count = tp.utils.count_ops_and_params(self.model, example_inputs)

        # self.model.cuda()
        # print(f"ACTIVE PARAMETERS ABSOLUTE ORIGINAL: {self.params_count_orig}")
        print(f"ACTIVE PARAMETERS ABSOLUTE PRUNED: {self.params_count}")
        # print(f"ACTIVE PARAMETERS: {100 * (params_count / self.params_count_orig)}")
        print(f'FLOPS PRUNED: {self.flops_count}')
        return self.flops_count, self.params_count
class Encoder2Mask(nn.Module):

    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.to(device)
        self.dict_of_weights = {n: p.detach() for n, p in self.model.named_parameters() if len(p.shape) == 4}
        self.tot_shape = 0
        max_dim = 0
        different_shapes = []

        self.tot_params = 0
        for n, p in self.model.named_parameters():
            if len(p.shape) == 4:
                self.tot_params += torch.numel(p)
                self.tot_shape += p.shape[1]

        for p in self.dict_of_weights.values():
            first_dim = p.shape[0]
            max_dim = max(max_dim, first_dim)
            if first_dim not in different_shapes:
                different_shapes.append((first_dim, p.shape[-2], p.shape[-1]))
        self.max_dim = max_dim//16
        print(max_dim)
        self.first_module = nn.ModuleDict({f'{dim}_{k1}_{k2}': nn.Linear(dim*k1*k2+1, self.max_dim) for dim, k1, k2 in different_shapes})
        # self.multihead_attn = nn.MultiheadAttention(self.max_dim, 1)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.max_dim, nhead=1)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.layers = nn.ModuleList([nn.MultiheadAttention(self.max_dim, 1) for _ in range(2)])
        # self.layers_bn = nn.ModuleList([nn.BatchNorm1d(self.max_dim) for _ in range(4)])
        self.output_layer = nn.Linear(self.max_dim, 1)
        self.activation = nn.ReLU()

        self.binarizer = Binarizer()
        self.active_params = 0
        self.to(device)
        self.device = device
        self.dict_of_pos_emb = {}
        self.dict_of_masks = {}
        len_dict = len(self.dict_of_weights)
        for i, (n, p) in enumerate(self.dict_of_weights.items()):
            x = torch.transpose(p, 0, 1).flatten(1)
            self.dict_of_pos_emb[n] = (torch.ones_like(x) * (((i / len_dict) - .5) ** 2)*2)[:, 0].unsqueeze(-1)
        self.stopped = False

    def stop_pruning(self):
        self.stopped = True

        self(get_masks=True)
        self.models_params = nn.ParameterDict({n.replace('.', '_'): nn.Parameter(p) for n,p in self.dict_of_weights.items()})
        self.optimizer = torch.optim.AdamW(self.models_params.parameters(), lr=5e-4)

    def reinsert_parameters(self):
        for n, m in self.dict_of_masks.items():
            for n2, p in self.models_params.items():
                if n.replace('.', '_') == n2:
                    modules = n.split('.')
                    for i, module in enumerate(modules):
                        if i == 0:
                            nn_module = getattr(self.model, module)
                        elif i < len(modules) - 1:
                            nn_module = getattr(nn_module, module)
                        else:
                            # current_param = getattr(nn_module, module)
                            # delattr(nn_module, module)
                            setattr(nn_module, module, torch.transpose(torch.transpose(p, 1, -1) * m.detach(), 1, -1))
                    break


    def forward(self, get_masks=False):
        if not self.stopped or get_masks:
            self.loss_to_add = torch.zeros(1).to(self.device)
            self.active_params = 0
            for i, (n, p) in enumerate(self.dict_of_weights.items()):

                dim, _, k1, k2 = tuple(p.shape)
                name = f'{dim}_{k1}_{k2}'
                x = torch.transpose(p, 0, 1).flatten(1)
                x = self.first_module[name](torch.cat([torch.abs(x), self.dict_of_pos_emb[n]], dim=-1))
                x = x.unsqueeze(0)
                for l in self.layers:
                    x, _ = l(x, x, x)
                    x = self.activation(x)
                x = x[0]
                x = self.output_layer(x).squeeze()
                y = self.binarizer.apply(x.clone())
                if get_masks:
                    self.dict_of_masks[n] = y.clone().detach()
                else:
                    p = torch.transpose(torch.transpose(p, 1, -1) * y, 1, -1)
                    self.loss_to_add = self.loss_to_add + torch.sum(torch.abs(x))
                    modules = n.split('.')
                    for j, module in enumerate(modules):
                        if j == 0:
                            nn_module = getattr(self.model, module)
                        elif j < len(modules) - 1:
                            nn_module = getattr(nn_module, module)
                        else:
                            if isinstance(getattr(nn_module, module), nn.Parameter):
                                delattr(nn_module, module)
                            setattr(nn_module, module, p)
                self.active_params += torch.count_nonzero(p.clone().detach()).item()
        else:
            self.reinsert_parameters()