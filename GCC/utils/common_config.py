"""
Author: Huasong Zhong
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import math
import numpy as np
import torch
import torchvision.transforms as transforms
from data.augment import Augment, Cutout
from utils.collate import collate_custom
from data import ConcatDataset


def get_criterion(p):
    if p['criterion'] == 'end2end':
        from losses.losses import RGCLoss
        criterion1 = RGCLoss(p['criterion_kwargs']['temperature'])
        from losses.losses import AGCLoss
        criterion2 = AGCLoss(p['criterion_kwargs']['entropy_weight'])
        return (criterion1, criterion2)

    elif p['criterion'] == 'confidence-cross-entropy':
        from losses.losses import ConfidenceBasedCE
        criterion = ConfidenceBasedCE(p['confidence_threshold'], p['criterion_kwargs']['apply_class_balancing'])

    else:
        raise ValueError('Invalid criterion {}'.format(p['criterion']))

    return criterion


def get_feature_dimensions_backbone(p):
    if p['backbone'] == 'resnet18':
        return 128

    elif p['backbone'] == 'resnet34':
        return 128

    elif p['backbone'] == 'resnet50':
        return 128

    else:
        raise NotImplementedError
"""
用于根据指定的架构和预训练的权重路径构建一个PyTorch模型。
该函数首先根据p['backbone']的值来选择合适的骨干网络。
如果train_db_name是cifar-10、cifar-20或stl-10，该函数会导入resnet_cifar或resnet_stl模块，并以resnet18架构初始化骨干网
如果train_db_name包含字符串 "imagenet"，则用resnet_stl模块的resnet18架构初始化骨干网
如果p['backbone']是resnet34，并且train_db_name包含字符串 "imagenet"，那么骨干网将用resnet_stl模块中的resnet34架构初始化
如果p['backbone']是resnet50，并且train_db_name包含字符串 "imagenet"，那么骨干网将用resnet模块的resnet50架构初始化
如果这些条件都不满足，就会出现NotImplementedError。

在选择骨干网后，该函数根据p['setup']的值初始化模型
如果 p['setup'] 是 simclr 或 moco，该函数从 models 模块导入 ContrastiveModel 类，并用骨干网和指定的 model_kwargs 来初始化模型
如果 p['setup'] 是 scan 或 selflabel，该函数从 models 模块导入 ClusteringModel 类，并用骨架、类的数量（p['num_classes']）和头的数量（p['num_heads']）初始化模型
如果 p['setup'] 是 selflabel，该函数断言 p['num_heads'] 是 1。如果 p['setup'] 是 end2end，该函数从 models 模块导入 End2EndModel 类，并用主干和指定的 model_kwargs 来初始化该模型
如果这些条件都不满足，就会产生一个ValueError

最后，如果 pretrain_path 不是 None 并且指定的路径存在，该函数将加载预训练的权重并将其转移到模型中
如果p['setup']是selflabel，该函数只继续使用最好的头，并删除所有其他的头
如果pretrain_path不是None，但指定的路径不存在，则会产生ValueError
如果pretrain_path是None，函数会返回初始化的模型。
"""

def get_model(p, pretrain_path=None):
    # 获取主干网络
    if p['backbone'] == 'resnet18':
        if p['train_db_name'] in ['cifar-10', 'cifar-20']:
            from models.resnet_cifar import resnet18
            backbone = resnet18()

        elif p['train_db_name'] == 'stl-10':
            from models.resnet_stl import resnet18
            backbone = resnet18()

        elif 'imagenet' in p['train_db_name']:
            from models.resnet_stl import resnet18
            backbone = resnet18()

        else:
            raise NotImplementedError

    elif p['backbone'] == 'resnet34':
        if 'imagenet' in p['train_db_name']:
            from models.resnet_stl import resnet34
            backbone = resnet34(feature_size=5)

    elif p['backbone'] == 'resnet50':
        if 'imagenet' in p['train_db_name']:
            from models.resnet import resnet50
            backbone = resnet50()

        else:
            raise NotImplementedError

    else:
        raise ValueError('Invalid backbone {}'.format(p['backbone']))

    # 根据Setup初始化模型
    if p['setup'] in ['simclr', 'moco']:
        from models.models import ContrastiveModel
        model = ContrastiveModel(backbone, **p['model_kwargs'])

    elif p['setup'] in ['scan', 'selflabel']:
        from models.models import ClusteringModel
        if p['setup'] == 'selflabel':
            assert(p['num_heads'] == 1)
        model = ClusteringModel(backbone, p['num_classes'], p['num_heads'])
    elif p['setup'] == 'end2end':
        from models.models import End2EndModel
        model = End2EndModel(backbone, **p['model_kwargs'])
    else:
        raise ValueError('Invalid setup {}'.format(p['setup']))

    # Load pretrained weights
    if pretrain_path is not None and os.path.exists(pretrain_path):
        state = torch.load(pretrain_path, map_location='cpu')

        if p['setup'] == 'selflabel': # Weights are supposed to be transfered from scan
            # We only continue with the best head (pop all heads first, then copy back the best head)
            model_state = state['model']

            all_heads = [k for k in model_state.keys() if 'contrastive_head' in k]
            for k in all_heads:
                print ("remove: {}".format(k))
                model_state.pop(k)

            missing = model.load_state_dict(model_state, strict=True)

        else:
            raise NotImplementedError

    elif pretrain_path is not None and not os.path.exists(pretrain_path):
        raise ValueError('Path with pre-trained weights does not exist {}'.format(pretrain_path))

    else:
        pass

    return model

"""
这段代码定义了一个名为get_train_dataset的函数，它接受一些参数并返回一个数据集，根据参数p['train_db_name']的值
函数会选择不同的数据集，包括CIFAR-10、CIFAR-20、STL-10、ImageNet10、ImageNet Dogs、TinyImageNet、ImageNet以及ImageNet的子集。
如果to_augmented_dataset参数为True，则返回一个AugmentedDataset，该数据集返回一个图像及其增强版本
如果to_neighbors_dataset参数为True，则返回一个NeighborsDataset，该数据集返回一个图像及其最近邻之一
如果to_end2end_dataset参数为True，则返回一个End2EndDataset，该数据集返回一个图像及其最近邻之一，用于端到端训练 
"""


def get_train_dataset(p, transform, to_augmented_dataset=False,
                        to_neighbors_dataset=False, to_end2end_dataset=False, split=None):
    # Base dataset
    if p['train_db_name'] == 'cifar-10':
        from data.cifar import CIFAR10
        dataset = CIFAR10(train=True, transform=transform, download=True)
        val_dataset = CIFAR10(train=False, transform=transform, download=True)

    elif p['train_db_name'] == 'cifar-20':
        from data.cifar import CIFAR20
        dataset = CIFAR20(train=True, transform=transform, download=True)

    elif p['train_db_name'] == 'stl-10':
        from data.stl import STL10
        dataset = STL10(split=split, transform=transform, download=True)

    elif p['train_db_name'] == 'imagenet10':
        from data.datasets_imagenet10 import ImageNet10
        dataset = ImageNet10(split='train', transform=transform, download=True)

    elif p['train_db_name'] == 'imagenet_dogs':
        from data.datasets_imagenet_dogs import ImageNetDogs
        dataset = ImageNetDogs(split='train', transform=transform, download=True)

    elif p['train_db_name'] == 'tiny_imagenet':
        from data.datasets_tiny_imagenet import TinyImageNet
        dataset = TinyImageNet(split='train', transform=transform, download=True)

    elif p['train_db_name'] == 'imagenet':
        from data.imagenet import ImageNet
        dataset = ImageNet(split='train', transform=transform)

    elif p['train_db_name'] in ['imagenet_50', 'imagenet_100', 'imagenet_200']:
        from data.imagenet import ImageNetSubset
        subset_file = './data/imagenet_subsets/%s.txt' %(p['train_db_name'])
        dataset = ImageNetSubset(subset_file=subset_file, split='train', transform=transform)

    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))

    # Wrap into other dataset (__getitem__ changes)
    if to_augmented_dataset: # Dataset returns an image and an augmentation of that image.
        from data.custom_dataset import AugmentedDataset
        dataset = AugmentedDataset(dataset)

    if to_neighbors_dataset: # Dataset returns an image and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset
        indices = np.load(p['topk_neighbors_train_path'])
        dataset = NeighborsDataset(dataset, indices, p['num_neighbors'])

    if to_end2end_dataset:
        from data.custom_dataset import End2EndDataset
        if not os.path.exists(p['topk_neighbors_val_path']):
            indices = None
        else:
            indices = np.load(p['topk_neighbors_val_path'])
        # dataset = ConcatDataset([dataset, val_dataset])
        dataset = End2EndDataset(dataset, indices, p['num_neighbors'])

    return dataset


def get_val_dataset(p, transform=None, to_neighbors_dataset=False, to_end2end_dataset=False):
    # Base dataset
    if p['val_db_name'] == 'cifar-10':
        from data.cifar import CIFAR10
        dataset = CIFAR10(train=False, transform=transform, download=True)
        train_dataset = CIFAR10(train=True, transform=transform, download=True)

    elif p['val_db_name'] == 'cifar-20':
        from data.cifar import CIFAR20
        dataset = CIFAR20(train=False, transform=transform, download=True)

    elif p['val_db_name'] == 'stl-10':
        from data.stl import STL10
        dataset = STL10(split='test', transform=transform, download=True)

    elif p['train_db_name'] == 'imagenet10':
        from data.datasets_imagenet10 import ImageNet10
        dataset = ImageNet10(split='train', transform=transform, download=True)

    elif p['train_db_name'] == 'imagenet_dogs':
        from data.datasets_imagenet_dogs import ImageNetDogs
        dataset = ImageNetDogs(split='train', transform=transform, download=True)

    elif p['train_db_name'] == 'tiny_imagenet':
        from data.datasets_tiny_imagenet import TinyImageNet
        dataset = TinyImageNet(split='train', transform=transform, download=True)

    elif p['val_db_name'] == 'imagenet':
        from data.imagenet import ImageNet
        dataset = ImageNet(split='val', transform=transform)

    elif p['val_db_name'] in ['imagenet_50', 'imagenet_100', 'imagenet_200']:
        from data.imagenet import ImageNetSubset
        subset_file = './data/imagenet_subsets/%s.txt' %(p['val_db_name'])
        dataset = ImageNetSubset(subset_file=subset_file, split='val', transform=transform)

    else:
        raise ValueError('Invalid validation dataset {}'.format(p['val_db_name']))

    # Wrap into other dataset (__getitem__ changes)
    if to_neighbors_dataset: # Dataset returns an image and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset
        indices = np.load(p['topk_neighbors_val_path'])
        dataset = NeighborsDataset(dataset, indices, 5) # Only use 5

    if to_end2end_dataset:
        from data.custom_dataset import End2EndDataset
        if not os.path.exists(p['topk_neighbors_val_path']):
            indices = None
        else:
            indices = np.load(p['topk_neighbors_val_path'])
        #dataset = ConcatDataset([dataset, train_dataset])
        dataset = End2EndDataset(dataset, indices, 5)

    return dataset


def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=True)


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=False)

"""
 该函数根据p['augmentation_strategy']的值返回一个数据增强的组合
 如果p['augmentation_strategy']的值为standard，则使用标准的数据增强策略，包括随机裁剪、随机水平翻转、转换为张量和归一化
 如果p['augmentation_strategy']的值为simclr，则使用SimCLR论文中的数据增强策略，包括随机裁剪、随机水平翻转、随机颜色抖动、随机灰度化、转换为张量和归一化
 如果p['augmentation_strategy']的值为ours，则使用我们论文中的数据增强策略，包括随机水平翻转、随机裁剪、强数据增强、转换为张量、归一化和Cutout
 如果p['augmentation_strategy']的值不是这三个值之一，则引发ValueError异常，提示Invalid augmentation strategy
"""
def get_train_transformations(p):
    if p['augmentation_strategy'] == 'standard':
        # Standard augmentation strategy
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    elif p['augmentation_strategy'] == 'simclr':
        # Augmentation strategy from the SimCLR paper
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(**p['augmentation_kwargs']['color_jitter'])
            ], p=p['augmentation_kwargs']['color_jitter_random_apply']['p']),
            transforms.RandomGrayscale(**p['augmentation_kwargs']['random_grayscale']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])

    elif p['augmentation_strategy'] == 'ours':
        # Augmentation strategy from our paper
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(p['augmentation_kwargs']['crop_size']),
            Augment(p['augmentation_kwargs']['num_strong_augs']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize']),
            Cutout(
                n_holes = p['augmentation_kwargs']['cutout_kwargs']['n_holes'],
                length = p['augmentation_kwargs']['cutout_kwargs']['length'],
                random = p['augmentation_kwargs']['cutout_kwargs']['random'])])

    else:
        raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))


def get_val_transformations(p):
    return transforms.Compose([
            transforms.Resize(p['transformation_kwargs']['resize']), # set for large image
            transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
            transforms.ToTensor(),
            transforms.Normalize(**p['transformation_kwargs']['normalize'])])


def get_optimizer(p, model, cluster_head_only=False):
    if cluster_head_only: # Only weights in the cluster head will be updated
        for name, param in model.named_parameters():
                if 'cluster_head' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert(len(params) == 2 * p['num_heads'])

    else:
        params = model.parameters()


    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])

    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']

    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2

    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
