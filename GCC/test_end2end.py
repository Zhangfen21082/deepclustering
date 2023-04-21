"""
Authors: Huasong Zhong
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch
import numpy as np

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_optimizer,\
                                adjust_learning_rate
from utils.evaluate_utils import contrastive_evaluate, get_predictions, hungarian_evaluate
from utils.memory import MemoryBank
from utils.train_utils import gcc_train
from utils.utils import fill_memory_bank, fill_memory_bank_mean
from termcolor import colored
from utils.aug_feat import AugFeat
from data import ConcatDataset

# 参数
parser = argparse.ArgumentParser(description='Graph Contrastive Clustering')
# 指定环境配置文件路径
parser.add_argument('--config_env',
                    help='Config file for the environment')
# 指定实验配置文件路径
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()

# 解释一下这段代码作用，并逐行进行解释（用中文回答）
def main():
    # 创建两个AugFeat对象，分别用于存储原始特征和增强后的特征（特征缓存器，以加快存储速度）
    org_feat_memory = AugFeat('./org_feat_memory', 4)
    aug_feat_memory = AugFeat('./aug_feat_memory', 4)

    # create_config函数，该函数返回一个配置字典p
    print (args.config_env)
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))

    # 利用get_model函数获取模型并答应模型的类名和参数数量
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    model = model.cuda()

    # 启用benchmark加速
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # 数据集
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    print('Train transforms:', train_transforms)
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)

    # 内存库MemoryBank', 'blue'))
    base_dataset = get_train_dataset(p, val_transforms, to_end2end_dataset=True, split='train') # Dataset for performance test
    # for compare with SCAN
    # base_dataset = get_val_dataset(p, val_transforms, to_end2end_dataset=True) # Dataset for performance test
    base_dataloader = get_val_dataloader(p, base_dataset)
    print('Dataset contains {} test samples'.format(len(base_dataset)))
    
    # 使用 MemoryBank 类创建一个内存库，该库用于存储特征向量。这个内存库的
    # 大小是 len(base_dataset)
    # 特征向量的维度是 p['model_kwargs']['features_dim']
    # 类别数是 p['num_classes']，温度是 p['criterion_kwargs']['temperature']
    memory_bank_base = MemoryBank(len(base_dataset),
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_base.cuda()

    # Checkpoint
    # end2end_model for kmeans model
    if os.path.exists(p['end2end_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['end2end_checkpoint']), 'blue'))
        checkpoint = torch.load(p['end2end_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])

        # imagenet10
        #model_dict = model.state_dict()
        #pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        #model_dict.update(pretrained_dict)
        #model.load_state_dict(model_dict)

        model.cuda()
    else:
        print(colored('No checkpoint file at {}'.format(p['end2end_checkpoint']), 'blue'))
        exit(-1)

    # 将数据集中的图像通过模型转换为特征向量，并将这些特征向量存储到内存库中
    fill_memory_bank(base_dataloader, model, memory_bank_base)

    # 内存库中查找最近的邻居，并计算在验证集上的准确率
    # for topk in range(5, 51, 5):
    for topk in range(5, 6, 5):
        indices, acc, detail_acc = memory_bank_base.mine_nearest_neighbors(topk)
        print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))

    # 将内存库中的特征向量和对应的标签保存到文件中
    memory_bank_base.cpu()
    with open (p['features'], 'wb') as f:
        np.save(f, memory_bank_base.features)
    with open (p['features'] + "_label", 'wb') as f:
        np.save(f, memory_bank_base.targets)

    #from tsne import kmeans
    #kmeans(memory_bank_base.features.cpu().numpy(), memory_bank_base.targets.cpu().numpy())

    # 使用预训练模型对数据集进行聚类，并输出聚类结果的统计信息
    predictions, features, targets = get_predictions(p, base_dataloader, model, return_features=True)
    lowest_loss_head = 0
    clustering_stats = hungarian_evaluate(lowest_loss_head, predictions, compute_confusion_matrix=False)
    print(clustering_stats)

if __name__ == '__main__':
    main()
