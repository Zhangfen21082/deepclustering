"""
Author: Huasong Zhong
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import json
import torch
import pickle
import numpy as np

"""
这个类名为AugFeat，它的作用是实现一个特征缓存器，用于以内存高效的方式存储特征
具体来说，它可以将特征向量和对应的索引存储在内存字典中，当内存字典中的特征向量数量达到一定阈值时，会将最早的特征向量弹出，
以便存储新的特征向量。此外，该类还提供了一个保存内存字典的方法。
在训练过程中，通常需要多次读取和处理数据，而这些操作可能会占用大量的时间和内存
通过使用这个特征缓存器，可以将特征向量存储在内存中，以便快速读取和处理
这样可以减少磁盘I/O和内存分配的开销，从而提高训练速度

- output_path：指定内存字典保存的路径。
- size：指定内存字典中每个键对应的特征向量数量的最大值。
- alpha：指定内存字典中每个键对应的特征向量的加权平均系数。
"""
class AugFeat:
    # 初始化了内存字典、输出路径、每个键对应的特征向量数量的最大值和加权平均系数
    def __init__(self, output_path, size=20, alpha=0.3):
        self.memory_dict = {}
        self.output_path = output_path
        self.size = size
        self.alpha = alpha
    """ 
    将特征向量和对应的索引存储在内存字典中。对于每个索引，
    如果它不在内存字典中，则将特征向量存储在一个新的列表中，并将该列表存储在内存字典中；
    如果它已经在内存字典中，则将特征向量添加到该索引对应的列表中。如果该列表的长度达到了最大值，则将最早的特征向量弹出
    """
    @torch.no_grad()
    def push(self, feats, indexes):
        for i in range(indexes.shape[0]):
            key = indexes[i].item()
            # data = feats[i][0].cpu().detach().numpy()
            data = feats[i].unsqueeze_(0)
            if key not in self.memory_dict.keys():
                self.memory_dict[key] = [data]
                #self.memory_dict[key] = data
            elif len(self.memory_dict[key]) == self.size:
                self.memory_dict[key].pop(0)
                self.memory_dict[key].append(data)
            else:
                self.memory_dict[key].append(data)
                #self.memory_dict[key] = self.alpha * self.memory_dict[key] + (1.0 - self.alpha) * data

    @torch.no_grad()
    # 从内存字典中弹出给定键对应的特征向量列表
    def pop(self, key):
        return self.memory_dict[key]
    # 将内存字典保存到指定的输出路径中
    def save(self):
        np.save(self.output_path, self.memory_dict)
