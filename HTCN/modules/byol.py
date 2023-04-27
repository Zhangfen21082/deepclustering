import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms as T

# helper functions

"""
如果val为None，则返回默认值defval，否则返回val
"""
def default(val, def_val):
    return def_val if val is None else val

"""
展平
"""
def flatten(t):
    return t.reshape(t.shape[0], -1)

"""
这个装饰器的作用是为了避免重复创建实例，提高代码的效率。
"""

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

"""
损失函数，用于计算两个向量x和y之间的相似度
的是最小化两个向量之间的欧几里得距离，从而使它们更加相似
"""

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


"""
随机应用一个函数fn到输入x上，以一定的概率p。如果随机数大于p，则直接返回输入x，
否则返回fn(x)。这个类通常用于数据增强，以增加模型的鲁棒性和泛化能力
"""

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


# 指数平均
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# 动量更新
def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(
        current_model.parameters(), ma_model.parameters()
    ):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


# MLP class for projector and predictor
class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x):
        return self.net(x)


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

"""
是一个神经网络的包装器，用于管理隐藏层输出的拦截并将其传递到projector和predictor网络中
"""
class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer=-2, cluster_num=10):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = None
        self.hook_registered = False
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.projection_size, self.projection_size),
            nn.ReLU(),
            nn.Linear(self.projection_size, cluster_num),
            nn.Softmax(dim=1)
        )

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f"hidden layer ({self.layer}) not found"
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton("projector")
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if not self.hook_registered:
            self._register_hook()

        if self.layer == -1:
            return self.net(x)

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f"hidden layer {self.layer} never emitted an output"
        return hidden

    def forward(self, x):
        representation = self.get_representation(x)
        projector = self._get_projector(representation)
        projection = projector(representation)
        cluster_prediction = self.cluster_projector(projection)
        return projection, cluster_prediction


# main class


class BYOL(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer=-2,
        projection_size=256,
        projection_hidden_size=4096,
        augment_fn=None,
        moving_average_decay=0.99,
        cluster_num=10,
    ):
        super().__init__()

        self.online_encoder = NetWrapper(
            net, projection_size, projection_hidden_size, layer=hidden_layer, cluster_num=cluster_num
        )
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(
            projection_size, projection_size, projection_hidden_size
        )

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size), torch.randn(2, 3, image_size, image_size),
                     torch.randn(2, 3, image_size, image_size))

    @singleton("target_encoder")
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert (
            self.target_encoder is not None
        ), "target encoder has not been created yet"
        update_moving_average(
            self.target_ema_updater, self.target_encoder, self.online_encoder
        )

    def forward(self, image_one, image_two, image_target):
        online_proj_one, o_i = self.online_encoder(image_one)
        online_proj_two, o_j = self.online_encoder(image_two)
        online_proj_three, o_k = self.online_encoder(image_target)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)
        online_pred_three = self.online_predictor(online_proj_three)

        with torch.no_grad():
            target_encoder = self._get_target_encoder()
            target_one, t_i = target_encoder(image_one)
            target_two, t_j = target_encoder(image_two)
            target_three, t_k = target_encoder(image_target)

        loss_one = loss_fn(online_pred_one, target_three.detach())
        loss_two = loss_fn(online_pred_three, target_one.detach())

        loss_three = loss_fn(online_pred_two, target_three.detach())
        loss_four = loss_fn(online_pred_three, target_two.detach())

        loss = loss_one + loss_two + loss_three + loss_four
        return loss.mean(), o_i, o_j, o_k, t_i, t_j, t_k

    def forward_cluster(self, x):
        z, c = self.target_encoder(x)
        c = torch.argmax(c, dim=1)
        return c
