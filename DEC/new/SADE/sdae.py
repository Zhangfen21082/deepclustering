from collections import OrderedDict
from cytoolz.itertoolz import concat, sliding_window
from typing import Callable, Iterable, Optional, Tuple, List
import torch
import torch.nn as nn

# 基本单元构造
def build_units(
    dimensions: Iterable[int], activation: Optional[torch.nn.Module]
) -> List[torch.nn.Module]:
    """
    给定一个维度list和可选的激活函数，返回一个单元list，每个单元为一个线性层跟上一个激活层

    :param dimensions: 可迭代对象
    :param activation: 所使用的激活函数，例如 nn.ReLU, 如果设置为None表示禁用
    :return: 返回一个sequential
    """

    def single_unit(in_dimension: int, out_dimension: int) -> torch.nn.Module:
        unit = [("linear", nn.Linear(in_dimension, out_dimension))]
        if activation is not None:
            unit.append(("activation", activation))
        # OrderedDict和普通的dict基本上是相似的，只有一点不同，那就是OrderedDict中键值对的顺序会保留插入时的顺序
        return nn.Sequential(OrderedDict(unit))

    return [
        single_unit(embedding_dimension, hidden_dimension)
        # slidig_window是一个工具，例如输入sliding_window(2, [1, 2, 3, 4])，其返回
        # 结果是(1, 2) (2, 3), (3, 4)
        # 具体用法 https://blog.csdn.net/weixin_30699955/article/details/102341020
        for embedding_dimension, hidden_dimension in sliding_window(2, dimensions)
    ]

# 参数初始化
def default_initialise_weight_bias_(
    weight: torch.Tensor, bias: torch.Tensor, gain: float
) -> None:
    """
    用于初始化SDAE线性单元权重系数偏置

    :param weight: 线性单元权重张量
    :param bias: 线性单元权重偏置
    :param gain: 用于初始化器的增益
    :return: None
    """

    # xavier_uniform_ 初始化方法
    nn.init.xavier_uniform_(weight, gain)
    nn.init.constant_(bias, 0)

# 堆叠自动编码器
class StackedDenoisingAutoEncoder(nn.Module):
    def __init__(
        self,
        dimensions: List[int],
        activation: torch.nn.Module = nn.ReLU(),
        final_activation: Optional[torch.nn.Module] = nn.ReLU(),
        # Callable是一个在 Python 中的标准类型，它表示任何可调用的对象，例如函数、方法、lambda 函数或类的实例
        # “`[torch.Tensor, torch.Tensor, float], None`”是函数的输入和输出类型的描述。可以看到，输入类型是一个元组
        # 包含两个 torch.Tensor 类型的参数和一个 float 类型的参数；输出类型是 None，表示该函数没有返回值
        # 参数默认值为default_initialise_weight_bias_
        weight_init: Callable[
            [torch.Tensor, torch.Tensor, float], None
        ] = default_initialise_weight_bias_,
        gain: float = nn.init.calculate_gain("relu"),
    ):
        """
        1. 自编码器由对称的编码器（encoder）和解码器（decode）组成，可以分别通过self.encoder和self.decoder获得
        2. 输入的维度是一个列表，例如[100, 10, 10, 5]就表示嵌入维度为100，隐藏维度为5，相应的自编码器的形状就会变为
            [100, 10, 10, 5, 10, 10, 100]

        :param dimensions: 维度输入，是一个list
        :param activation: 激活函数（除最后一层外）, 默认使用nn.ReLU
        :param final_activation: 最后一层激活函数，如果设置为None表示禁用，默认使用nn.ReLU
        :param weight_init: 初始化权重系数和偏置所使用的函数，默认采用default_initialise_weight_bias_
        :param gain: 传递给weight_init的增益参数
        """
        super(StackedDenoisingAutoEncoder, self).__init__()
        self.dimensions = dimensions
        self.embedding_dimension = dimensions[0]  # 嵌入维度
        self.hidden_dimension = dimensions[-1]  # 潜在维度
        # 构造编码器
        # 编码器最后一层不选用激活函数
        encoder_units = build_units(self.dimensions[:-1], activation)
        encoder_units.extend(
            build_units([self.dimensions[-2], self.dimensions[-1]], None)
        )
        self.encoder = nn.Sequential(*encoder_units)
        # 构造解码器
        # 解码器最后一层激活函数选用final_activation
        decoder_units = build_units(reversed(self.dimensions[1:]), activation)
        decoder_units.extend(
            build_units([self.dimensions[1], self.dimensions[0]], final_activation)
        )
        self.decoder = nn.Sequential(*decoder_units)
        # 权重系数和偏置初始化
        for layer in concat([self.encoder, self.decoder]):
            weight_init(layer[0].weight, layer[0].bias, gain)

    def get_stack(self, index: int) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """
        给定一个[0, len(self.dimensions)-2]内的索引，返回对应的子编码器进行逐层预训练

        :param index: 字子编码器索引
        :return: 编码器和解码器单元组
        """
        if (index > len(self.dimensions) - 2) or (index < 0):
            raise ValueError(
                "Requested subautoencoder cannot be constructed, index out of range."
            )
        return self.encoder[index].linear, self.decoder[-(index + 1)].linear

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(batch)
        return self.decoder(encoded)
