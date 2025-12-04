# P0
import numpy as np
import random
import math
import os
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import transformers
import matplotlib.pyplot as plt
# %matplotlib inline

from tqdm.notebook import tqdm
from math import sqrt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from torchinfo import summary


#P1
# convert str to datatime
def convert_to_time(hmm):
    year, month, day, hour, minute, second = int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(hmm[5])
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)


# load .mat data
def loadMat(matfile):
    data = scipy.io.loadmat(matfile)
    filename = matfile.split("/")[-1].split(".")[0]
    col = data[filename]
    col = col[0][0][0][0]
    print("col_shape:",col.shape)
    size = col.shape[0]
    print("size_shape:",size)

    data = []
    for i in range(size):
        k = list(col[i][3][0].dtype.fields.keys())
        d1, d2 = {}, {}
        if str(col[i][0][0]) != 'impedance':
            for j in range(len(k)):
                t = col[i][3][0][0][j][0];
                l = [t[m] for m in range(len(t))]
                d2[k[j]] = l
        d1['type'], d1['temp'], d1['time'], d1['data'] = str(col[i][0][0]), int(col[i][1][0]), str(convert_to_time(col[i][2][0])), d2
        data.append(d1)

    return data


# get capacity data
def getBatteryCapacity(Battery):
    cycle, capacity = [], []
    i = 1
    for Bat in Battery:
        if Bat['type'] == 'discharge':
            capacity.append(Bat['data']['Capacity'][0])
            cycle.append(i)
            i += 1
    return [cycle, capacity]


# get the charge data of a battery
def getBatteryValues(Battery, Type='charge'):
    data=[]
    for Bat in Battery:
        if Bat['type'] == Type:
            data.append(Bat['data'])
    return data

# P1.1
Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']
dir_path = 'nasa/'

Battery = {}
for name in Battery_list:
    print('Load Dataset ' + name + '.mat ...')
    path = dir_path + name + '.mat'
    data = loadMat(path)
    Battery[name] = getBatteryCapacity(data)


# P2.1
def build_instances(sequence, window_size):
    # sequence: list of capacity
    x, y = [], []
    for i in range(len(sequence) - window_size):
        features = sequence[i:i + window_size]
        target = sequence[i + window_size]

        x.append(features)
        y.append(target)

    return np.array(x).astype(np.float32), np.array(y).astype(np.float32)


def split_dataset(data_sequence, train_ratio=0.0, capacity_threshold=0.0):
    if capacity_threshold > 0:
        max_capacity = max(data_sequence)
        capacity = max_capacity * capacity_threshold
        point = [i for i in range(len(data_sequence)) if data_sequence[i] < capacity]
    else:
        point = int(train_ratio + 1)
        if 0 < train_ratio <= 1:
            point = int(len(data_sequence) * train_ratio)
    train_data, test_data = data_sequence[:point], data_sequence[point:]

    return train_data, test_data


# leave-one-out evaluation: one battery is sampled randomly; the remainder are used for training.
def get_train_test(data_dict, name, window_size=8):
    data_sequence = data_dict[name][1]
    train_data, test_data = data_sequence[:window_size + 1], data_sequence[window_size + 1:]
    train_x, train_y = build_instances(train_data, window_size)
    for k, v in data_dict.items():
        if k != name:
            data_x, data_y = build_instances(v[1], window_size)
            train_x, train_y = np.r_[train_x, data_x], np.r_[train_y, data_y]

    return train_x, train_y, list(train_data), list(test_data)




def relative_error(y_test, y_predict):
    n = len(y_test)
    re_sum = 0
    for t in range(n):
        re_sum += abs(y_test[t] - y_predict[t]) / y_test[t]
    re = re_sum / n
    return re



from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    re = relative_error(y_test, y_predict)
    return re, mae, rmse




def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True




# Integrating the UnetTSF model between the Autoencoder and the Transformer (Net) in the provided architecture.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class block_model(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, input_channels, input_len, out_len):
        super(block_model, self).__init__()
        self.channels = input_channels
        self.input_len = input_len
        self.out_len = out_len

        self.Linear_channel = nn.Linear(self.input_len, self.out_len)
        self.ln = nn.LayerNorm(out_len)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # (B,C,N,T) --> (B,C,N,T)
        output = self.Linear_channel(x)
        return output


class Model2(nn.Module):
    def __init__(self, input_channels=64, out_channels=64, seq_len=720, pred_len=720):
        super(Model2, self).__init__()

        self.input_channels = input_channels
        self.out_channels = out_channels
        self.input_len = seq_len
        self.out_len = pred_len

        # 下采样设定
        n1 = 1
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * int(self.input_len)]
        down_in = [int(self.input_len / filters[i]) for i in range(5)]
        down_out = [int(self.out_len / filters[i]) for i in range(5)]

        # 最大池化层
        self.Maxpool1 = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.Maxpool2 = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.Maxpool3 = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.Maxpool4 = nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.GlobalAvgPool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化

        # 左边特征提取层
        self.down_block1 = block_model(self.input_channels, down_in[0], down_out[0])
        self.down_block2 = block_model(self.input_channels, down_in[1], down_out[1])
        self.down_block3 = block_model(self.input_channels, down_in[2], down_out[2])
        self.down_block4 = block_model(self.input_channels, down_in[3], down_out[3])
        self.down_block5 = block_model(self.input_channels, down_in[4], down_out[4])  # 由于全局平均池化的输出长度为1

        # 右边特征融合层
        self.up_block4 = block_model(self.input_channels, down_out[3] + down_out[4], down_out[3])
        self.up_block3 = block_model(self.input_channels, down_out[2] + down_out[3], down_out[2])
        self.up_block2 = block_model(self.input_channels, down_out[1] + down_out[2], down_out[1])
        self.up_block1 = block_model(self.input_channels, down_out[0] + down_out[1], down_out[0])

        # 输出映射
        self.linear_out = nn.Linear(self.input_channels, self.out_channels)

    def forward(self, x):
        x1 = x.permute(0, 3, 1, 2)  # (B,N,T,C) -> (B,C,N,T)
        e1 = self.down_block1(x1)  # (B,C,N,T) -> (B,C,N,T)

        x2 = self.Maxpool1(x1)  # (B,C,N,T) -> (B,C,N,T/2)
        e2 = self.down_block2(x2)  # (B,C,N,T/2) -> (B,C,N,T/2)

        x3 = self.Maxpool2(x2)  # (B,C,N,T/2) -> (B,C,N,T/4)
        e3 = self.down_block3(x3)  # (B,C,N,T/4) -> (B,C,N,T/4)

        x4 = self.Maxpool3(x3)  # (B,C,N,T/4) -> (B,C,N,T/8)
        e4 = self.down_block4(x4)  # (B,C,N,T/8) -> (B,C,N,T/8)

        # 全局平均池化
        x5 = self.GlobalAvgPool(x1)  # (B,C,N,T) -> (B,C,1,1)
        e5 = self.down_block5(x5)  # (B,C,1,1) -> (B,C,1,1)

        # 第五层向第四层融合
        d4 = torch.cat((e4, e5), dim=-1)  # (B,C,N,T/8) + (B,C,1,1) -> (B,C,N,T/8+1)
        d4 = self.up_block4(d4)  # (B,C,N,T/8+1) -> (B,C,N,T/8)

        # 第四层向第三层融合
        d3 = torch.cat((e3, d4), dim=-1)  # (B,C,N,T/4) + (B,C,N,T/8) -> (B,C,N,3T/8)
        d3 = self.up_block3(d3)  # (B,C,N,3T/8) -> (B,C,N,T/4)

        # 第三层向第二层融合
        d2 = torch.cat((e2, d3), dim=-1)  # (B,C,N,T/2) + (B,C,N,T/4) -> (B,C,N,3T/4)
        d2 = self.up_block2(d2)  # (B,C,N,3T/4) -> (B,C,N,T/2)

        # 第二层向第一层融合
        d1 = torch.cat((e1, d2), dim=-1)  # (B,C,N,T) + (B,C,N,T/2) -> (B,C,N,3T/2)
        out = self.up_block1(d1)  # (B,C,N,3T/2) -> (B,C,N,T)

        out = self.linear_out(out.permute(0, 2, 3, 1))  # (B,C,N,T) -> (B,N,T,C)
        return out






class dilated_inception2(nn.Module):
    def __init__(self, cin, cout, seq_len, kernel_set=None, base_dilation_factor=1):
        super(dilated_inception2, self).__init__()
        self.tconv = nn.ModuleList()
        self.padding = 0  # No padding
        self.seq_len = seq_len
        self.base_dilation_factor = base_dilation_factor
        if kernel_set is None:
            self.kernel_set = [2, 4, 8, 3*int(cin)//4]  # Default kernel sizes
        else:
            self.kernel_set = kernel_set
        cout = int(cout / len(self.kernel_set))  # Divide output channels by number of kernels

        # Calculate appropriate dilation factors for each kernel
        self.dilation_factors = self.calculate_dilation_factors(self.seq_len, self.kernel_set, self.base_dilation_factor)

        for kern, dilation_factor in zip(self.kernel_set, self.dilation_factors):
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

        # Calculate input size for the fully connected layer
        min_time_dim = min([self.seq_len - dilation_factor * (kern - 1) for kern, dilation_factor in
                            zip(self.kernel_set, self.dilation_factors)])
        lin_input_size = min_time_dim

        self.out = nn.Sequential(
            nn.Linear(lin_input_size, cin),
            nn.ReLU(),
            nn.Linear(cin, self.seq_len)
        )

    def calculate_dilation_factors(self, seq_len, kernel_set, base_dilation_factor):
        # A simple strategy to calculate dilation factors
        # Here we use a heuristic to spread dilation factors across the kernel sizes
        dilation_factors = [max(1, base_dilation_factor * (seq_len // (2 * k))) for k in kernel_set]
        return dilation_factors

    def forward(self, input):
        # input: (B, C, N, T)
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))  # Perform dilated convolutions with different kernel sizes

        # Align the time dimension by truncating to the minimum length
        min_time_dim = min([xi.size(3) for xi in x])
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -min_time_dim:]

        x = torch.cat(x, dim=1)  # Concatenate along the channel dimension
        x = self.out(x)  # Apply fully connected layers
        return x

class temporal_conv2(nn.Module):
    def __init__(self, cin, cout, seq_len, base_dilation_factor=1):
        super(temporal_conv2, self).__init__()

        self.filter_convs = dilated_inception2(cin=cin, cout=cout, seq_len=seq_len, base_dilation_factor=base_dilation_factor)
        self.gated_convs = dilated_inception2(cin=cin, cout=cout, seq_len=seq_len, base_dilation_factor=base_dilation_factor)
        self.silu_convs = dilated_inception2(cin=cin, cout=cout, seq_len=seq_len, base_dilation_factor=base_dilation_factor)
        self.silu_activation = nn.SiLU()  # Instantiate the SiLU activation function

    def forward(self, X):
        # X:(B,C,N,T)
        filter = self.filter_convs(X)  # 执行左边的DIL层: (B,C,N,T)-->(B,C,N,T)
        filter = torch.tanh(filter)  # 左边的DIL层后接一个tanh激活函数,生成输出:(B,C,N,T)-->(B,C,N,T)
        silu = self.silu_convs(X)
        silu = self.silu_activation(silu)  # Apply SiLU activation function to the tensor
        gate = self.gated_convs(X)  # 执行右边的DIL层: (B,C,N,T)-->(B,C,N,T)
        gate = torch.sigmoid(gate)  # 右边的DIL层后接一个sigmoid门控函数,生成权重表示:(B,C,N,T)-->(B,C,N,T)
        # out = filter * gate * silu # 执行逐元素乘法: (B,C,N,T) * (B,C,N,T) = (B,C,N,T)
        out = filter * gate
        return out




from src.efficient_kan import KAN1,KANLinear1

from torch.nn import init

"Squeeze-and-Excitation Networks"

class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        # 在空间维度上,将H×W压缩为1×1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 包含两层全连接,先降维,后升维。最后接一个sigmoid函数
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            torch.nn.SiLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            torch.nn.SiLU()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # (B,C,H,W)
        B, C, H, W = x.size()
        # Squeeze: (B,C,H,W)-->avg_pool-->(B,C,1,1)-->view-->(B,C)
        y = self.avg_pool(x).view(B, C)
        # Excitation: (B,C)-->fc-->(B,C)-->(B, C, 1, 1)
        y = self.fc(y).view(B, C, 1, 1)
        # scale: (B,C,H,W) * (B, C, 1, 1) == (B,C,H,W)
        out = x * y
        return out

class LocalSEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=3):
        super(LocalSEAttention, self).__init__()
        # 局部卷积层，提取局部信息
        self.conv = nn.Conv2d(channel, channel, kernel_size=kernel_size, padding=kernel_size//2, groups=channel)
        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层，先降维，后升维，最后接一个sigmoid函数
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # (B, C, H, W)
        B, C, H, W = x.size()
        # 局部卷积，提取局部信息
        local_feature = self.conv(x)
        # Squeeze: (B,C,H,W) --> avg_pool --> (B,C,1,1) --> view --> (B,C)
        y = self.avg_pool(local_feature).view(B, C)
        # Excitation: (B,C) --> fc --> (B,C) --> (B, C, 1, 1)
        y = self.fc(y).view(B, C, 1, 1)
        # scale: (B,C,H,W) * (B, C, 1, 1) == (B,C,H,W)
        out = x * y
        return out



from torch import Tensor
from typing import Optional


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate, norm_first=False, batch_first=False):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense1 = nn.Linear(embed_dim, dense_dim)
        self.dense2 = nn.Linear(dense_dim, embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.norm_first = norm_first  # 允许 layernorm 在注意力和前馈网络之前或之后执行

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # fastpath 选项：启用高效路径以减少推理时的内存占用
        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled() and not self.training

        # 使用注意力机制
        if is_fastpath_enabled:
            # 如果启用了 fastpath，使用更高效的路径
            attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask,
                                            need_weights=False)
        else:
            attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)

        attn_output = self.dropout1(attn_output)

        if self.norm_first:
            # 如果启用了 norm_first，先执行 LayerNorm
            src = src + attn_output
            out1 = self.layernorm1(src)
        else:
            out1 = self.layernorm1(src + attn_output)

        # 前馈网络
        dense_output = self.dense1(out1)
        dense_output = self.dense2(dense_output)
        dense_output = self.dropout2(dense_output)

        if self.norm_first:
            # 如果启用了 norm_first，先执行 LayerNorm
            src = out1 + dense_output
            out2 = self.layernorm2(src)
        else:
            out2 = self.layernorm2(out1 + dense_output)

        return out2


class TransformerEncoderLayers(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate, num_layers, norm_first=False, batch_first=False):
        super(TransformerEncoderLayers, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, dense_dim, num_heads, dropout_rate, norm_first, batch_first)
            for _ in range(num_layers)
        ])

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        for layer in self.layers:
            # 每层都支持 mask 和 padding mask
            x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return x


import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional


class Trend_aware_attention(nn.Module):
    '''
    Trend_aware_attention 机制
    X:      [batch_size, num_step, num_vertex, D]
    K:      注意力头数
    d:      每个注意力头的输出维度
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, K, d, kernel_size):
        super(Trend_aware_attention, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.FC_v = nn.Linear(D, D)
        self.FC = nn.Linear(D, D)
        self.kernel_size = kernel_size
        self.padding = self.kernel_size - 1
        self.cnn_q = nn.Conv2d(D, D, (1, self.kernel_size), padding=(0, self.padding))
        self.cnn_k = nn.Conv2d(D, D, (1, self.kernel_size), padding=(0, self.padding))
        self.norm_q = nn.BatchNorm2d(D)
        self.norm_k = nn.BatchNorm2d(D)

        # 调试：打印 kernel_size 和 padding
        print(f"Initialized Trend_aware_attention with kernel_size={self.kernel_size} and padding={self.padding}")

    def forward(self, X):
        batch_size = X.shape[0]
        print("Input X shape:", X.shape)  # 调试：检查输入 X 的形状

        X_ = X.permute(0, 3, 2, 1)  # (B, T, N, D) --> (B, D, N, T)
        print("X_ shape after permute:", X_.shape)  # 调试

        query = self.norm_q(self.cnn_q(X_))[:, :, :, :-self.padding].permute(0, 3, 2, 1)  # 生成 query
        key = self.norm_k(self.cnn_k(X_))[:, :, :, :-self.padding].permute(0, 3, 2, 1)  # 生成 key
        value = self.FC_v(X)  # 生成 value

        print("Query shape:", query.shape)  # 调试
        print("Key shape:", key.shape)  # 调试
        print("Value shape:", value.shape)  # 调试

        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)

        print("Query shape after split:", query.shape)  # 调试
        print("Key shape after split:", key.shape)  # 调试
        print("Value shape after split:", value.shape)  # 调试

        query = query.permute(0, 2, 1, 3)  # (B*k, N, T, d)
        key = key.permute(0, 2, 3, 1)  # (B*k, N, d, T)
        value = value.permute(0, 2, 1, 3)  # (B*k, N, T, d)

        print("Query shape after permute:", query.shape)  # 调试
        print("Key shape after permute:", key.shape)  # 调试
        print("Value shape after permute:", value.shape)  # 调试

        attention = (query @ key) * (self.d ** -0.5)  # 点积注意力
        print("Attention shape:", attention.shape)  # 调试

        attention = F.softmax(attention, dim=-1)

        X = (attention @ value)  # 加权 value
        print("X shape after attention:", X.shape)  # 调试

        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self.FC(X)
        return X.permute(0, 2, 1, 3)  # (B, N, T, D) --> (B, T, N, D)


class TransformerEncoderLayer2(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate, kernel_size, norm_first=False, batch_first=True):
        super(TransformerEncoderLayer2, self).__init__()

        # 使用 TAA 替换原有的 MultiheadAttention
        self.self_attn = Trend_aware_attention(num_heads, embed_dim // num_heads, kernel_size)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense1 = nn.Linear(embed_dim, dense_dim)
        self.dense2 = nn.Linear(dense_dim, embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.norm_first = norm_first

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        batch_size, num_step, embed_dim = src.size()
        src = src.view(batch_size, num_step, 1, embed_dim)  # 调整输入格式

        attn_output = self.self_attn(src)
        attn_output = attn_output.view(batch_size, num_step, embed_dim)
        attn_output = self.dropout1(attn_output)

        if self.norm_first:
            src = src.view(batch_size, num_step, embed_dim) + attn_output
            out1 = self.layernorm1(src)
        else:
            out1 = self.layernorm1(src.view(batch_size, num_step, embed_dim) + attn_output)

        dense_output = self.dense1(out1)
        dense_output = self.dense2(dense_output)
        dense_output = self.dropout2(dense_output)

        if self.norm_first:
            src = out1 + dense_output
            out2 = self.layernorm2(src)
        else:
            out2 = self.layernorm2(out1 + dense_output)

        return out2


class TransformerEncoderLayers2(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate, num_layers, kernel_size, norm_first=False,
                 batch_first=True):
        super(TransformerEncoderLayers2, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer2(embed_dim, dense_dim, num_heads, dropout_rate, kernel_size, norm_first, batch_first)
            for _ in range(num_layers)
        ])

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        for layer in self.layers:
            x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return x






# P2.23 Transformer位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, feature_len, feature_size, dropout=0.0):
        '''
        Args:
            feature_len: the feature length of input data (required).
            feature_size: the feature size of input data (required).
            dropout: the dropout rate (optional).
        '''
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(feature_len, feature_size)
        position = torch.arange(0, feature_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, feature_size, 2).float() * (-math.log(10000.0) / feature_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        print("x_size:", x.shape)
        print("pe_size:",self.pe.shape)
        x = x + self.pe

        return x





class Net(nn.Module):
    def __init__(self, feature_size=16, hidden_dim=32, feature_num=1, num_layers=1, nhead=1, dropout=0.0,
                 noise_level=0.01,alpha_3_1=1, alpha_3_2=1):
        '''
        Args:
            feature_size: the feature size of input data (required).
            hidden_dim: the hidden size of Transformer block (required).
            feature_num: the number of features, such as capacity, voltage, and current; set 1 for only sigle feature (optional).
            num_layers: the number of layers of Transformer block (optional).
            nhead: the number of heads of multi-attention in Transformer block (optional).
            dropout: the dropout rate of Transformer block (optional).
            noise_level: the noise level added in Autoencoder (optional).
        '''
        super(Net, self).__init__()
        self.auto_hidden = int(feature_size / 2)
        input_size = self.auto_hidden

        if feature_num == 1:
            # Transformer treated as an Encoder when modeling for a sigle feature like only capacity data
            self.pos = PositionalEncoding(feature_len=feature_num, feature_size=input_size)
            encoder_layers = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=hidden_dim,
                                                        dropout=dropout, batch_first=True)
        elif feature_num > 1:
            # Transformer treated as a sequence model when modeling for multi-features like capacity, voltage, and current data
            self.pos = PositionalEncoding(feature_len=16, feature_size=16)
            encoder_layers = nn.TransformerEncoderLayer(d_model=feature_num, nhead=nhead, dim_feedforward=hidden_dim,
                                                        dropout=dropout, batch_first=True)
        self.cell = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.cell2 = TransformerEncoderLayers(embed_dim, dense_dim, num_heads, dropout_rate, num_layers, norm_first, batch_first)
        self.cell3 = TransformerEncoderLayers2(embed_dim, dense_dim, num_heads, dropout_rate, num_layers, 4, norm_first, batch_first)


        self.linear = nn.Linear(feature_num * feature_size, 1)
        # self.autoencoder = Autoencoder(input_size=feature_size, hidden_dim=self.auto_hidden, noise_level=noise_level)    #不知道，如果feature_num =2的话，input_size会不会取2*feature_size呢？主要是input_size是否是x的所有尺寸乘积还是只要feature_size这一个维度？
        # 实例化Model2并存储为一个属性
        self.model2 = Model2(input_channels=feature_num, out_channels=feature_num, seq_len=feature_size,pred_len=feature_size)
        # self.model4 = OptimizedBayesianCNN()
        # self.model3 = ECAAttention(kernel_size=3)
        # self.model_3_4 = Multi_GTU(num_of_timesteps=16, in_channels=16, time_strides=1, kernel_size=[3,5,7], pool=True)
        self.model_3_5 = temporal_conv2(cin=feature_num, cout=feature_num, base_dilation_factor=1,seq_len=feature_size)
        self.kan1 = KANLinear1(feature_num * feature_size, 1)
        self.model_1_1 = SEAttention(channel=16,reduction=4)
        self.model_1_1gaidong = LocalSEAttention(channel=16, reduction=4, kernel_size=3)


    def forward(self, x):

        batch_size, feature_num, feature_size = x.shape
        print("shape_x:",x.shape)
        # out, decode = self.autoencoder(x)
        # print("shape_x_autoencoder后的：",out.shape)
        out1 = x
        if feature_num > 1:
            out1 = out1.reshape(batch_size, -1, feature_num)
        print("Encoded output shape:", out1.shape)
        # # out = self.pos(out)
        #
        # out1 = self.pos(out1)
        # (B,N,T,C)
        out1 = out1.reshape(batch_size,-1,feature_size,feature_num)
        print("out reshape适应Unet：",out1.shape)

        out1 = out1.permute(0, 3, 1, 2)
        # out1 = self.model_3_5(out1)
        out1 = self.model_3_5(out1)
        # out1 = self.model_3_5(out1)
        out1 = out1.permute(0, 2, 3, 1)
        # out1 = self.model2(out1)



        out2 = x
        if feature_num > 1:
            out2 = out2.reshape(batch_size, -1,
                                feature_num)
        print("并列trm out2初次reshape shape:", out2.shape)
        out2 = self.pos(out2)
        out2 = self.cell3(
            out2)

        # (B,N,T,C)
        out2 = out2.reshape(batch_size, -1, feature_size,
                            feature_num)
        print("并列trm out2 reshape适应Unet：", out2.shape)
        # out2 = self.model2(out2)

        out0 = alpha_3_1 * out1 + alpha_3_2 * out2
        out0 = out0.permute(0, 3, 1, 2)   #B,C,N,T
        out0 = self.model_larry_4(out0)

        # out0 = out1






        # out0 = self.model2(out0)  # (B,N,T,C)-->(B,N,T,C)
        # print("out0_model2后的:",out0.shape)
        # out = out.permute(0, 3, 1, 2)
        # out = self.model_3_5(out)
        # out = out.permute(0, 2, 3, 1)
        # out = self.model3(out)
        # out = out.reshape(batch_size,-1,feature_num)
        # print("out3_reshape回3维：",out.shape)

        # out = out.reshape(batch_size,feature_num,feature_size)
        # out, decode = self.autoencoder(x)
        # print("shape_x_autoencoder后的：",out.shape)
        out0 = out0.reshape(batch_size,-1,feature_num)


        # out = self.pos(out)
        # out = self.cell(
        #     out)  # sigle feature: (batch_size, feature_num, auto_hidden) or multi-features: (batch_size, auto_hidden, feature_num)
        out0 = out0.reshape(batch_size, -1)  # (batch_size, feature_num*auto_hidden)
        print("out3_cell+reshape后:", out0.shape)
        # out0 = self.kan1(out0)  # out shape: (batch_size, 1);
        out0 = self.linear(out0)

        return out0



# P2.3
def train(lr=0.01, feature_size=8, feature_num=1, hidden_dim=32, num_layers=1, nhead=1, dropout=0.0, epochs=1000,
          weight_decay=0.0, seed=0, alpha=0.0, noise_level=0.0, metric='re', device=('cuda:0' if torch.cuda.is_available() else 'cpu'),
          alpha_3_1=1, alpha_3_2=1):
    '''
        Args:
            lr: learning rate for training (required).
            feature_size: the feature size of input data (required).
            feature_num: the number of features, such as capacity, voltage, and current; set 1 for only sigle feature (optional).
            hidden_dim: the hidden size of Transformer block (required).
            num_layers: the number of layers of Transformer block (optional).
            nhead: the number of heads of multi-attention in Transformer block (optional).
            dropout: the dropout rate of Transformer block (optional).
            epochs:
            weight_decay:
            seed: (optional).
            alpha: (optional).
            noise_level: the noise level added in Autoencoder (optional).
            metric: (optional).
            device: the device for training (optional).
        '''
    score_list, fixed_result_list, moving_result_list = [], [], []

    min_rmse_record = {}

    setup_seed(seed)
    for i in range(4):
        name = Battery_list[i]
        train_x, train_y, train_data, test_data = get_train_test(Battery, name, feature_size)

        print(f"--- Battery: {name} ---")

        print("train_x:")
        print(type(train_x))
        if isinstance(train_x, np.ndarray):
            print(f"ShapeShape: {train_x.shape}")
        print(train_x)

        print("train_y:")
        print(type(train_y))
        if isinstance(train_y, np.ndarray):
            print(f"Shape: {train_y.shape}")
        print(train_y)

        print("train_data:")
        print(type(train_data))
        print(f"Length: {len(train_data)}")
        print(train_data)

        print("test_data:")
        print(type(test_data))
        print(f"Length: {len(test_data)}")
        print(test_data)

        print("----------------------")

        test_sequence = train_data + test_data
        print(f"Shape of test_sequence: {np.array(test_sequence).shape}")
        # print('sample size: {}'.format(len(train_x)))

        model = Net(feature_size=feature_size, hidden_dim=hidden_dim, feature_num=K, num_layers=num_layers,
                    nhead=nhead, dropout=dropout, noise_level=noise_level, alpha_3_1=alpha_3_1, alpha_3_2=alpha_3_2)
        model = model.to(device)


        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        test_x = train_data.copy()
        loss_list, y_fixed_slice, y_moving_slice = [0], [], []
        rmse, re = 1, 1
        score_, score = [1], [1]

        metrics_record = {}  # Dictionary to store metrics

        for epoch in range(epochs):
            print(f'第{i}个电池，第{epoch}次训练，已完成{epoch / epochs}')
            print(f'第{seed}个种子，第{epoch}次训练，已完成{epoch / epochs}')
            x, y = np.reshape(train_x / Rated_Capacity, (-1, feature_num, feature_size)), np.reshape(
                train_y / Rated_Capacity, (-1, 1))
            print("shape_x1:",x.shape,"shape_y1:",y.shape)
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            print("shape_x2:", x.shape, "shape_y2:", y.shape)
            x = x.repeat(1, K, 1)
            print("shape_x3:", x.shape, "shape_y3:", y.shape)
            output = model(x)
            print("shape_x4:", output.shape, "shape_decode:")
            output = output.reshape(-1, 1)
            print("shape_x5:",output.shape,"shape_y4:",y.shape)
            loss = criterion(output, y) # + alpha * criterion(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                test_x = train_data.copy()
                fixed_point_list, moving_point_list = [], []
                t = 0
                while (len(test_x) - len(train_data)) < len(test_data):
                    x = np.reshape(np.array(test_x[-feature_size-1:-1]) / Rated_Capacity,
                                   (-1, feature_num, feature_size)).astype(np.float32)
                    x = torch.from_numpy(x).to(device)
                    print("while里的test_x后feature_size个:",x.shape,"while里的test_x：",np.array(test_x).shape)
                    x = x.repeat(1, K, 1)
                    pred = model(x)
                    next_point = pred.data.cpu().numpy()[0, 0] * Rated_Capacity
                    test_x.append(
                        next_point)  # The test values are added to the original sequence to continue to predict the next point
                    fixed_point_list.append(
                        next_point)  # Saves the predicted value of the last point in the output sequence
                    print(f"Length of fixed_point_list: {len(fixed_point_list)}")

                    x = np.reshape(np.array(test_sequence[t:t + feature_size]) / Rated_Capacity,
                                   (-1, 1, feature_size)).astype(np.float32)
                    x = torch.from_numpy(x).to(device)
                    x = x.repeat(1, K, 1)
                    pred = model(x)
                    next_point = pred.data.cpu().numpy()[0, 0] * Rated_Capacity
                    moving_point_list.append(
                        next_point)  # Saves the predicted value of the last point in the output sequence
                    print(f"Length of moving_point_list: {len(moving_point_list)}")
                    t += 1
                    print("t多少：",t)

                y_fixed_slice.append(fixed_point_list)  # Save all the predicted values
                y_moving_slice.append(moving_point_list)

                loss_list.append(loss)
                # rmse = evaluation(y_test=test_data, y_predict=y_fixed_slice[-1])
                # re = relative_error(y_test=test_data, y_predict=y_fixed_slice[-1])
                re, mae, rmse = calculate_metrics(test_data, y_fixed_slice[-1])
                print(f'Epoch: {epoch + 1}, RE: {re:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}')
                # print('epoch:{:<2d} | loss:{:<6.4f} | RMSE:{:<6.4f} | RE:{:<6.4f}'.format(epoch, loss, rmse, re))
                # Store metrics every 10 epochs
                metrics_record[epoch] = {'RE': re, 'MAE': mae, 'RMSE': rmse}

                # 打印模型结构和参数量
                summary(model, input_size=(1, 16, feature_size))
                # 在每个epoch结束时打印模型参数量
                total_params = sum(p.numel() for p in model.parameters())
                print(f"Total number of parameters: {total_params}")

            if metric == 're':
                score = [re]
            elif metric == 'rmse':
                score = [rmse]
            else:
                score = [re, rmse]
            # if (loss < 1e-3) and (score_[0] < score[0]):
            # 设置不同的早停条件
            if i == 0:
                if rmse < 2e-2:
                    break
            elif i == 1:
                if rmse < 2e-2:
                    break
            elif i == 2:
                if rmse < 2e-2:
                    break
            elif i == 3:
                if rmse < 2e-2:
                    break
            score_ = score.copy()

        score_list.append(score_)
        fixed_result_list.append(train_data.copy() + y_fixed_slice[-1])
        moving_result_list.append(train_data.copy() + y_moving_slice[-1])

        # Find and record minimum RMSE for the current battery
        min_rmse_epoch = min(metrics_record, key=lambda k: metrics_record[k]['RMSE'])
        min_rmse = metrics_record[min_rmse_epoch]['RMSE']
        min_rmse_record[name] = {'min_rmse': min_rmse, 'epoch': min_rmse_epoch}

        print(f"Battery {name} - Minimum RMSE: {min_rmse:.4f} at Epoch {min_rmse_epoch}")

        min_rmse_record[name] = {'min_rmse': min_rmse, 'epoch': min_rmse_epoch, 'alpha_3_1': alpha_3_1,
                                 'alpha_3_2': alpha_3_2}

    return score_list, fixed_result_list, moving_result_list, min_rmse_record



def train_2(lr=0.01, feature_size=8, feature_num=1, hidden_dim=32, num_layers=1, nhead=1, dropout=0.0, epochs=1000,
           weight_decay=0.0, seed=0, alpha=0.0, noise_level=0.0, metric='re',
           device=('cuda:0' if torch.cuda.is_available() else 'cpu'),
           alpha_3_1=1, alpha_3_2=1, K=16):
    '''
        Args:
            lr: 学习率。
            feature_size: 输入数据的特征大小。
            feature_num: 特征数量，如容量、电压和电流；仅使用单个特征时设置为1。
            hidden_dim: Transformer块的隐藏尺寸。
            num_layers: Transformer块的层数。
            nhead: Transformer块中多头注意力的头数。
            dropout: Transformer块的dropout率。
            epochs: 训练的轮数。
            weight_decay: 权重衰减。
            seed: 随机种子。
            alpha: 可选参数。
            noise_level: 自编码器中添加的噪声水平。
            metric: 评估指标。
            device: 训练设备。
            alpha_3_1: 自定义参数。
            alpha_3_2: 自定义参数。
            K: 重复次数（默认为16）
    '''
    score_list, fixed_result_list, moving_result_list = [], [], []

    min_rmse_record = {}

    setup_seed(seed)
    for i in range(4):
        name = Battery_list[i]
        train_x, train_y, train_data, test_data = get_train_test(Battery, name, feature_size)

        test_sequence = train_data + test_data

        model = Net(feature_size=feature_size, hidden_dim=hidden_dim, feature_num=K, num_layers=num_layers,
                   nhead=nhead, dropout=dropout, noise_level=noise_level,
                   alpha_3_1=alpha_3_1, alpha_3_2=alpha_3_2)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        test_x = train_data.copy()
        loss_list, y_fixed_slice, y_moving_slice = [0], [], []
        rmse, re = 1, 1
        score_, score = [1], [1]

        metrics_record = {}  # 存储指标的字典

        for epoch in range(epochs):
            print(f'第{i}个电池，第{epoch}次训练，已完成{epoch / epochs:.2%}')
            print(f'第{seed}个种子，第{epoch}次训练，已完成{epoch / epochs:.2%}')
            x, y = np.reshape(train_x / Rated_Capacity, (-1, feature_num, feature_size)), np.reshape(
                train_y / Rated_Capacity, (-1, 1))
            x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).float().to(device)
            x = x.repeat(1, K, 1)
            output = model(x)
            output = output.reshape(-1, 1)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                test_x = train_data.copy()
                fixed_point_list, moving_point_list = [], []
                t = 0
                while (len(test_x) - len(train_data)) < len(test_data):
                    x = np.reshape(np.array(test_x[-feature_size:]) / Rated_Capacity,
                                   (-1, feature_num, feature_size)).astype(np.float32)
                    x = torch.from_numpy(x).to(device)
                    x = x.repeat(1, K, 1)
                    pred = model(x)
                    next_point = pred.data.cpu().numpy()[0, 0] * Rated_Capacity
                    test_x.append(next_point)
                    fixed_point_list.append(next_point)

                    x = np.reshape(np.array(test_sequence[t:t + feature_size]) / Rated_Capacity,
                                   (-1, 1, feature_size)).astype(np.float32)
                    x = torch.from_numpy(x).to(device)
                    x = x.repeat(1, K, 1)
                    pred = model(x)
                    next_point = pred.data.cpu().numpy()[0, 0] * Rated_Capacity
                    moving_point_list.append(next_point)
                    t += 1

                y_fixed_slice.append(fixed_point_list)
                y_moving_slice.append(moving_point_list)

                loss_list.append(loss.item())
                re, mae, rmse = calculate_metrics(test_data, y_fixed_slice[-1])
                metrics_record[epoch] = {'RE': re, 'MAE': mae, 'RMSE': rmse}

                # 设置不同的早停条件(以根据需要控制，得到数据or画图.mine，跑过了，是有用的）
                if i == 0:
                    if rmse < 2e-2:
                        break
                elif i == 1:
                    if rmse < 2e-2:
                        break
                elif i == 2:
                    if rmse < 2e-2:
                        break
                elif i == 3:
                    if rmse < 2e-2:
                        break

                print(f'Epoch: {epoch + 1}, RE: {re:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}')

        # 找到当前电池的最小 RMSE 并记录
        min_rmse_epoch = min(metrics_record, key=lambda k: metrics_record[k]['RMSE'])
        min_metrics = metrics_record[min_rmse_epoch]
        min_rmse = min_metrics['RMSE']
        min_re = min_metrics['RE']
        min_mae = min_metrics['MAE']
        min_rmse_record[name] = {'min_rmse': min_rmse, 'min_re': min_re, 'min_mae': min_mae, 'epoch': min_rmse_epoch}

        # 计算对应的y_fixed_slice和y_moving_slice索引
        # y_fixed_slice被每10个epoch添加一次，索引对应于epoch=(9,19,29,...)
        # min_rmse_epoch对应的y_fixed_slice索引为 (min_rmse_epoch +1)//10 -1
        index = (min_rmse_epoch + 1) // 10 - 1
        # 防止索引越界
        if index < 0:
            index = 0
        elif index >= len(y_fixed_slice):
            index = len(y_fixed_slice) - 1

        fixed_result_list.append(train_data.copy() + y_fixed_slice[index])
        moving_result_list.append(train_data.copy() + y_moving_slice[index])

    return score_list, fixed_result_list, moving_result_list, min_rmse_record


# P2.5
Rated_Capacity = 2.0
feature_size = 16
feature_num = 1
dropout = 0.00
epochs = 2000
nhead = 8
hidden_dim = 16
num_layers = 2
lr = 0.005
weight_decay = 0.0
noise_level = 0.01
alpha = 0.01
alpha_3_1 = 1
alpha_3_2 = 1
metric = 're'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
K = 16
seed = 4
#
embed_dim = 16
dense_dim = 16
num_heads = 8
dropout_rate = 0.0
batch_first = True
norm_first = True
SCORE = []
print('seed:{}'.format(seed))



#打印一个种子，当下种子

def calculate_metrics(test_data, predict_data):
    test_data = np.array(test_data)  # 将列表转换为numpy数组
    predict_data = np.array(predict_data)  # 将列表转换为numpy数组
    re = np.mean(np.abs((test_data - predict_data) / test_data))  # Relative Error
    mae = np.mean(np.abs(test_data - predict_data))  # Mean Absolute Error
    rmse = np.sqrt(np.mean((test_data - predict_data) ** 2))  # Root Mean Squared Error
    return re, mae, rmse




# 保存文件
import csv
import time
import matplotlib.pyplot as plt


# Define the calculate_metrics function
def calculate_metrics(true_values, predicted_values):
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = sqrt(mean_squared_error(true_values, predicted_values))
    re = relative_error(true_values, predicted_values)
    return re, mae, rmse



def evaluate_and_print3(seeds, lr=lr, feature_size=feature_size, feature_num=feature_num, hidden_dim=hidden_dim,
                        num_layers=num_layers, weight_decay=weight_decay,
                        epochs=epochs, device=('cuda:0' if torch.cuda.is_available() else 'cpu'), alpha_3_1=1,
                        alpha_3_2=1):
    model_fixed_preds = []
    model_moving_preds = []
    seed_metrics = []
    all_min_rmse_records = []

    for seed in seeds:
        setup_seed(seed)
        _, fixed_result_list, moving_result_list, min_rmse_record = train(lr=lr, feature_size=feature_size,
                                                                          feature_num=feature_num,
                                                                          hidden_dim=hidden_dim, num_layers=num_layers,
                                                                          weight_decay=weight_decay,
                                                                          epochs=epochs, seed=seed, device=device,
                                                                          alpha_3_1=alpha_3_1, alpha_3_2=alpha_3_2)
        model_fixed_preds.append(fixed_result_list)
        model_moving_preds.append(moving_result_list)
        all_min_rmse_records.append(min_rmse_record)

    avg_fixed_preds = [[] for _ in range(len(Battery_list))]
    avg_moving_preds = [[] for _ in range(len(Battery_list))]
    for seed_index, (fixed_preds, moving_preds) in enumerate(zip(model_fixed_preds, model_moving_preds)):
        print(f"Metrics for Seed {seeds[seed_index]}:")
        total_re, total_mae, total_rmse = 0, 0, 0
        num_batteries = len(Battery_list)

        for i, battery_name in enumerate(Battery_list):
            re, mae, rmse = calculate_metrics(Battery[battery_name][1], fixed_preds[i][-len(Battery[battery_name][1]):])
            print(f"  Battery {battery_name} - RE: {re:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
            total_re += re
            total_mae += mae
            total_rmse += rmse
            if seed_index == 0:
                avg_fixed_preds[i] = fixed_preds[i][-len(Battery[battery_name][1]):]
                avg_moving_preds[i] = moving_preds[i][-len(Battery[battery_name][1]):]
            else:
                avg_fixed_preds[i] = [sum(x) for x in
                                      zip(avg_fixed_preds[i], fixed_preds[i][-len(Battery[battery_name][1]):])]
                avg_moving_preds[i] = [sum(x) for x in
                                       zip(avg_moving_preds[i], moving_preds[i][-len(Battery[battery_name][1]):])]

        avg_re = total_re / num_batteries
        avg_mae = total_mae / num_batteries
        avg_rmse = total_rmse / num_batteries
        seed_metrics.append((seeds[seed_index], avg_re, avg_mae, avg_rmse))

        print(
            f"  Average Metrics for Seed {seeds[seed_index]} - RE: {avg_re:.4f}, MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}")

        for battery_name, metrics in all_min_rmse_records[seed_index].items():
            print(f"  Battery {battery_name} - Minimum RMSE: {metrics['min_rmse']:.4f} at Epoch {metrics['epoch']}")

    # Calculate the average predictions
    avg_fixed_preds = [[value / len(seeds) for value in battery] for battery in avg_fixed_preds]
    avg_moving_preds = [[value / len(seeds) for value in battery] for battery in avg_moving_preds]

    # Save seed metrics to file
    with open('nasa系列_多个种子各自分数.csv', 'w', newline='') as csvfile:
        fieldnames = ['Seed', 'RE', 'MAE', 'RMSE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for seed, avg_re, avg_mae, avg_rmse in seed_metrics:
            writer.writerow({'Seed': seed, 'RE': avg_re, 'MAE': avg_mae, 'RMSE': avg_rmse})

    # Save battery predictions to file
    for i, battery_name in enumerate(Battery_list):
        test_data = Battery[battery_name][1]
        fixed_predict_data = avg_fixed_preds[i]
        moving_predict_data = avg_moving_preds[i]
        x = list(range(len(test_data)))
        threshold = [Rated_Capacity * 0.7] * len(test_data)

        with open(f'nasa系列，单个电池多个种子平均分数_battery_{battery_name}.csv', 'w', newline='') as csvfile:
            fieldnames = ['x', 'test_data', 'fixed_predict_data', 'moving_predict_data', 'threshold'] + \
                         [f'seed_{seed}_fixed' for seed in seeds] + [f'seed_{seed}_moving' for seed in seeds]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for k in range(len(test_data)):
                row = {
                    'x': x[k],
                    'test_data': test_data[k],
                    'fixed_predict_data': fixed_predict_data[k],
                    'moving_predict_data': moving_predict_data[k],
                    'threshold': threshold[k]
                }
                for seed_index, seed in enumerate(seeds):
                    row[f'seed_{seed}_fixed'] = model_fixed_preds[seed_index][i][k]
                    row[f'seed_{seed}_moving'] = model_moving_preds[seed_index][i][k]

                writer.writerow(row)

    return all_min_rmse_records


def evaluate_and_print3_2(seeds, lr=lr, feature_size=feature_size, feature_num=feature_num, hidden_dim=hidden_dim,
                          num_layers=num_layers, weight_decay=weight_decay,
                          epochs=epochs, device=('cuda:0' if torch.cuda.is_available() else 'cpu'),
                          alpha_3_1=1, alpha_3_2=1, K=16):
    model_fixed_preds = []
    model_moving_preds = []
    seed_metrics = []
    all_min_rmse_records = []

    seed_average_min_rmse = {}
    seed_battery_metrics = {}

    for seed in seeds:
        setup_seed(seed)
        score_list, fixed_result_list, moving_result_list, min_rmse_record = train_2(
            lr=lr, feature_size=feature_size, feature_num=feature_num,
            hidden_dim=hidden_dim, num_layers=num_layers, nhead=nhead,
            dropout=dropout, noise_level=noise_level, weight_decay=weight_decay,
            epochs=epochs, seed=seed, device=device,
            alpha_3_1=alpha_3_1, alpha_3_2=alpha_3_2, K=K
        )
        model_fixed_preds.append(fixed_result_list)
        model_moving_preds.append(moving_result_list)
        all_min_rmse_records.append(min_rmse_record)

        # 收集每个电池的指标
        battery_metrics = min_rmse_record  # 包含每个电池的 min_rmse, min_re, min_mae, epoch

        # 计算 4 个电池的平均最小 RMSE
        total_min_rmse = sum([battery_metrics[name]['min_rmse'] for name in Battery_list])
        avg_min_rmse = total_min_rmse / len(Battery_list)
        seed_average_min_rmse[seed] = avg_min_rmse
        seed_battery_metrics[seed] = battery_metrics

    # 找到平均最小 RMSE 最低的种子
    best_seed = min(seed_average_min_rmse, key=seed_average_min_rmse.get)
    best_seed_metrics = seed_battery_metrics[best_seed]
    avg_min_re = sum([best_seed_metrics[name]['min_re'] for name in Battery_list]) / len(Battery_list)
    avg_min_mae = sum([best_seed_metrics[name]['min_mae'] for name in Battery_list]) / len(Battery_list)
    avg_min_rmse = seed_average_min_rmse[best_seed]

    # 返回最佳种子的预测结果
    seed_index = seeds.index(best_seed)
    best_fixed_preds = model_fixed_preds[seed_index]
    best_moving_preds = model_moving_preds[seed_index]

    # 将最佳种子的电池指标保存到 'nasa系列_多个种子各自分数.csv'
    with open('nasa系列_多个种子各自分数.csv', 'w', newline='') as csvfile:
        fieldnames = ['Battery', 'Seed', 'Min_RE', 'Min_MAE', 'Min_RMSE', 'Epoch', 'alpha_3_1', 'lr']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for battery_name in Battery_list:
            metrics = best_seed_metrics[battery_name]
            writer.writerow({
                'Battery': battery_name,
                'Seed': best_seed,
                'Min_RE': metrics['min_re'],
                'Min_MAE': metrics['min_mae'],
                'Min_RMSE': metrics['min_rmse'],
                'Epoch': metrics['epoch'],
                'alpha_3_1': alpha_3_1,
                'lr': lr
            })

        # 写入平均最小 RE、MAE、RMSE
        writer.writerow({
            'Battery': 'Average',
            'Seed': best_seed,
            'Min_RE': avg_min_re,
            'Min_MAE': avg_min_mae,
            'Min_RMSE': avg_min_rmse,
            'Epoch': '',
            'alpha_3_1': alpha_3_1,
            'lr': lr
        })

    # 保存最佳种子的电池预测结果
    for i, battery_name in enumerate(Battery_list):
        test_data = Battery[battery_name][1]
        fixed_predict_data = best_fixed_preds[i][-len(test_data):]
        moving_predict_data = best_moving_preds[i][-len(test_data):]
        x = list(range(len(test_data)))
        threshold = [Rated_Capacity * 0.7] * len(test_data)

        with open(f'nasa系列，单个电池多个种子平均分数_battery_{battery_name}.csv', 'w', newline='') as csvfile:
            fieldnames = ['x', 'test_data', 'fixed_predict_data', 'moving_predict_data', 'threshold']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for k in range(len(test_data)):
                row = {
                    'x': x[k],
                    'test_data': test_data[k],
                    'fixed_predict_data': fixed_predict_data[k],
                    'moving_predict_data': moving_predict_data[k],
                    'threshold': threshold[k]
                }
                writer.writerow(row)

    # 打印摘要
    print(f"\n最佳种子: {best_seed}，平均最小 RMSE: {avg_min_rmse:.4f}")
    for battery_name in Battery_list:
        metrics = best_seed_metrics[battery_name]
        print(f"电池 {battery_name} - 最小 RMSE: {metrics['min_rmse']:.4f}, 最小 RE: {metrics['min_re']:.4f}, 最小 MAE: {metrics['min_mae']:.4f}，发生在 Epoch {metrics['epoch']}")

    return all_min_rmse_records, model_fixed_preds, model_moving_preds



if __name__ == "__main__":
    # 定义需要遍历的参数值
    alpha_3_1_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    lr_values = [0.005]
    seeds = [0,1,3]
    results = []
    start_time = time.time()

    for alpha_3_1 in alpha_3_1_values:
        alpha_3_2 = 1 - alpha_3_1  # 计算 alpha_3_2
        # alpha_3_2 = 0  # 计算 alpha_3_2
        # alpha_3_2 = 0  # 计算 alpha_3_2
        print(f"alpha_3_1: {alpha_3_1}, calculated alpha_3_2: {alpha_3_2}")  # 打印计算得到的 alpha_3_2 值

        for lr in lr_values:
            print(f"\nEvaluating for alpha_3_1: {alpha_3_1}, lr: {lr}")
            # 调用修改后的 evaluate_and_print3_2 函数
            all_min_rmse_records, model_fixed_preds, model_moving_preds = evaluate_and_print3_2(
                seeds, lr=lr, alpha_3_1=alpha_3_1, alpha_3_2=alpha_3_2, K=K
            )

            # 记录每个种子的平均最小 RMSE
            seed_average_min_rmse = {}
            for seed_index, min_rmse_record in enumerate(all_min_rmse_records):
                seed = seeds[seed_index]
                total_min_rmse = sum([metrics['min_rmse'] for metrics in min_rmse_record.values()])
                avg_min_rmse = total_min_rmse / len(Battery_list)
                seed_average_min_rmse[seed] = avg_min_rmse

                # 记录每个电池的最小 RMSE、RE、MAE
                for battery_name, metrics in min_rmse_record.items():
                    result = {
                        'alpha_3_1': alpha_3_1,
                        'alpha_3_2': alpha_3_2,
                        'lr': lr,
                        'seed': seed,
                        'battery_name': battery_name,
                        'min_rmse': metrics['min_rmse'],
                        'min_re': metrics['min_re'],
                        'min_mae': metrics['min_mae'],
                        'epoch': metrics['epoch']
                    }
                    results.append(result)
                    print(f"\nalpha_3_1: {alpha_3_1}, alpha_3_2: {alpha_3_2}, lr: {lr}, Seed {seed}, Battery {battery_name} - "
                          f"Min RMSE: {metrics['min_rmse']:.4f}, Min RE: {metrics['min_re']:.4f}, "
                          f"Min MAE: {metrics['min_mae']:.4f} at Epoch {metrics['epoch']}")

            # 找到平均最小 RMSE 最低的种子（小组）
            best_seed = min(seed_average_min_rmse, key=seed_average_min_rmse.get)
            best_avg_min_rmse = seed_average_min_rmse[best_seed]
            print(f"\nBest Seed: {best_seed} with Average Min RMSE: {best_avg_min_rmse:.4f}, 'alpha_3_1': {alpha_3_1}, 'alpha_3_2': {alpha_3_2}, 'lr': {lr}")

            # 保存最佳种子的结果到文件
            best_seed_metrics = all_min_rmse_records[seeds.index(best_seed)]
            avg_min_re = sum([metrics['min_re'] for metrics in best_seed_metrics.values()]) / len(Battery_list)
            avg_min_mae = sum([metrics['min_mae'] for metrics in best_seed_metrics.values()]) / len(Battery_list)
            avg_min_rmse = best_avg_min_rmse

            # 将最佳种子的电池指标保存到 '论文1，nasa系列_多个种子各自分数.csv'
            with open('论文1，nasa系列_多个种子各自分数.csv', 'w', newline='') as csvfile:
                fieldnames = ['Battery', 'Seed', 'Min_RE', 'Min_MAE', 'Min_RMSE', 'Epoch', 'alpha_3_1', 'alpha_3_2', 'lr']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for battery_name in Battery_list:
                    metrics = best_seed_metrics[battery_name]
                    writer.writerow({
                        'Battery': battery_name,
                        'Seed': best_seed,
                        'Min_RE': metrics['min_re'],
                        'Min_MAE': metrics['min_mae'],
                        'Min_RMSE': metrics['min_rmse'],
                        'Epoch': metrics['epoch'],
                        'alpha_3_1': alpha_3_1,
                        'alpha_3_2': alpha_3_2,
                        'lr': lr
                    })

                # 写入平均最小 RE、MAE、RMSE
                writer.writerow({
                    'Battery': 'Average',
                    'Seed': best_seed,
                    'Min_RE': avg_min_re,
                    'Min_MAE': avg_min_mae,
                    'Min_RMSE': avg_min_rmse,
                    'Epoch': '',
                    'alpha_3_1': alpha_3_1,
                    'alpha_3_2': alpha_3_2,
                    'lr': lr
                })

            # 保存最佳种子的电池预测结果
            fixed_preds = model_fixed_preds[seeds.index(best_seed)]
            moving_preds = model_moving_preds[seeds.index(best_seed)]

            for i, battery_name in enumerate(Battery_list):
                test_data = Battery[battery_name][1]
                fixed_predict_data = fixed_preds[i][-len(test_data):]
                moving_predict_data = moving_preds[i][-len(test_data):]
                x = list(range(len(test_data)))
                threshold = [Rated_Capacity * 0.7] * len(test_data)

                with open(f'论文1，nasa系列，单个电池多个种子平均分数_battery_{battery_name}.csv', 'w', newline='') as csvfile:
                    fieldnames = ['x', 'test_data', 'fixed_predict_data', 'moving_predict_data', 'threshold']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writeheader()
                    for k in range(len(test_data)):
                        row = {
                            'x': x[k],
                            'test_data': test_data[k],
                            'fixed_predict_data': fixed_predict_data[k],
                            'moving_predict_data': moving_predict_data[k],
                            'threshold': threshold[k]
                        }
                        writer.writerow(row)

    # 汇总打印结果
    print("\nSummary of all results:")
    for result in results:
        print(f"alpha_3_1: {result['alpha_3_1']}, alpha_3_2: {result['alpha_3_2']}, lr: {result['lr']}, Seed: {result['seed']}, "
              f"Battery: {result['battery_name']}, Min RMSE: {result['min_rmse']:.4f}, "
              f"Min RE: {result['min_re']:.4f}, Min MAE: {result['min_mae']:.4f}, Epoch: {result['epoch']}")

    # **添加以下代码来计算并打印平均分数**
    # 根据 alpha_3_1、alpha_3_2、lr、seed 进行分组
    from collections import defaultdict

    grouped_results = defaultdict(lambda: {'rmse': [], 're': [], 'mae': []})

    for result in results:
        key = (result['alpha_3_1'], result['alpha_3_2'], result['lr'], result['seed'])
        grouped_results[key]['rmse'].append(result['min_rmse'])
        grouped_results[key]['re'].append(result['min_re'])
        grouped_results[key]['mae'].append(result['min_mae'])

    # 计算并打印每个组合的平均分数
    print("\nAverage scores per combination of alpha_3_1, alpha_3_2, lr, seed:")
    for key, metrics in grouped_results.items():
        avg_rmse = sum(metrics['rmse']) / len(metrics['rmse'])
        avg_re = sum(metrics['re']) / len(metrics['re'])
        avg_mae = sum(metrics['mae']) / len(metrics['mae'])
        alpha_3_1, alpha_3_2, lr, seed = key
        print(f"alpha_3_1: {alpha_3_1}, alpha_3_2: {alpha_3_2}, lr: {lr}, Seed: {seed}, "
              f"Average Min RMSE: {avg_rmse:.4f}, Average Min RE: {avg_re:.4f}, Average Min MAE: {avg_mae:.4f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")