from turtle import forward
import torch
import torch.nn as nn
from torch.nn import functional as F
# from torch.nn import init
from torch.nn.parameter import Parameter
import copy
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import math

def swish(x):
    return x * torch.sigmoid(x)

# class GELU(nn.Module):
#     # old version of pytorch does not have gelu
#     def __init__(self):
#         super(GELU, self).__init__()

#     def forward(self, x):
#         return x * 0.5 * (1.0 + torch.erf(x / torch.sqrt(2.0)))


# old version of pytorch does not have gelu
# def gelu(x):
#     return x * 0.5 * (1.0 + torch.erf(x / torch.sqrt(2.0)))

ACT2FN = {"gelu": F.gelu, "relu": F.relu, "swish": swish}


class adaptive_conv1d(nn.Module):

    def __init__(self, out_channel, kernel_size, stride=1, bias=True, padding=None):
        super(adaptive_conv1d, self).__init__()
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.padding = padding

        self._reset_weights()
    
    def _reset_weights(self):
        self._kernels = []
        for _ in range(self.out_channel):
            weight = Parameter(torch.empty(1, 1, self.kernel_size))
            weight = nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            self._kernels.append(copy.deepcopy(weight))
        
        if self.bias:
            self._bias = Parameter(torch.empty(self.out_channel))
        else:
            self._bias = 0

    def _conv_forward(self, input):
        if self.padding == "same":
            _P = "same"
        else:
            _P = "valid"

        feature_maps = []
        in_channel   = input.shape[1]
        for i in range(self.out_channel):
            constr = None
            for j in range(in_channel):
                fff = F.conv1d(input[:, j:j+1, :], self._kernels[i], bias=None, stride=self.stride, padding=_P)
                if constr is None:
                    constr = fff
                else:
                    constr += fff
            feature_maps.append(constr)
        feature_maps = torch.cat(feature_maps, axis=1)
        return feature_maps
    
    def  forward(self, input):
        return self._conv_forward(input)


class Mlp(nn.Module):
    def __init__(self, emb_size):
        super(Mlp, self).__init__()
        self.fc1 = Linear(emb_size, emb_size*4)
        self.fc2 = Linear(emb_size*4, emb_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, emb_size, num_heads):
        super(Attention, self).__init__()
        self.vis = True
        self.num_attention_heads = num_heads
        self.attention_head_size = int(emb_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(emb_size, self.all_head_size)
        self.key = Linear(emb_size, self.all_head_size)
        self.value = Linear(emb_size, self.all_head_size)

        self.out = Linear(self.all_head_size, emb_size)
        self.attn_dropout = Dropout(0.)
        self.proj_dropout = Dropout(0.)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Block(nn.Module):
    def __init__(self, emb_size, num_heads):
        super(Block, self).__init__()
        self.hidden_size = emb_size
        self.attention_norm = nn.LayerNorm(emb_size, eps=1e-6)
        self.ffn_norm =nn. LayerNorm(emb_size, eps=1e-6)
        self.ffn = Mlp(emb_size)
        self.attn = Attention(emb_size, num_heads)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):
    def __init__(self, emb_size, num_heads, num_layers):
        super().__init__()
        self.vis = True
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(emb_size, eps=1e-6)
        self.num_layers = num_layers
        for _ in range(self.num_layers):
            layer = Block(emb_size, num_heads)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        attn_weights = []
        for layer_block in self.layer:
            x, weights = layer_block(x)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(x)
        return encoded, attn_weights


class Scalable_transformer(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.transformer_size_h = 15
        self.transformer_size_w = 20
        self.num_heads = 6
        self.num_layers = 6

        self.conv1 = nn.Conv2d(in_c, in_c//4, kernel_size=1, bias=False)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.transformer_size_h*self.transformer_size_w, in_c//4))
        self.dropout = nn.Dropout(0.1)

        self.encoder = Encoder(in_c//4, self.num_heads, self.num_layers)
        self.conv2 = nn.Conv2d(in_c//4, in_c, kernel_size=1, bias=False)

    def forward(self, x):
        shape_h = x.size(2)
        shape_w = x.size(3)
        x = F.interpolate(x, size=(self.transformer_size_h, self.transformer_size_w), mode='bilinear')
        x = self.conv1(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = x + self.position_embedding
        x = self.dropout(x)

        x, weights = self.encoder(x)
        x = x.transpose(-1, -2)
        b, c, _ = x.size()
        x = x.reshape(b, c, self.transformer_size_h, self.transformer_size_w)
        x = self.conv2(x)
        x = F.interpolate(x, size=(shape_h, shape_w), mode='bilinear')
        return x, weights







