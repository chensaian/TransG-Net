from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, PReLU
from torch_geometric.nn import NNConv, Set2Set, GATConv, GatedGraphConv, SAGEConv
from torch_geometric.nn import global_mean_pool


def drop_path(x, drop_prob: float = 0., training: bool = False):  
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def Spectra_Embedding(x, spec_length, embed_dim):

    batch_size = x.shape[0]
    x = torch.reshape(x, (batch_size, spec_length //
                      embed_dim, embed_dim))  
    return x

class Attention(nn.Module):

    def __init__(self,
                 dim,  
                 num_heads=2,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape  
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(
            drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransG_Net(nn.Module):
    def __init__(self, spec_length=2000, num_classes=1,
                 embed_dim=40, depth=12, num_heads=2, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., norm_layer=None,
                 act_layer=None, num_features=16, dim=32, add_des=False):
        
        
        # MSTransformer
        super(TransG_Net, self).__init__()
        self.num_classes = num_classes
        self.spec_length = spec_length
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, (spec_length//embed_dim) + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[
                      i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.norm_des = norm_layer(12)
        self.head = nn.Linear(self.num_features+12, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)



        # GNN
        self.lin0 = torch.nn.Linear(num_features, dim)

        nnn = Sequential(Linear(4, 64), ReLU(), Linear(64, dim*dim))
        self.conv = NNConv(dim, dim, nnn, aggr='mean')
        self.gru = GRU(dim, dim)
        self.gat = GATConv(dim, dim)
        self.add_des = add_des
        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin11 = torch.nn.Linear(dim+12, dim)
        self.lin12 = torch.nn.Linear(dim, dim)
        self.lin1 = torch.nn.Linear(dim, dim)

        self.lin2 = torch.nn.Linear(dim, 1)
        self.slop = 0.1

    def forward(self, data):

        # MSTransformer
        # [B , xrd_length] --> [B , xrd_length/embed_dim , embed_dim]
        x = data.ms_spec
        x = Spectra_Embedding(x, self.spec_length, self.embed_dim)
        # [1, 1, 100] -> [B, 1, 100] -> [B, xrd_length/embed_dim + 1, 100]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)


        # GNN 
        out = F.leaky_relu(self.lin0(data.x), negative_slope=self.slop)
        # h = out.unsqueeze(0)
        for i in range(1):
            m1 = self.conv(out, data.edge_index, data.edge_attr)
            # m1 = self.conv(out, data.edge_index)
            m1 = F.leaky_relu(m1, negative_slope=self.slop)
        # for i in range(4):
        #     m2 = F.relu(self.conv(out, data.edge_index, data.edge_attr))
        m1 = F.leaky_relu(self.gat(m1, data.edge_index),
                          negative_slope=self.slop)
        for i in range(2):
            m3 = F.leaky_relu(self.conv(m1, data.edge_index,
                              data.edge_attr), negative_slope=self.slop)
        # for i in range(2):
        #     m4 = F.relu(self.conv(out, data.edge_index, data.edge_attr))
        m3 = F.leaky_relu(self.gat(m3, data.edge_index),
                          negative_slope=self.slop)
        for i in range(3):
            m5 = F.leaky_relu(self.conv(m3, data.edge_index,
                              data.edge_attr), negative_slope=self.slop)
        m5 = F.leaky_relu(self.gat(m5, data.edge_index),
                          negative_slope=self.slop)
        out = torch.mean(torch.stack([m1, m3, m5]), 0)
        out = global_mean_pool(out, data.batch)
        if self.add_des:
            out_des = data.des
            out = torch.cat((out, out_des), 1)
            out = F.relu(self.lin11(out))
        else:
            out = F.relu(self.lin12(out))
        out = F.leaky_relu(self.lin1(out), negative_slope=self.slop)
        out = self.lin2(out)
        out = out.view(-1)  
        x = x[:, 0]
        
        
        out_des = self.norm_des(data.des)
        x = torch.cat((x, out_des), 1)
        x = self.head(x)

        return x           # regression token


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def TransG_Net_model(num_classes: int = 1):

    model = TransG_Net(spec_length=2000,
                              embed_dim=40,
                              depth=12,
                              num_heads=2,
                              num_classes=num_classes)
    return model
