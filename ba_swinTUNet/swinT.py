#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@ Author: Rindon
@ Date: 2024-04-15 15:28:44
@ LastEditors: Rindon
@ LastEditTime: 2024-05-17 16:47:10
@ Description: 组件化的swin-Transformer(可能有问题,需要check)
可以作为再下一次的发表内容？
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def drop_path(x, drop_prob: float = 0., training: bool = False):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class DropPath(nn.Module):
    """
    DropPath class
    原理 :Drop Path就是随机将深度学习网络中的多分支结构随机删除。
    功能 :一般可以作为正则化手段加入网络,但是会增加网络训练的难度。如果设置的drop prob过高,模型甚至有可能不收敛。
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Identity(nn.Module):
    """ Identity layer
    The output of this layer is the input without any change.
    Use this layer to avoid if condition in some forward methods
    这一层的输出是没有任何变化的输入。
    使用该层可避免某些前向方法中的 if 条件
    """
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class PatchEmbedding(nn.Module):
    """Patch Embeddings
    Apply patch embeddings on input images. Embeddings is implemented using a Conv2D op.
    Attributes:
        image_size: int, input image size, default: 224
        patch_size: int, size of patch, default: 4
        in_channels: int, input image channels, default: 3
        embed_dim: int, embedding dimension, default: 96
    """

    def __init__(self, image_size=224, patch_size=4, in_channels=3, embed_dim=96,norm_layer=None):
        super().__init__()
        image_size = (image_size, image_size) # TODO: add to_2tuple
        patch_size = (patch_size, patch_size)
        patches_resolution = [image_size[0]//patch_size[0], image_size[1]//patch_size[1]]
        self.image_size = image_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(in_channels=in_channels,
                                     out_channels=embed_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()


    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],#宽度方向的右侧和高度方向的左侧
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class PatchMerging(nn.Module):
    """ Patch Merging class
    Merge multiple patch into one path and keep the out dim.
    Spefically, merge adjacent 2x2 patches(dim=C) into 1 patch.
    The concat dim 4*C is rescaled to 2*C
    Attributes:
        input_resolution: tuple of ints, the size of input
        dim: dimension of single patch
        reduction: nn.Linear which maps 4C to 2C dim
        norm: nn.LayerNorm, applied after linear layer.
    """

    def __init__(self, input_resolution, dim,norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        h, w = self.input_resolution
        b, l, c = x.shape
        x = x.view([b, h, w, c])

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (h % 2 == 1) or (2 % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, 2 % 2, 0, h % 2))

        x0 = x[:, 0::2, 0::2, :] # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :] # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :] # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :] # [B, H/2, W/2, C]
        x = torch.concat([x0, x1, x2, x3], -1) #[B, H/2, W/2, 4*C]
        x = x.view([b, -1, 4*c]) # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)   # [B, H/2*W/2, 2*C]

        return x

class Mlp(nn.Module):
    """ MLP module
    Impl using nn.Linear and activation is GELU, dropout is applied.
    Ops: fc -> act -> dropout -> fc -> dropout
    Attributes:
        fc1: nn.Linear
        fc2: nn.Linear
        act: GELU
        dropout1: dropout after fc1
        dropout2: dropout after fc2
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

def windows_partition(x, window_size):
    """ partite windows into window_size x window_size
    Args:
        x: Tensor, shape=[b, h, w, c]
        window_size: int, window size
    Returns:
        x: Tensor, shape=[num_windows*b, window_size, window_size, c]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return x

def windows_reverse(windows, window_size, H, W):
    """ Window reverse
    Args:
        windows: (n_windows * B, window_size, window_size, C)
        window_size: (int) window size
        H: (int) height of image
        W: (int) width of image
    Returns:
        x: (B, H, W, C)
    """

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """Window based multihead attention, with relative position bias.
    Both shifted window and non-shifted window are supported.
    Attributes:
        dim: int, input dimension (channels)
        window_size: int, height and width of the window
        num_heads: int, number of attention heads
        qkv_bias: bool, if True, enable learnable bias to q,k,v, default: True
        qk_scale: float, override default qk scale head_dim**-0.5 if set, default: None
        attention_dropout: float, dropout of attention
        dropout: float, dropout for output
    """

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 attn_drop=0., 
                 proj_drop=0.):
        super(WindowAttention, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.dim = dim
        self.dim_head = dim // num_heads
        #print(self.dim,self.dim_head,self.num_heads)
        self.scale = self.dim_head ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # relative position index for each token inside window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w])) # [2, window_h, window_w]
        coords_flatten = torch.flatten(coords, 1) # [2, window_h * window_w]
        # 2, window_h * window_w, window_h * window_w
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        '''        
        # Swin-T v2, Scaled cosine attention
        self.tau = torch.create_parameter(
            shape = [num_heads, window_size[0]*window_size[1], window_size[0]*window_size[1]],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(1))
            '''

    '''
    def transpose_multihead(self, x):
        new_shape = x.shape[:-1] + [self.num_heads, self.dim_head]
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])
        return x
        '''
    '''
    def get_relative_pos_bias_from_pos_index(self):
        # relative_position_bias_table is a ParamBase object
        # https://github.com/PaddlePaddle/Paddle/blob/067f558c59b34dd6d8626aad73e9943cf7f5960f/python/paddle/fluid/framework.py#L5727
        table = self.relative_position_bias_table # N x num_heads
        # index is a tensor
        index = self.relative_position_index.reshape([-1]) # window_h*window_w * window_h*window_w
        # NOTE: paddle does NOT support indexing Tensor by a Tensor
        relative_position_bias = paddle.index_select(x=table, index=index)
        return relative_position_bias
    '''
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)       
        

        #q = q * self.scale
        #attn = paddle.matmul(q, k, transpose_y=True)        #SwinV2,修改此处为余弦注意力\
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        '''        
        # SwinV2, Scaled cosine attention
        qk = paddle.matmul(q, k, transpose_y=True)        
        q2 = paddle.multiply(q, q).sum(-1).sqrt().unsqueeze(3)
        k2 = paddle.multiply(k, k).sum(-1).sqrt().unsqueeze(3)
        attn = qk/paddle.clip(paddle.matmul(q2, k2, transpose_y=True), min=1e-6)
        attn = attn/paddle.clip(self.tau.unsqueeze(0), min=0.01)

        relative_position_bias = self.get_relative_pos_bias_from_pos_index() 

        relative_position_bias = relative_position_bias.reshape(
            [self.window_size[0] * self.window_size[1],
             self.window_size[0] * self.window_size[1],
             -1])       

        # nH, window_h*window_w, window_h*window_w
        relative_position_bias = relative_position_bias.transpose([2, 0, 1])  
        attn = attn + relative_position_bias.unsqueeze(0)
        ''' 
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)  

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)    
        return x

class SwinTransformerBlock(nn.Module):
    """Swin transformer block
    Contains window multi head self attention, droppath, mlp, norm and residual.
    Attributes:
        dim: int, input dimension (channels)
        input_resolution: int, input resoultion
        -->input_resolution: tuple, input resoultion

        num_heads: int, number of attention heads
        windos_size: int, window size, default: 7
        shift_size: int, shift size for SW-MSA, default: 0
        mlp_ratio: float, ratio of mlp hidden dim and input embedding dim, default: 4.
        qkv_bias: bool, if True, enable learnable bias to q,k,v, default: True
        qk_scale: float, override default qk scale head_dim**-0.5 if set, default: None
        dropout: float, dropout for output, default: 0.
        attention_dropout: float, dropout of attention, default: 0.
        droppath: float, drop path rate, default: 0.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., norm_layer=nn.LayerNorm,
                 attn_drop=0., drop_path=0.):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = windows_partition(img_mask, self.window_size)
            mask_windows = mask_windows.reshape((-1, self.window_size * self.window_size))
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = torch.where(attn_mask != 0,
                                     torch.ones_like(attn_mask) * float(-100.0),   #这里，关于mask是否真的必要，这部分使整个代码变得复杂了极多
                                     attn_mask)                                     #有些时候，其实我们也想结合图像边缘之间的关系                                
            attn_mask = torch.where(attn_mask == 0,                                #如果将-100设置为0网络也能work的话，Swin将大大减少代码量
                                     torch.zeros_like(attn_mask),
                                     attn_mask)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x,attn_mask):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = x.view(B, H, W, C)
        #x = self.norm1(x)   # [bs,H*W,C]   #后归一化，移到做完attantion之后
        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        
        '''new_shape = [B, H, W, C]
        x = x.reshape(new_shape) # [bs,H,W,C]'''
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        x_windows = windows_partition(shifted_x, self.window_size)  # [bs*num_windows,7,7,C]
        x_windows = x_windows.reshape([-1, self.window_size * self.window_size, C]) # [bs*num_windows,7*7,C]

        attn_windows = self.attn(x_windows, mask=attn_mask)    # [bs*num_windows,7*7,C]
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]

        shifted_x = windows_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)      # [bs,H*W,C] 
        x = self.norm1(x)   # [bs,H*W,C]    #移到这里

        if self.drop_path is not None:
            x = shortcut + self.drop_path(x)
        else:
            x = shortcut + x
        shortcut = x       # [bs,H*W,C]

        '''
        SwinV2,将此处修改为后归一化
        x = self.norm2(x)       # [bs,H*W,C]
        '''

        x = self.mlp(x)         # [bs,H*W,C]
        x = self.norm2(x)       #放在这里

        if self.drop_path is not None:
            x = shortcut + self.drop_path(x)
        else:
            x = shortcut + x
        return x

class SwinT(nn.Module):

    """
    the input shape and output shape is euqal to Conv2D
    use this module can replace Conv2D by SwinT in any scene
    Attribute:
    input_channels:the channels of inputs
    resolution:the only different from cnn, it need resolution to detemine its forward process
    attention_parameters:{num_heads, window_size, mlp_ratio, qkv_bias, qk_scale, dropout, attention_dropout, droppath}
    downsample:like cnn pooling, default:False

    没有使用output_channels因为transformer本身提取特征能力很强,另外下采样会使用patch_merging进行维度*2
    另外由于其输入和输出与CNN完全一致,所以扩大通道可以直接在前面加卷积。
    虽然有那么一点点不一样，但是因为搭建网络还是会把图像大小这样重要信息记一下，所以问题不是很大
    重要的是，它可以完全替换掉任意基于卷积模型中的二维卷积层，因为输入和输出形状完全同卷积，因此十分方便
    在卷积和注意力混用的模型中会更加方便
    """

    def __init__(self, in_channels, input_resolution, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., downsample=None):
        super().__init__()
        self.dim = in_channels
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.blocks = nn.ModuleList()
        for i in range(2):
            self.blocks.append(
                SwinTransformerBlock(dim=in_channels, input_resolution=input_resolution,
                    num_heads=num_heads, window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path))

        if downsample:
            self.downsample = PatchMerging(input_resolution, dim=in_channels)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = windows_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        B, C, H, W = x.shape
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        x = x.view([B, C, H * W])
        x = x.flatten(2).transpose(1, 2)    #[B, H*W, C]

        for block in self.blocks:
            x = block(x,attn_mask)                
        if self.downsample is not None:
            x = self.downsample(x)      
            x = x.flatten(2).transpose(1, 2)    #[B, C * 2, H//2 * W//2]
            #x = x.reshape([B, C * 2, H//2, W//2])
            x = x.view([B, C*2, H//2, W//2])
        else:
            x = x.flatten(2).transpose(1, 2)
            #x = x.reshape([B, C, H, W])
            x = x.view([B, C, H, W])
        return x

if __name__ == '__main__':
    tmp = torch.tensor(np.random.rand(16, 128, 32, 32), dtype=torch.float32)
    print(tmp.shape)
    sts = SwinT(in_channels=128, input_resolution=(32,32), num_heads=4, window_size=4, qkv_bias=False, drop=0.1,
                    attn_drop=0.1, drop_path=0.1,downsample=False)
    out = sts(tmp)
    print(out.shape)