# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import drop_path, to_2tuple
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model
import utils
from utils import load_state_dict

####para lora
import loralib as lora

##############################################################################

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


##############################################################################
def adapt_checkpoint_for_lora(original_state_dict, model):
    """
    Adapta el state_dict de un modelo entrenado sin LoRA para que sea compatible
    con el modelo actual que incluye parámetros LoRA (como 'lora_A', 'lora_B')
    y también rellena con ceros los sesgos que falten (p.ej. 'attn.qkv.bias').
    """
    # Obtén el state_dict actual del modelo (que contiene todas las claves)
    model_state = model.state_dict()
    # Copia el state_dict del checkpoint
    new_state_dict = dict(original_state_dict)
    for key, param in model_state.items():
        if key not in new_state_dict:
            # Rellena las claves de LoRA
            if "lora_" in key:
                print(f"Adaptando clave faltante {key} con ceros.")
                new_state_dict[key] = torch.zeros_like(param)
            # Rellena también el sesgo qkv si falta (por ejemplo, "attn.qkv.bias")
            elif key.endswith("attn.qkv.bias"):
                print(f"Adaptando sesgo faltante {key} con ceros.")
                new_state_dict[key] = torch.zeros_like(param)
    return new_state_dict


##############################################################################

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


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
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., attn_head_dim=None, use_rpb=False, window_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        # relative positional bias option
        self.use_rpb = use_rpb
        if use_rpb:
            self.window_size = window_size
            self.rpb_table = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
            trunc_normal_(self.rpb_table, std=.02)

            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, h, w
            coords_flatten = torch.flatten(coords, 1)  # 2, h*w
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, h*w, h*w
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # h*w, h*w, 2
            relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)  # h*w, h*w
            self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.use_rpb:
            relative_position_bias = self.rpb_table[self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)  # h*w,h*w,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, h*w, h*w
            attn += relative_position_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


############### 1. Modificar la clase Attention para usar LoRA ##################

class LoRAAttention(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., attn_head_dim=None,
                 use_rpb=False, window_size=14,
                 r=4, lora_alpha=1, lora_dropout=0., merge_weights=True):
        super().__init__(dim, num_heads, qkv_bias, qk_scale,
                         attn_drop, proj_drop, attn_head_dim, use_rpb, window_size)
        # mismas dimensiones que en la capa original
        self.qkv = lora.Linear(dim, 3 * dim, r=r, lora_alpha=lora_alpha,
                               lora_dropout=lora_dropout, merge_weights=merge_weights, bias=qkv_bias)
        self.proj = lora.Linear(dim, dim, r=r, lora_alpha=lora_alpha,
                                lora_dropout=lora_dropout, merge_weights=merge_weights, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias,
                                  torch.zeros_like(self.v_bias, requires_grad=False),
                                  self.v_bias))
        # La salida de self.qkv tendrá tamaño [B, N, (3*dim)//num_heads * num_heads] = [B, N, 3*dim]
        qkv = F.linear(x, self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if self.use_rpb:
            relative_position_bias = self.rpb_table[self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)
            attn += relative_position_bias.permute(2, 0, 1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


##############################################################################################


""" original block
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, use_rpb=False, window_size=14):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim,
            use_rpb=use_rpb, window_size=window_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x """


############################## Block with Lora

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 init_values=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 attn_head_dim=None,
                 use_rpb=False,
                 window_size=14,
                 attn_class=LoRAAttention,  # Por defecto LoRAAttention
                 r=4  # Aquí se pasa el rango
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
            use_rpb=use_rpb,
            window_size=window_size,
            r=r  # Pasa 'r' en lugar de 'rank'
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if init_values and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, mask_cent=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.mask_cent = mask_cent

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        if self.mask_cent:
            x[:, -1] = x[:, -1] - 0.5
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31


def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


##################################### Colorization #################################


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class CnnHead(nn.Module):
    def __init__(self, embed_dim, num_classes, window_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.window_size = window_size

        self.head = nn.Conv2d(embed_dim, num_classes, kernel_size=3, stride=1, padding=1, padding_mode='reflect')

    def forward(self, x):
        x = rearrange(x, 'b (p1 p2) c -> b c p1 p2', p1=self.window_size, p2=self.window_size)
        x = self.head(x)
        x = rearrange(x, 'b c p1 p2 -> b (p1 p2) c')
        return x


class LocalAttentionHead(nn.Module):
    def __init__(
            self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, use_rpb=False, window_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        # masking attn
        mask = torch.ones((window_size ** 2, window_size ** 2))
        kernel_size = 3
        for i in range(window_size):
            for j in range(window_size):
                cur_map = torch.ones((window_size, window_size))
                stx, sty = max(i - kernel_size // 2, 0), max(j - kernel_size // 2, 0)
                edx, edy = min(i + kernel_size // 2, window_size - 1), min(j + kernel_size // 2, window_size - 1)
                cur_map[stx:edx + 1, sty:edy + 1] = 0
                cur_map = cur_map.flatten()
                mask[i * window_size + j] = cur_map
        self.register_buffer('mask', mask)

        # relative positional bias option
        self.use_rpb = use_rpb
        if use_rpb:
            self.window_size = window_size
            self.rpb_table = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
            trunc_normal_(self.rpb_table, std=.02)

            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, h, w
            coords_flatten = torch.flatten(coords, 1)  # 2, h*w
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, h*w, h*w
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # h*w, h*w, 2
            relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)  # h*w, h*w
            self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # masking attn
        mask_value = max_neg_value(attn)
        attn.masked_fill_(self.mask.bool(), mask_value)

        if self.use_rpb:
            relative_position_bias = self.rpb_table[self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)  # h*w,h*w,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, h*w, h*w
            attn += relative_position_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class IColoriT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    ############# original


    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=512, embed_dim=512, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_rpb=False, avg_hint=False, head_mode='default', mask_cent=False):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 2 * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.avg_hint = avg_hint

        # self.mask_token = nn.Parameter(torch.zeros(2))
        # trunc_normal_(self.mask_token, std=.02)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim, mask_cent=mask_cent)
        num_patches = self.patch_embed.num_patches  # 2

        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            init_values=init_values, use_rpb=use_rpb, window_size=img_size // patch_size)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        if head_mode == 'linear':
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        elif head_mode == 'cnn':
            self.head = CnnHead(embed_dim, num_classes, window_size=img_size // patch_size)
        elif head_mode == 'locattn':
            self.head = LocalAttentionHead(embed_dim, num_classes, window_size=img_size // patch_size)
        else:
            raise NotImplementedError('Check head type')

        self.tanh = nn.Tanh()

        self.apply(self._init_weights) """

    ####################### 2 clase IColoriT para usar LoRA en attention
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=512, embed_dim=512, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_rpb=False, avg_hint=False, head_mode='default', mask_cent=False, rank=1):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.avg_hint = avg_hint

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim, mask_cent=mask_cent)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                use_rpb=use_rpb,
                window_size=img_size // patch_size,
                attn_class=LoRAAttention,  # <--- le dices "usa LoRAAttention"
                r=rank
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        if head_mode == 'linear':
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        elif head_mode == 'cnn':
            self.head = CnnHead(embed_dim, num_classes, window_size=img_size // patch_size)
        elif head_mode == 'locattn':
            self.head = LocalAttentionHead(embed_dim, num_classes, window_size=img_size // patch_size)
        else:
            raise NotImplementedError('Check head type')

        self.tanh = nn.Tanh()

        self.apply(self._init_weights)

    ######### 3 congelar los parámetros que no sean LoRA

    def freeze_except_lora(self):
        for name, param in self.named_parameters():
            if 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):
        # x: (B, 3, H, W)   (L normalizada en canal 0, a/b normalizados en 1/2)
        # mask:
        #   - patch-mask 2D: (B, N) con 0=hint, 1=no-hint
        #   - o "Zhang" 4D : (B, 3, H, W) -> [a_norm, b_norm, mask01]

        B, _, H, W = x.shape

        if mask.dim() == 2:
            # ===== PATCH-MASK 2D =====
            _, L = mask.shape
            hint_size = int(math.sqrt(H * W // L))
            _device = '.cuda' if x.device.type == 'cuda' else ''

            # 0=hint, 1=no-hint  -> upsample a tamaño HxW
            mask = torch.reshape(mask, (B, H // hint_size, W // hint_size))
            _mask = mask.unsqueeze(1).type(f'torch{_device}.FloatTensor')  # (B,1,H/hint,W/hint)
            _full_mask = F.interpolate(_mask, scale_factor=hint_size)  # (B,1,H,W) float
            full_mask = _full_mask.type(f'torch{_device}.BoolTensor')  # (B,1,H,W) bool

            # En rama patch-mask, el paper original enmascara a/b fuera de hints
            if self.avg_hint:
                _avg_x = F.interpolate(x, size=(H // hint_size, W // hint_size), mode='bilinear')
                _avg_x[:, 1, :, :].masked_fill_(mask.squeeze(1), 0)
                _avg_x[:, 2, :, :].masked_fill_(mask.squeeze(1), 0)
                x_ab = F.interpolate(_avg_x, scale_factor=hint_size, mode='nearest')[:, 1:, :, :]
                x = torch.cat((x[:, 0:1, :, :], x_ab), dim=1)
            else:
                x[:, 1, :, :].masked_fill_(full_mask.squeeze(1), 0)
                x[:, 2, :, :].masked_fill_(full_mask.squeeze(1), 0)

        elif mask.dim() == 4:
            # ===== ZHANG 4D =====
            # mask = (B,3,H,W): canal 0=a_norm, 1=b_norm, 2=mask01 (1=hint)
            _device = '.cuda' if x.device.type == 'cuda' else ''
            _full_mask_ab = mask.type(f'torch{_device}.FloatTensor')[:, :2, :, :]  # (B,2,H,W)
            x[:, 1:, :, :] = _full_mask_ab  # pisa a/b con lo que viene
            _full_mask = mask[:, 2, :, :].unsqueeze(1).type(f'torch{_device}.FloatTensor')  # (B,1,H,W)
        else:
            raise NotImplementedError(f'Check the mask dimension: got {mask.dim()}.')

        # Canal extra si el modelo es 4ch (el 4º canal suele ser 1-mask)
        if self.in_chans == 4:
            x = torch.cat((x, 1 - _full_mask), dim=1)

        # ----- debug opcional -----
        print(f"[DEBUG][forward_features] x post-masking shape: {x.shape}")
        print(f"[DEBUG][forward_features] x post-masking stats: min={x.min().item():.4f}, "
              f"max={x.max().item():.4f}, mean={x.mean().item():.4f}")

        x = self.patch_embed(x)
        print(f"[DEBUG][forward_features] patch_embed output shape: {x.shape}")

        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        print(f"[DEBUG][forward_features] after pos_embed sum shape: {x.shape}")

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            print(f"[DEBUG][forward_features] After block {i} shape: {x.shape}")
        x = self.norm(x)
        print(f"[DEBUG][forward_features] After norm shape: {x.shape}")
        print(f"[DEBUG][forward_features] x final stats: min={x.min().item():.4f}, "
              f"max={x.max().item():.4f}, mean={x.mean().item():.4f}")
        print("[DEBUG][forward_features] Saliendo de forward_features\n")
        return x

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        # Operación dummy para asegurar que todos los parámetros sean usados en el grafo de cómputo
        dummy = sum(p.sum() for p in self.parameters() if p.requires_grad)  # new
        x = x + 0 * dummy  # new
        x = self.head(x)
        x = self.tanh(x)
        return x


@register_model
def icolorit_tiny_4ch_patch8_224(pretrained=False, **kwargs):
    model = IColoriT(
        num_classes=128,
        img_size=224,
        patch_size=8,
        in_chans=4,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def icolorit_tiny_4ch_patch16_224(pretrained=False, **kwargs):
    model = IColoriT(
        num_classes=512,
        img_size=224,
        patch_size=16,
        in_chans=4,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def icolorit_tiny_4ch_patch32_224(pretrained=False, **kwargs):
    model = IColoriT(
        num_classes=2048,
        img_size=224,
        patch_size=32,
        in_chans=4,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def icolorit_small_4ch_patch16_224(pretrained=False, **kwargs):
    model = IColoriT(
        img_size=224,
        patch_size=16,
        in_chans=4,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def icolorit_base_4ch_patch16_224(pretrained=False, **kwargs):
    init_ckpt_path = kwargs.pop('init_ckpt', None)
    model = IColoriT(
        num_classes=512,
        img_size=224,
        patch_size=16,
        in_chans=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()
    ###original####
    # if pretrained:
    # checkpoint = torch.load(
    # kwargs["init_ckpt"], map_location="cpu"
    # )
    # model.load_state_dict(checkpoint["model"])
    # return model

    ###### AGREGADO POR MI PARA FINETUNEAR########
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    """if pretrained:
        #init_ckpt_path = kwargs.get("init_ckpt")  # Obtiene el path del checkpoint si está presente, de lo contrario devuelve None
        if init_ckpt_path is not None:  # Si se proporcionó un path, carga el checkpoint
            checkpoint = torch.load(init_ckpt_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            print("fine tune model on")
            print("Loaded pretrained model from checkpoint:",init_ckpt_path)"""
    if pretrained:
        if init_ckpt_path is not None:
            checkpoint = torch.load(init_ckpt_path, map_location="cpu")
            adapted_state = adapt_checkpoint_for_lora(checkpoint["model"], model)
            model.load_state_dict(adapted_state)
            print("fine tune model on")
            print("Loaded pretrained model from checkpoint:", init_ckpt_path)

    return model


############################ definido por mi para acortar la red

@register_model
def icolorit_base_4ch_patch16_224_depth_6(pretrained=False, **kwargs):
    init_ckpt_path = kwargs.pop('init_ckpt', None)
    model = IColoriT(
        num_classes=512,
        img_size=224,
        patch_size=16,
        in_chans=4,
        embed_dim=768,
        depth=6,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        if init_ckpt_path is not None:  # Si se proporcionó un path, carga el checkpoint
            checkpoint = torch.load(init_ckpt_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            print("fine tune model on")
            print("Loaded pretrained model from checkpoint:", init_ckpt_path)
    return model


@register_model
def icolorit_base_4ch_patch16_224_depth_4(pretrained=False, **kwargs):
    init_ckpt_path = kwargs.pop('init_ckpt', None)
    model = IColoriT(
        num_classes=512,
        img_size=224,
        patch_size=16,
        in_chans=4,
        embed_dim=768,
        depth=4,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        if init_ckpt_path is not None:  # Si se proporcionó un path, carga el checkpoint
            checkpoint = torch.load(init_ckpt_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            print("fine tune model on")
            print("Loaded pretrained model from checkpoint:", init_ckpt_path)
    return model


@register_model
def icolorit_base_4ch_patch16_224_depth_8(pretrained=False, **kwargs):
    init_ckpt_path = kwargs.pop('init_ckpt', None)
    model = IColoriT(
        num_classes=512,
        img_size=224,
        patch_size=16,
        in_chans=4,
        embed_dim=768,
        depth=8,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        if init_ckpt_path is not None:  # Si se proporcionó un path, carga el checkpoint
            checkpoint = torch.load(init_ckpt_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            print("fine tune model on")
            print("Loaded pretrained model from checkpoint:", init_ckpt_path)
    return model


@register_model
#def icolorit_base_4ch_patch16_224_lora(pretrained=False, **kwargs):
def icolorit_base_4ch_patch16_224_lora(pretrained=False, rank=4, **kwargs):
    init_ckpt_path = kwargs.pop('init_ckpt', None)
    model = IColoriT(
        num_classes=512,
        img_size=224,
        patch_size=16,
        in_chans=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.,
        **kwargs)
    model.default_cfg = _cfg()

    print(f"Using rank = {rank}")  # Imprime el rank para confirmar

    # Opcional: Imprime los parámetros para depuración
    #for name, param in model.named_parameters():
        #print(name, param.requires_grad)

    if pretrained and init_ckpt_path is not None:
        checkpoint = torch.load(init_ckpt_path, map_location="cpu")
        # Adaptamos el checkpoint para que sea compatible con los parámetros LoRA
        adapted_state = adapt_checkpoint_for_lora(checkpoint["model"], model)
        model.load_state_dict(adapted_state)
        print("Fine tune model on")
        print("Loaded pretrained model from checkpoint:", init_ckpt_path)

    return model