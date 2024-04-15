import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from torch_scatter import scatter_max, scatter_mean
from torch_scatter import segment_csr
from torch_geometric.nn.pool import voxel_grid
from utils import misc
from typing import Union
from torch import Tensor
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from typing import Union
from torch import Tensor
from .Point_PN import Point_PN_scan
from .mv_utils import PCViews
from collections import OrderedDict
import random

RESOLUTION_WIDTH = 192
RESOLUTION_HEIGHT = 304

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class AdapterSuper_noout(nn.Module):
    def __init__(self,
                 embed_dims,
                 reduction_dims,
                 drop_rate_adapter=0
                        ):
        super(AdapterSuper_noout, self).__init__()
    
        self.embed_dims = embed_dims

        # Follow towards unified
        self.super_reductuion_dim = reduction_dims
        self.dropout = nn.Dropout(p=drop_rate_adapter)

        if self.super_reductuion_dim > 0:
            self.ln1 = nn.Linear(self.embed_dims, self.super_reductuion_dim)
            self.activate = QuickGELU()
            self.ln2 = nn.Linear(self.super_reductuion_dim, self.embed_dims)
            self.init_weights()
        
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)


        self.apply(_init_weights)

    def forward(self, x, identity=None):
        out = self.ln1(x)
        out = self.activate(out)
        out = self.dropout(out)
        out = self.ln2(out)
        if identity is None:
            identity = x
        return out

def bilinear_interpolation_3d_to_2d(x, y, pos_embed):

    img_height = 304
    img_width = 192
    patch_size = 16


    grid_x = (2.0 * x / (img_width - 1)) - 1
    grid_y = (2.0 * y / (img_height - 1)) - 1
    grid = torch.stack([grid_x, grid_y], dim=-1)
    grid = grid.unsqueeze(2)

    pos_embed_reshaped = pos_embed.permute(0, 2, 1).view(1, -1, int(img_height / patch_size), int(img_width / patch_size)).repeat(grid.shape[0],1,1,1)
    pos_embed_reshaped = pos_embed_reshaped.cuda()

    interpolated_pos_embed = F.grid_sample(pos_embed_reshaped, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return interpolated_pos_embed.squeeze()

class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

## Transformers
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, adapter_dim=None,drop_rate_adapter=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
     
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.adapter = AdapterSuper_noout(embed_dims=dim, reduction_dims=adapter_dim, drop_rate_adapter=drop_rate_adapter)
        self.out_transform_3d = nn.Sequential(nn.BatchNorm1d(dim), nn.GELU())
        self.out_transform_2d = nn.ModuleList([nn.Sequential(nn.BatchNorm1d(dim), nn.GELU()) for i in range(6)])

    def forward(self, x, center1=None, idx_ptr=None, sorted_cluster_indices=None, cluster=None,grid_shape_x=None, grid_shape_y=None, mask=None, flat_grid_index=None,attn1=None, norm3=None, args=None):
        x = x + self.drop_path(self.attn(self.norm1(x)))

        x_ffn = self.drop_path(self.mlp(self.norm2(x)))
        x = x + x_ffn + args.scale_factor*self.adapter(x_ffn)

        pospara_list = []
        B,G,_ = x.shape
        cls_x = x[:,0]
        x = x[:,1:]
        G = G-1
        
        feat = x.reshape(-1,x.shape[-1])
        feat_max = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        feat_mean = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="mean")
        x_3d = self.out_transform_3d((feat_max+feat_mean))[cluster].reshape(x.shape)

        for i in range(center1.shape[1]):
            main_center = center1[:,i,:,:]

            flat_x = x.reshape(-1, x.shape[-1])
            flat_x = self.drop_path(attn1[i](norm3[i](flat_x.clone()), mask=mask[i]))+flat_x.clone()

            max_feature, max_indices = scatter_max(flat_x, flat_grid_index[i], dim=0, dim_size=main_center.shape[0] * grid_shape_x * grid_shape_y)
            mean_feature = scatter_mean(flat_x, flat_grid_index[i], dim=0, dim_size=main_center.shape[0] * grid_shape_x * grid_shape_y)

            d2_view_post_feat = self.out_transform_2d[i]((mean_feature+max_feature))[flat_grid_index[i]].view(x.shape)

            pospara_list.append(d2_view_post_feat)

        x_sup = torch.stack(pospara_list,0)
        x_sup = x_sup.transpose(0,1)

        cosine_similarities = [(F.cosine_similarity(tensor_2d, x_3d, dim=-1)+1)/2 for tensor_2d in pospara_list]
        stacked_similarities = torch.stack(cosine_similarities, dim=0).to(x_sup.device)
        stacked_similarities = stacked_similarities.transpose(0,1)
        stacked_similarities /= stacked_similarities.sum(dim=1, keepdim=True)
        x_sup_weight = torch.sum(x_sup * stacked_similarities.unsqueeze(-1), dim=1)

        x = x+args.coef_pro*x_sup_weight
        x = torch.cat((cls_x.unsqueeze(1), x), 1)
        return x

class Attention1(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mid_dim=12):
        super().__init__()
        self.num_heads = num_heads 
        self.scale = qk_scale or mid_dim ** -0.5
        self.qkv = nn.Linear(dim, mid_dim * 3, bias=qkv_bias) 
        self.attn_drop = nn.Dropout(attn_drop) #now is 0, because for grid sample, it's not necessary to drop
        self.proj = nn.Linear(mid_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) 
        self.mid_dim = mid_dim

    def forward(self, x, mask=None):
        B, N, C = x.shape[0]//128, 128, x.shape[-1]  # B, N, C are batch size, number of points, and channel dimension respectively
        #qkv = self.qkv(x).reshape(B * N, 3, self.num_heads, 18 // self.num_heads).permute(1, 0, 2, 3)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.mid_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Reshape and apply the mask
            mask = torch.where(mask, torch.tensor(-100000.0, device=mask.device), torch.tensor(0.0, device=mask.device))
            attn = attn + mask.unsqueeze(1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B * N, self.mid_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# finetune model
@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group

        self.patch_embed = Point_PN_scan(k_neighbors=config.patchknn)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        ##projection
        pc_views = PCViews(config.trans)
        self.get_pos = pc_views.get_pos
        self.patch_tokens_2d = 229 ##including the cls token
        base_ckpt = torch.load("./ckpts/imagebind_audio.pth")
        self.pos_embed_2d = base_ckpt['modality_preprocessors.audio.pos_embedding_helper.pos_embed']
        self.pos_embed_2d.requires_grad = False
        ###### trans
        self.attn1 = nn.ModuleList([Attention1(
                self.trans_dim, qkv_bias=True, proj_drop=0., mid_dim=config.attn2d_dim)for i in range(6)])
        self.norm3 = nn.ModuleList([nn.LayerNorm(self.trans_dim) for i in range(6)])

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=self.trans_dim, num_heads=self.num_heads, mlp_ratio=4., qkv_bias=True,
                drop=0., attn_drop=0., 
                drop_path = dpr[i] if isinstance(dpr, list) else dpr,adapter_dim=config.adapter_dim,drop_rate_adapter=config.drop_rate_adapter
                )
            for i in range(self.depth)])

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        self.build_loss_func()
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        self.apply(self._init_weights)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)

            for key in ckpt.keys():
                if key in ['model', 'net', 'network', 'state_dict', 'base_model']:
                    ckpt = ckpt[key]
            ckpt_state_dict = ckpt

            base_ckpt = dict()
            for k, v in ckpt_state_dict.items():
                k = k.replace("modality_preprocessors.audio.", "")
                k = k.replace("modality_trunks.audio.", "")
                if 'modality_heads.audio.0' in k:
                    k = k.replace("modality_heads.audio.0", "norm")
                if 'norm_' in k:
                    k = k.replace("norm_", "norm")
                if 'in_proj_weight' in k:
                    k = k.replace("in_proj_weight", "qkv.weight")
                if 'in_proj_bias' in k:
                    k = k.replace("in_proj_bias", "qkv.bias")
                if 'out_proj' in k:
                    k = k.replace("out_proj", "proj")
                base_ckpt[k] = v

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts, args):
        x = pts.clone().transpose(1, 2).contiguous()
        center, group_input_tokens = self.patch_embed(x,pts)  # B G N
        center_p,group_input_tokens = center,group_input_tokens.transpose(1, 2)

        pos_x,pos_y = self.get_pos(center_p, args)
        x_y = torch.stack([pos_x, pos_y], dim=-1)
        x_y = x_y.reshape(center_p.shape[0],-1,center_p.shape[1],2)
        ####save it as the image
        x_y_2d = x_y

        self.patch_pos_embed_2D = self.pos_embed_2d[:,1:]
        self.pos_embed_2d.requires_grad = False
        self.patch_pos_embed_2D.requires_grad = False

        interpolated_pos_embed = bilinear_interpolation_3d_to_2d(pos_x, pos_y, self.patch_pos_embed_2D)
        interpolated_pos_embed = interpolated_pos_embed.reshape(center_p.shape[0],-1,center_p.shape[1],self.trans_dim)
     
        interpolated_pos_embed = torch.mean(interpolated_pos_embed, dim=1)
        interpolated_pos_embed = interpolated_pos_embed.squeeze()

        batchs = list()
        for i in range(x_y_2d.shape[0]):
            batchs.append(torch.full((x_y_2d.shape[2],), i, dtype=torch.long, device=group_input_tokens.device))
        batchs = torch.cat(batchs, dim=0)
        coord = center_p.reshape(-1, 3)
        start = segment_csr(coord, torch.cat([batchs.new_zeros(1), torch.cumsum(batchs.bincount(), dim=0)]),
                    reduce="min") 
        d3_grid_size = args.coef_3dgird

        cluster = voxel_grid(pos=coord - start[batchs], size=d3_grid_size, batch=batchs, start=0)
        unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])

        mask_list = list()
        flat_grid_index_list = list()
        grid_size = args.coef_2dgird
        if  RESOLUTION_WIDTH % args.coef_2dgird==0:
            grid_shape_x = (RESOLUTION_WIDTH // grid_size)
        else:
            grid_shape_x = (RESOLUTION_WIDTH + grid_size - 1) // grid_size 
        if RESOLUTION_HEIGHT % args.coef_2dgird==0:
            grid_shape_y = (RESOLUTION_HEIGHT // grid_size)
        else:
            grid_shape_y = (RESOLUTION_HEIGHT + grid_size - 1) // grid_size 
        for i in range(x_y_2d.shape[1]):
            main_center = x_y_2d[:,i,:,:]
            x_index = (main_center[:, :, 0] / grid_size).floor().long()
            y_index = (main_center[:, :, 1] / grid_size).floor().long()
            grid_index = x_index + y_index * grid_shape_x

            batch_index = torch.arange(main_center.shape[0], device=group_input_tokens.device).view(-1, 1).repeat(1, main_center.shape[1]).view(-1)
            flat_grid_index = grid_index.view(-1) + (batch_index * grid_shape_x * grid_shape_y)

            mask = torch.zeros((group_input_tokens.shape[0], group_input_tokens.shape[1], group_input_tokens.shape[1]), dtype=bool, device=group_input_tokens.device)
            grid_index_reshaped = flat_grid_index.reshape(group_input_tokens.shape[0], group_input_tokens.shape[1])
            mask = grid_index_reshaped[:, None, :] == grid_index_reshaped[:, :, None]
            mask = ~mask
            mask_list.append(mask)
            flat_grid_index_list.append(flat_grid_index)

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        tokens = [cls_tokens, group_input_tokens]
        pos = [cls_pos, interpolated_pos_embed]

        pos = torch.cat(pos, dim=1)
        x = torch.cat(tokens, dim=1)
        # transformer
        x += pos
        for _, block in enumerate(self.blocks):
            x = block(x, center1=x_y_2d, idx_ptr=idx_ptr, sorted_cluster_indices=sorted_cluster_indices, cluster=cluster,grid_shape_x=grid_shape_x, grid_shape_y=grid_shape_y, mask=mask_list, flat_grid_index=flat_grid_index_list,attn1=self.attn1, norm3=self.norm3, args=args)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret
