""" Vision Transformer (ViT) for Point Cloud Understanding in PyTorch
Hacked together by / Copyright 2020, Ross Wightman
Modified to 3D application by / Copyright 2022@Pix4Point team
"""
import logging
from typing import List
import torch
import torch.nn as nn
from ..layers import create_norm, create_linearblock, create_convblock1d, three_interpolation, \
    furthest_point_sample, random_sample
from ..layers.attention import Block
import numpy as np
from ..peft_module.adapter import AdapterSuper
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import scatter_max, scatter_mean
from torch_scatter import segment_csr
import torch.nn.functional as F
import os
from ..peft_module.mv_utils import PCViews
from ..build import MODELS, build_model_from_cfg
from pointnet2_ops import pointnet2_utils
from .Point_PN import Point_PN_scan

RESOLUTION_WIDTH = 192
RESOLUTION_HEIGHT = 304

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data, fps_idx

def fps_2d(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    return fps_idx.long()

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

import torch

def bilinear_interpolation_3d_to_2d(x, y, pos_embed,args):
    # 图像大小和补丁大小
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

@MODELS.register_module()
class PointViT(nn.Module):
    """ Point Vision Transformer ++: with early convolutions
    """
    def __init__(self,
                 in_channels=3,
                 embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4., qkv_bias=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_args={'norm': 'ln', 'eps': 1.0e-6},
                 act_args={'act': 'gelu'},
                 add_pos_each_block=False,
                 global_feat='cls,max',
                 distill=False, 
                 adapter_args={'adapter_dim': 16, 
                             'adapter_drop_path_rate': 0.1
                             }, 
                 attn2d_dim=0,
                 patchknn=64,
                 trans=0.,
                 **kwargs
                 ):
        """
        Args:
            in_channels (int): number of input channels. Default: 6. (p + rgb)
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = Point_PN_scan(k_neighbors=patchknn)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        if self.patch_embed.out_channels != self.embed_dim: 
            self.proj = nn.Linear(self.patch_embed.out_channels, self.embed_dim)
        else:
            self.proj = nn.Identity() 
        self.add_pos_each_block = add_pos_each_block
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.depth = depth
        ##projection
        pc_views = PCViews(trans=trans)
        self.get_pos = pc_views.get_pos
        self.patch_tokens_2d = 229 ##including the cls token
        base_ckpt = torch.load("./ckpts/imagebind_audio.pth")

        self.pos_embed_2d = base_ckpt['modality_preprocessors.audio.pos_embedding_helper.pos_embed']
        self.pos_embed_2d.requires_grad = False
        ######
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_args=norm_args, act_args=act_args, adapter_dim=adapter_args['adapter_dim'], drop_rate_adapter=adapter_args['adapter_drop_path_rate']
            )
            for i in range(depth)])
        self.norm = create_norm(norm_args, self.embed_dim)  # Norm layer is extremely important here!
        self.global_feat = global_feat.split(',')
        self.out_channels = len(self.global_feat)*embed_dim
        self.distill_channels = embed_dim
        self.k_neighbors = 16
        if attn2d_dim!=0:
            self.attn1 = nn.ModuleList([Attention1(
                embed_dim, qkv_bias=qkv_bias, proj_drop=drop_rate, mid_dim=attn2d_dim)for i in range(6)])
            self.norm3 = nn.ModuleList([nn.LayerNorm(embed_dim) for i in range(6)])
        self.attn2d_dim = attn2d_dim
        # distill
        if distill:
            self.dist_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            self.dist_pos = nn.Parameter(torch.randn(1, 1, self.embed_dim))
            self.n_tokens = 2
        else:
            self.dist_token = None
            self.n_tokens = 1
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.cls_pos, std=.02)
        if self.dist_token is not None:
            torch.nn.init.normal_(self.dist_token, std=.02)
            torch.nn.init.normal_(self.dist_pos, std=.02)
        self.apply(self._init_weights)
    
    def mv_proj(self, pc):
        img = self.get_img(pc).cuda()
        img = img.unsqueeze(1).repeat(1, 3, 1, 1)
        return img

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', 'dist_token', 'dist_token'}

    def get_num_layers(self):
        return self.depth

    def forward(self, p, x=None, args=None):
        if hasattr(p, 'keys'): 
            p, x = p['pos'], p['x'] if 'x' in p.keys() else None
        if x is None:
            x = p.clone().transpose(1, 2).contiguous()
    
        center, group_input_tokens, center_idx, neighbor_idx, post_center = self.patch_embed(x,p)
        center_p, x = center, self.proj(group_input_tokens.transpose(1, 2))
        ###projection
        self.patch_pos_embed_2D = self.pos_embed_2d[:,1:]
        self.pos_embed_2d.requires_grad = False
        self.patch_pos_embed_2D.requires_grad = False

        pos_x,pos_y = self.get_pos(center_p, args)
        x_y = torch.stack([pos_x, pos_y], dim=-1)
        x_y = x_y.reshape(center_p.shape[0],-1,center_p.shape[1],2)
        
        x_y_2d = x_y
        zeros = torch.zeros(*x_y.shape[:-1], 1, device=x_y.device)
        x_y = torch.cat([x_y, zeros], dim=-1)
    
        interpolated_pos_embed = bilinear_interpolation_3d_to_2d(pos_x, pos_y, self.patch_pos_embed_2D, args)
        interpolated_pos_embed = interpolated_pos_embed.reshape(center_p.shape[0],-1,center_p.shape[1],self.embed_dim)
        interpolated_pos_embed_raw = interpolated_pos_embed.clone()

        interpolated_pos_embed = torch.mean(interpolated_pos_embed, dim=1)
        interpolated_pos_embed = interpolated_pos_embed.squeeze()

        pos_embed = [self.cls_pos.expand(x.shape[0], -1, -1), interpolated_pos_embed]
        tokens = [self.cls_token.expand(x.shape[0], -1, -1), x]
        if self.dist_token is not None:
            pos_embed.insert(1, self.dist_pos.expand(x.shape[0], -1, -1)) 
            tokens.insert(1, self.dist_token.expand(x.shape[0], -1, -1)) 
        
        batchs = list()
        for i in range(x_y_2d.shape[0]):
            batchs.append(torch.full((x_y_2d.shape[2],), i, dtype=torch.long, device=x.device))
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

            batch_index = torch.arange(main_center.shape[0], device=x.device).view(-1, 1).repeat(1, main_center.shape[1]).view(-1)
            flat_grid_index = grid_index.view(-1) + (batch_index * grid_shape_x * grid_shape_y)

            mask = torch.zeros((x.shape[0], x.shape[1], x.shape[1]), dtype=bool, device=x.device)
            grid_index_reshaped = flat_grid_index.reshape(x.shape[0], x.shape[1])
            mask = grid_index_reshaped[:, None, :] == grid_index_reshaped[:, :, None]
            mask = ~mask
            mask_list.append(mask)
            flat_grid_index_list.append(flat_grid_index)

            
        pos_embed = torch.cat(pos_embed, dim=1)
        x = torch.cat(tokens, dim=1)
        if self.add_pos_each_block:
            for block in self.blocks:
                if self.attn3d_dim!=0:
                    x, x_3d = block(x + pos_embed)
                else:
                    x, x_3d = block(x + pos_embed)
        else:
            x = self.pos_drop(x + pos_embed)
            for block in self.blocks:
                x = block(x, center_idx=center_idx,neighbors_idx=neighbor_idx, center1=x_y_2d, center2=center_p, center3=post_center,args=args,attn1=self.attn1, norm3=self.norm3,idx_ptr=idx_ptr, sorted_cluster_indices=sorted_cluster_indices, cluster=cluster,grid_shape_x=grid_shape_x, grid_shape_y=grid_shape_y, mask=mask_list, flat_grid_index=flat_grid_index_list)
        x = self.norm(x)
        return None, None, x

    def forward_cls_feat(self, p, x=None, args=None):  # p: p, x: features
   
        _, _, x = self.forward(p, x, args=args)
        token_features = x[:, self.n_tokens:, :]
        cls_feats = []
        for token_type in self.global_feat:
            if 'cls' in token_type:
                cls_feats.append(x[:, 0, :])
            elif 'max' in token_type:
                cls_feats.append(torch.max(token_features, dim=1, keepdim=False)[0])
            elif token_type in ['avg', 'mean']:
                cls_feats.append(torch.mean(token_features, dim=1, keepdim=False))
        global_features = torch.cat(cls_feats, dim=1)
        
        if self.dist_token is not None and self.training:
            return global_features, x[:, 1, :]
        else: 
            return global_features

    def forward_seg_feat(self, p, x=None):  # p: p, x: features
        p_list, x_list, x = self.forward(p, x)
        x_list[-1] = x.transpose(1, 2)
        return p_list, x_list