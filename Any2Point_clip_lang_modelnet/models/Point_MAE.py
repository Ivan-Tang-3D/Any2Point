import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.nn import fps
from torch_scatter import scatter_max, scatter_mean
from torch_scatter import segment_csr
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from .Point_PN import Point_PN_scan
from .mv_utils import PCViews
from collections import OrderedDict
import random

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

def bilinear_interpolation_3d_to_1d(x, pos_embed):

    x_normalized = x * (torch.tensor([76.], requires_grad=False, device=x.device) / (x.max() - x.min()))
    x_left = torch.floor(x_normalized).long()
    x_right = torch.ceil(x_normalized).long()

    x_left = x_left.clamp(0, 76)
    x_right = x_right.clamp(0, 76)

    pos_embed_left = pos_embed[x_left]
    pos_embed_right = pos_embed[x_right]

    weight_left = (x_right.float() - x_normalized).unsqueeze(-1)
    weight_right = (x_normalized - x_left.float()).unsqueeze(-1)
    
    interpolated_pos_embed = weight_left * pos_embed_left + weight_right * pos_embed_right


    return interpolated_pos_embed.squeeze()

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
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,adapter_dim=None,drop_rate_adapter=None):
        super().__init__()
        self.ln_1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.ln_2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(dim, mlp_hidden_dim)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(mlp_hidden_dim, dim))
        ]))

        self.attn = nn.MultiheadAttention(dim, num_heads)

        self.adapter = AdapterSuper_noout(embed_dims=dim, reduction_dims=adapter_dim, drop_rate_adapter=drop_rate_adapter)
        self.out_transform_3d = nn.Sequential(nn.BatchNorm1d(dim), nn.GELU())
        self.out_transform_1d = nn.ModuleList([nn.Sequential(nn.BatchNorm1d(dim), nn.GELU()) for i in range(6)])

    def attention(self, x: torch.Tensor):
        return self.attn(x, x, x, need_weights=False, attn_mask=None)[0]

    def forward(self, x, center1=None, idx_ptr=None, sorted_cluster_indices=None, cluster=None,grid_shape=None, mask=None, flat_grid_index=None,attn1=None, norm3=None, args=None):
        x = x + self.attention(self.ln_1(x))

        x_ffn = self.mlp(self.ln_2(x))
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
            main_center = center1[:,i,:]
            flat_x = x.reshape(-1, x.shape[-1])
            flat_x = attn1[i](norm3[i](flat_x.clone()), mask=mask[i])+flat_x.clone()
            if args.maxmean_1d:
                max_feature, max_indices = scatter_max(flat_x, flat_grid_index[i], dim=0, dim_size=main_center.shape[0] * grid_shape)
                mean_feature = scatter_mean(flat_x, flat_grid_index[i], dim=0, dim_size=main_center.shape[0] * grid_shape)

                d2_view_post_feat = self.out_transform_1d[i]((mean_feature+max_feature))[flat_grid_index[i]].view(x.shape)
            else:
                mean_feature = scatter_mean(flat_x, flat_grid_index[i], dim=0, dim_size=main_center.shape[0] * grid_shape)

                d2_view_post_feat = self.out_transform_1d[i]((mean_feature))[flat_grid_index[i]].view(x.shape)
            pospara_list.append(d2_view_post_feat)
        x_sup = torch.stack(pospara_list,0)
        x_sup = x_sup.transpose(0,1)

        cosine_similarities = [(F.cosine_similarity(tensor_1d, x_3d, dim=-1)+1)/2 for tensor_1d in pospara_list]
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
        pc_views = PCViews()
        self.get_pos = pc_views.get_pos
        base_ckpt = torch.load("./ckpts/ViT-L-14.pt")
        self.pos_embed_1d = base_ckpt.state_dict()['positional_embedding']
        self.pos_embed_1d.requires_grad = False
        ######

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]

        self.resblocks = nn.ModuleList([
            Block(
                dim=self.trans_dim, num_heads=self.num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop=0., attn_drop=0., 
                drop_path = dpr[i] if isinstance(dpr, list) else dpr,adapter_dim=config.adapter_dim,drop_rate_adapter=config.drop_rate_adapter
                )
            for i in range(self.depth)])
        if config.attn1d_dim!=0:
            self.attn1 = nn.ModuleList([Attention1(
                self.trans_dim, qkv_bias=True, proj_drop=0., mid_dim=config.attn1d_dim)for i in range(6)])
            self.norm3 = nn.ModuleList([nn.LayerNorm(self.trans_dim) for i in range(6)])

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
            ckpt = ckpt.state_dict()

            for key in ckpt.keys():
                if key in ['model', 'net', 'network', 'state_dict', 'base_model']:
                    ckpt = ckpt[key]
            ckpt_state_dict = ckpt

            base_ckpt = dict()
            #clip
            for k, v in ckpt_state_dict.items():
                if 'visual' not in k:
                    if 'positional_embedding' in k:
                        base_ckpt[k] = v
                    elif 'transformer' in k:
                        k = k.replace("transformer.", "")
                        base_ckpt[k] = v
                    elif 'ln_final' in k:
                        k = k.replace("ln_final", "norm")
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

        pos_x,aug_points,rot_mat, translation = self.get_pos(center_p, args)
        pos_x = pos_x.reshape(center_p.shape[0],-1,center_p.shape[1])

        interpolated_pos_embed = bilinear_interpolation_3d_to_1d(pos_x, self.pos_embed_1d)
        interpolated_pos_embed = interpolated_pos_embed.reshape(center_p.shape[0],-1,center_p.shape[1],self.trans_dim)

        interpolated_pos_embed = torch.mean(interpolated_pos_embed, dim=1)
        interpolated_pos_embed = interpolated_pos_embed.squeeze()

        batchs = list()
        for i in range(pos_x.shape[0]):
            batchs.append(torch.full((pos_x.shape[2],), i, dtype=torch.long, device=group_input_tokens.device))
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
        if 77 % args.coef_1dgird==0:
            grid_size = args.coef_1dgird
            grid_shape = (77 // grid_size)
        else:
            grid_size = args.coef_1dgird
            grid_shape = (77 + grid_size - 1) // grid_size 
        for i in range(pos_x.shape[1]):
            main_center = pos_x[:,i,:]
            grid_index = (main_center / grid_size).floor().long()

            batch_index = torch.arange(main_center.shape[0], device=main_center.device).view(-1, 1) * grid_shape
            flat_grid_index = grid_index + batch_index.expand_as(grid_index)
            flat_grid_index = flat_grid_index.view(-1)

            mask = torch.zeros((group_input_tokens.shape[0], group_input_tokens.shape[1],group_input_tokens.shape[1]), dtype=bool, device=group_input_tokens.device)
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
        for _, block in enumerate(self.resblocks):
            x = block(x, center1=pos_x, idx_ptr=idx_ptr, sorted_cluster_indices=sorted_cluster_indices, cluster=cluster,grid_shape=grid_shape, mask=mask_list, flat_grid_index=flat_grid_index_list,attn1=self.attn1, norm3=self.norm3, args=args)
        x = self.norm(x)
       
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1) 
        ret = self.cls_head_finetune(concat_f)
        return ret
