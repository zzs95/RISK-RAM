""" Class-Attention in Image Transformers (CaiT)

Paper: 'Going deeper with Image Transformers' - https://arxiv.org/abs/2103.17239

Original code and weights from https://github.com/facebookresearch/deit, copyright below

Modifications and additions for timm hacked together by / Copyright 2021, Ross Wightman
"""
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from functools import partial

import torch
import torch.nn as nn

from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, use_fused_attn
from timm.models._manipulate import checkpoint_seq
from timm.models.cait import ClassAttn, LayerScaleBlockClassAttn, LayerScaleBlock, TalkingHeadAttn
__all__ = ['Cait', ]

class Cait(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to adapt to our cait models
    def __init__(
            self,
            # img_size=224,
            # patch_size=16,
            # in_chans=3,
            num_patches=16,
            num_classes=1000,
            global_pool='token',
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            pos_drop_rate=0.,
            proj_drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            block_layers=LayerScaleBlock,
            block_layers_token=LayerScaleBlockClassAttn,
            patch_layer=PatchEmbed,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            attn_block=TalkingHeadAttn,
            mlp_block=Mlp,
            init_values=1e-4,
            attn_block_token_only=ClassAttn,
            mlp_block_token_only=Mlp,
            depth_token_only=2,
            mlp_ratio_token_only=4.0
    ):
        super().__init__()
        assert global_pool in ('', 'token', 'avg')

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.grad_checkpointing = False

        # self.patch_embed = patch_layer(
        #     img_size=img_size,
        #     patch_size=patch_size,
        #     in_chans=in_chans,
        #     embed_dim=embed_dim,
        # )

        # num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.Sequential(*[block_layers(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[i],
            norm_layer=norm_layer,
            act_layer=act_layer,
            attn_block=attn_block,
            mlp_block=mlp_block,
            init_values=init_values,
        ) for i in range(depth)])

        self.blocks_token_only = nn.ModuleList([block_layers_token(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio_token_only,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            attn_block=attn_block_token_only,
            mlp_block=mlp_block_token_only,
            init_values=init_values,
        ) for _ in range(depth_token_only)])

        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        def _matcher(name):
            if any([name.startswith(n) for n in ('cls_token', 'pos_embed', 'patch_embed')]):
                return 0
            elif name.startswith('blocks.'):
                return int(name.split('.')[1]) + 1
            elif name.startswith('blocks_token_only.'):
                # overlap token only blocks with last blocks
                to_offset = len(self.blocks) - len(self.blocks_token_only) + 1
                return int(name.split('.')[1]) + to_offset
            elif name.startswith('norm.'):
                return len(self.blocks)
            else:
                return float('inf')
        return _matcher

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'token', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, region_attn=None):
        # x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # region reweight
        if region_attn != None:
            attn = region_attn[:,:, None].repeat(1, 1, 5).reshape(x.shape[0], -1, 1) # b, 29, 5 -> b, 29*5, 1
            x = x * attn
        for i, blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x, cls_tokens)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, region_attn):
        x = self.forward_features(x, region_attn)
        x = self.forward_head(x)
        return x


if __name__ == '__main__':
    from timm.models.cait import cait_xxs24_224
    model = cait_xxs24_224()
    x = torch.randn([2,3,224,224])
    out = model(x)
    
