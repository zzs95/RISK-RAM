import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from dataset.constants import COVID_REGIONS, REPORT_KEYS, REGION_IDS
from models.losses_surv import CoxPHLoss
import torchtuples as tt
from models.MLP import MLPVanilla
from models.encoders.image.resnet import ResNet, AttentionPool2d
from models.encoders.image.cait import Cait
class MLPHead(nn.Module):
    def __init__(self, in_channels, representation_size, out_relu=True):
        super().__init__()
        self.out_relu = out_relu
        self.fc5 = nn.Linear(in_channels, representation_size)
        self.fc6 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        if self.out_relu:
            x = F.relu(x)
        return x   


class RegionSPModel(nn.Module):
    """
    Full model consisting of:
        - object detector encoder
        - binary classifier for selecting regions for sentence genneration
        - binary classifier for detecting if a region is abnormal or normal (to encode this information in the region feature vectors)
        - language model decoder
    """

    def __init__(self):
        super().__init__()
        self.embed_dim = 1024
        self.representation_size = 1024
        self.global_resolution = 16
        self.img_size = 224
        self.risk_embed_dim = 2048
        self.img_encoder = ResNet(name='resnet50', in_channels=1, pretrained=True, pool_method='mean')
        self.img_encoder = ResNet(name='resnet50', in_channels=1, pretrained=False, pool_method='mean')
        # CHECKPOINT = './checkpoints/prior_resnet50.pt'
        # checkpoint = torch.load(CHECKPOINT, map_location=torch.device("cpu"))
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()        
        # for k, v in checkpoint.items():
        #     new_state_dict['encoder.'+k] = v
        # self.img_encoder.load_state_dict(new_state_dict)
        
        # for k,v in self.img_encoder.named_parameters():
        #     if k.replace('encoder.', '') in checkpoint.keys():
        #         v.requires_grad = False        
        
        # self.clin_risk_feat_encoder = MLPVanilla(in_features=16, num_nodes=[self.risk_embed_dim,self.risk_embed_dim], out_features=None,
        #                     batch_norm=True, dropout=0.1, output_bias=False, output_activation=None)
        # self.clin_fuse_projection = MLPHead(self.risk_embed_dim*2, self.risk_embed_dim, out_relu=False)
        
        
        # self.global_attention_pooler = AttentionPool2d(in_features=2048, feat_size=7, embed_dim=2048, num_heads=8)  
        self.risk_predictor = nn.Sequential(nn.Linear(self.risk_embed_dim*1, 1), torch.nn.Sigmoid())   
        self.lambda_surv = 0.1        
        self.criterion_surv = CoxPHLoss()     
                    
        # # CaiT
        # region embed
        # self.roi_embed_dim = 768
        # self.roi_embedder = nn.ModuleList()
        # self.roi_resolution_list = [4,2,2,1,1]
        # self.roi_channel_list = [64, 256, 512, 1024, 2048]
        # for i in range(len(self.roi_channel_list)):
        #     in_feat_len = self.roi_channel_list[i] * self.roi_resolution_list[i]**2
        #     self.roi_embedder.append(MLPHead(in_feat_len, self.roi_embed_dim))
        
        # self.region_attn_layer = Cait(num_patches=29*5, num_classes=1, embed_dim=self.roi_embed_dim,
        #                         depth=2, num_heads=12, depth_token_only=2,
        #                         drop_rate=0., pos_drop_rate=0., proj_drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,)
        
        # self.region_attention_projection = nn.Sequential(MLPHead(2048, 29, out_relu=False), nn.LayerNorm(normalized_shape=29))
   
    
    def region_encode(self, batch, multi_scale_feats, device):
        # roi_align
        batch_size = len(batch["boxes"])
        batch['boxes'] = [batch['boxes'][i].to(device, non_blocking=True) for i in range(batch_size)]
        multi_scale_embed = []
        for i, feat in enumerate(multi_scale_feats):
            scale = feat.shape[-1] / self.img_size
            output_size = self.roi_resolution_list[i]
            multi_scale_embed.append(self.roi_embedder[i](
                torchvision.ops.roi_align(input=feat, boxes=batch['boxes'], output_size=output_size, spatial_scale=scale, aligned=False).squeeze()).reshape(batch_size, 29, 1, -1))
        region_ms_feats = torch.concat(multi_scale_embed, dim=2)  # b, 29, 5, 192 
        region_ms_feats = region_ms_feats.reshape(batch_size, -1, self.roi_embed_dim)# b, 29*5, 192 000011112222....
        return region_ms_feats


    def forward(self, run_params, batch, device, return_score=False):
        batch_size = batch["clin_feat"].size(0)
        images = batch["images"].to(device, non_blocking=True)
        # with torch.no_grad():
        local_feats, global_feats, multi_scale_feats = self.img_encoder.encoder(images, return_features=True)
        
        # region SP
        # region_ms_feats = self.region_encode(batch, multi_scale_feats, device)
        # x = self.region_attn_layer(region_ms_feats, None)
        # risk_score = torch.sigmoid(x)
        
        # region_attention SP
        # region_ms_feats = self.region_encode(batch, multi_scale_feats, device)
        # region_attn = self.region_attention_projection(global_feats)
        # x = self.region_attn_layer(region_ms_feats, region_attn)
        # risk_score = torch.sigmoid(x)
        
        # global atten SP
        # global_feats = self.global_attention_pooler(local_feats)
        # risk_score = self.risk_predictor(global_feats.reshape(batch_size, -1))
        
        # base SP
        risk_score = self.risk_predictor(global_feats.reshape(batch_size, -1))
        
        
        if return_score:
            return risk_score
        risk_score = tt.tuplefy(risk_score.float())
        time_label = batch["time_label"].float().to(device, non_blocking=True)
        event_label = batch["event_label"].long().to(device, non_blocking=True)
        surv_label = (time_label, event_label)
        surv_label = tt.tuplefy(surv_label).to_device(device)
        loss_surv = self.criterion_surv(*risk_score, *surv_label)
        
        loss_dict = {}
        loss_dict['loss'] = 0
        loss_dict['surv_loss'] = loss_surv 
        loss_dict['loss'] += loss_surv * self.lambda_surv
        return loss_dict

    @torch.no_grad()
    def risk_predict(self, run_params, batch, device):
        risk_score = self.forward(run_params, batch, device, return_score=True)
        return risk_score

    def risk_predict_grad(self, batch, device='cuda'):
        batch_size = batch["clin_feat"].size(0)
        images = batch["images"].to(device, non_blocking=True)
        local_feats, global_feats, multi_scale_feats = self.img_encoder.encoder(images, return_features=True)

        # base SP
        risk_score = self.risk_predictor(global_feats.reshape(batch_size, -1))
        return risk_score        