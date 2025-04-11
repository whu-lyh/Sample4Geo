import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sample4geo.model.ConvNext import ConvNext
from sample4geo.model.dinov2 import DINOv2


class SaliencyCVGL(nn.Module):
    def __init__(self, config, agg_config):
        super().__init__()
        self.street_encoder = DINOv2(model_name=config.arch_name, 
                       aggregator_name=config.aggregator_name, agg_config=agg_config,
                       num_trainable_blocks=config.num_trainable_blocks, 
                       return_token=True if config.aggregator_name == 'salad' else False)
        self.satellite_encoder = DINOv2(model_name=config.arch_name, 
                       aggregator_name=config.aggregator_name, agg_config=agg_config,
                       num_trainable_blocks=config.num_trainable_blocks, 
                       return_token=True if config.aggregator_name == 'salad' else False)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.feature_dim = self.street_encoder.feature_dim

    @torch.no_grad()
    def fetch_feat_stre(self, stre_input:torch.Tensor):
        with torch.no_grad():
            stre_embeds = self.street_encoder(stre_input) # BNC
            stre_embeds = F.normalize(stre_embeds, dim=-1) # BNC # normalize before projection(aggregation) layer
        return stre_embeds

    @torch.no_grad()
    def fetch_feat_sate(self, sate_input:torch.Tensor):
        with torch.no_grad():
            sate_embeds = self.satellite_encoder(sate_input) # BNC
            sate_embeds = F.normalize(sate_embeds, dim=-1) # BNC # normalize before projection(aggregation) layer
        return sate_embeds

    def forward(self, stre_input, sate_input):
        stre_embeds = self.fetch_feat_stre(stre_input)
        sate_embeds = self.fetch_feat_sate(sate_input)
        return stre_embeds, sate_embeds

def build_aggregator_config(config):
    if config.aggregator_name == 'salad':
        agg_config={
            'num_channels': config.num_channels,
            'num_clusters': config.num_clusters,
            'cluster_dim': config.cluster_dim,
            'token_dim': config.token_dim,
        }
    elif config.aggregator_name == 'gem':
        agg_config={
            'p': 3
        }
    elif config.aggregator_name == 'netvlad':
        agg_config={
            'feature_size': config.num_channels,
            'num_clusters': config.num_clusters,
            'output_dim': config.token_dim
        }
    else:
        raise NotImplementedError(f'Sorry, <{config.aggregator_name}> aggregator is not implemented!')
    return agg_config

def get_model(config):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if config.model == 'ConvNext':
        model = ConvNext(pretrained=True, img_size=config.img_size)
        # to keep same transform
        data_config = model.get_config()
        # print(data_config)
        mean = data_config["mean"]
        std = data_config["std"]
    elif config.model == 'DINOv2':
        agg_config = build_aggregator_config(config)
        model = DINOv2(model_name=config.arch_name, 
                       aggregator_name=config.aggregator_name, agg_config=agg_config,
                       num_trainable_blocks=config.num_trainable_blocks, 
                       return_token=True if config.aggregator_name == 'salad' else False)
    elif config.model == 'SaliencyCVGL':
        agg_config = build_aggregator_config(config)
        model = SaliencyCVGL(config, agg_config)
    return model, mean, std