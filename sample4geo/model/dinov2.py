import numpy as np
import timm
import torch
import torch.nn as nn

from .get_aggregators import get_aggregator

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

class DINOv2(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture 
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        aggregator_name (str):
        agg_config (dict): 
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """
    def __init__(
            self,
            model_name='dinov2_vitb14',
            aggregator_name='gem',
            agg_config={},
            num_trainable_blocks=4,
            norm_layer=True,
            return_token=True
        ):
        super().__init__()
        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        try:
            self.backbone = torch.hub.load('/root/.cache/torch/hub/facebookresearch_dinov2_main', model_name, trust_repo=True, source='local')
        except:
            self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        self.aggregator_name = aggregator_name
        self.aggregator = get_aggregator(agg_arch=aggregator_name, agg_config=agg_config)
        self.num_channels = DINOV2_ARCHS[model_name]
        if self.aggregator_name == 'salad':
            self.feature_dim = agg_config["token_dim"] + agg_config["cluster_dim"]*agg_config["num_clusters"]
        elif self.aggregator_name == 'netvlad':
            self.feature_dim = agg_config["output_dim"]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward_backbone(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """

        B, C, H, W = x.shape
        x = self.backbone.prepare_tokens_with_masks(x)
        
        # First blocks are frozen
        with torch.no_grad():
            for blk in self.backbone.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for blk in self.backbone.blocks[-self.num_trainable_blocks:]:
            x = blk(x)

        if self.norm_layer:
            x = self.backbone.norm(x)
        
        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)

        if self.return_token:
            return f, t
        return f

    def forward_aggregator(self, x):
        if self.aggregator_name == 'netvlad': # reorganized as BNC
            B, C, _, _ = x.shape
            x = x.reshape((B, -1, C))
        x = self.aggregator(x)
        return x

    def forward(self, x, x2=None):
        if x2 is not None:
            x = self.forward_backbone(x)
            x = self.forward_aggregator(x)
            x2 = self.forward_backbone(x2)
            x2 = self.forward_aggregator(x2)
            return x, x2
        else:
            x = self.forward_backbone(x)
            x = self.forward_aggregator(x)
            return x