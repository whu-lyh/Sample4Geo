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
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """
    def __init__(
            self,
            model_name='dinov2_vits14',
            aggregator_name='gem',
            num_trainable_blocks=4,
            norm_layer=True,
            return_token=False
        ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name) # poor network
        # self.backbone = timm.create_model('vit_small_path14_dinov2', pretrained=False)
        # state_dict = torch.load('/workspace/WorkSpacePR/Sample4Geo/pretrain_models/dinov2_vits14.pth', map_location='cpu')
        # self.backbone.load_state_dict(state_dict, strict=False)
        # pretrained_weight_path = '/workspace/WorkSpacePR/Sample4Geo/pretrain_models/vit_small_patch14_dinov2.bin'
        # self.backbone = timm.create_model(
        #     'vit_small_patch14_dinov2', True, num_classes=0,
        #     pretrained_cfg_overlay=dict(file=pretrained_weight_path, custom_load=False),
        # )
        agg_config={
            # 'num_channels': 384,
            # 'num_clusters': 64,
            # 'cluster_dim': 128,
            # 'token_dim': 256,
            'p': 3
        }
        self.aggregator = get_aggregator(agg_arch=aggregator_name, agg_config=agg_config)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.feature_dim = 384#agg_config["token_dim"]
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