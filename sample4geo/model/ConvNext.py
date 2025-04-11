import numpy as np
import timm
import torch
import torch.nn as nn


class ConvNext(nn.Module):
    def __init__(self, pretrained=True, img_size=384):
        super(ConvNext, self).__init__()
        self.img_size = img_size
        model_name = 'convnext_base.fb_in22k_ft_in1k_384' # raw sample4Geo model
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.feature_dim = self.get_feature_dim()

    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config

    def get_feature_dim(self,):
        # create a dummy input to get the feature dimension
        dummy_input = torch.zeros(1, *self.model.default_cfg['input_size'])
        with torch.no_grad():
            feature_output = self.model(dummy_input)
        return feature_output.shape[1]

    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)
        
    def forward(self, img1, img2=None):
        if img2 is not None:
            image_features1 = self.model(img1)     
            image_features2 = self.model(img2)
            return image_features1, image_features2             
        else:
            image_features = self.model(img1)
            return image_features
