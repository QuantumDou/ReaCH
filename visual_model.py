from open_clip import create_model_from_pretrained
import torch
import torch.nn as nn
from open_clip.model import CLIP, CustomTextCLIP
import json
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F
import random
import numpy as np

class CLIPConfig:
    cast_dtype: str = 'fp16'
    config_path: str = 'checkpoint/BiomedCLIP/open_clip_config.json'
    def __init__(self):

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.n_ca_heads = 12
        self.ca_dropout = 0.1
        self.d_input = 768
        self.method = 'biomed'
        self.pretrained_cfg = config['preprocess_cfg']
        self.model_cfg = config['model_cfg']

class CLIPEncoder(nn.Module):
    def __init__(self, config,device):
        super(CLIPEncoder, self).__init__()
        self.config = config
        self.img_encoder = CustomTextCLIP(**config.model_cfg, cast_dtype=config.cast_dtype)
        checkpoint = torch.load('checkpoint/BiomedCLIP/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.pth')
        self.img_encoder.load_state_dict(checkpoint, strict=False)
        self.img_encoder.to(device)

            
    def forward(self, images):
        with torch.no_grad():
            image_features = self.img_encoder.encode_image(images)
        return image_features[:,1:,:].contiguous()
    




        
