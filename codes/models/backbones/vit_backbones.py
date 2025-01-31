import timm
import torch.nn as nn
from torchvision import models as models_2d
import torch
import torch.nn.functional as F
from codes.registry import MODELS
from timm.models.helpers import load_state_dict

def resnet_50(pretrained='imagenet'):
    if pretrained=='imagenet':
        model = models_2d.resnet50(pretrained=False)
        model.load_state_dict(torch.load('./resnet50-0676ba61.pth'))
    elif pretrained=='random':
        model = models_2d.resnet50(pretrained=False)
    elif pretrained=='cholec_ssl':
        model = models_2d.resnet50(pretrained=False)
        model.load_state_dict(torch.load('/gpfsdswork/projects/rech/okw/ukw13bv/rendezvous-main/pytorch/converted_vissl_moco_r50_checkpoint.torch'), strict=True)
    else:
        raise NotImplementedError
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims

    # if pretrained=='':
    #     model = models_2d.resnet50(pretrained=False)
    #     model.load_state_dict(torch.load('/gpfsdswork/projects/rech/okw/ukw13bv/rendezvous-main/pytorch/converted_vissl_moco_r50_checkpoint.torch'), strict=True)


class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


@timm.models._registry.register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    pretrained_weight = kwargs['pretrained_weight']
    del kwargs['pretrained_weight'] 
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=12, **kwargs)
    model = timm.models.vision_transformer._create_vision_transformer('vit_small_patch16_224', **model_kwargs)
    if pretrained: # load dino weights
        state_dict = load_state_dict(pretrained_weight)
        model.load_state_dict(state_dict)
    return model


@timm.models._registry.register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    pretrained_weight = kwargs['pretrained_weight']
    del kwargs['pretrained_weight'] 
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = timm.models.vision_transformer._create_vision_transformer('vit_base_patch16_224', **model_kwargs)
    
    if pretrained: # load pretrained weights
        state_dict = load_state_dict(pretrained_weight)
        x, y = model.load_state_dict(state_dict, strict=False)
    return model

@timm.models._registry.register_model
def vit_base_patch14_dinov2(pretrained=False, **kwargs):
    """ ViT-B/14 for DINOv2
    """
    pretrained_weight = kwargs['pretrained_weight']
    del kwargs['pretrained_weight'] 
    model_args = dict(
        patch_size=14, embed_dim=768, depth=12, num_heads=12, init_values=1e-5, img_size=518,
    )
    model = timm.models.vision_transformer._create_vision_transformer(
        'vit_base_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    if pretrained: # load dino weights
        state_dict = load_state_dict(pretrained_weight)
        model.load_state_dict(state_dict)
    return model



@timm.models._registry.register_model
def vit_large_patch14_dinov2(pretrained=False, **kwargs):
    """ ViT-B/14 for DINOv2
    """
    pretrained_weight = kwargs['pretrained_weight']
    del kwargs['pretrained_weight'] 
    model_args = dict(
        patch_size=14, embed_dim=1024, depth=24, num_heads=16, init_values=1e-5, img_size=518,
    )
    model = timm.models.vision_transformer._create_vision_transformer(
        'vit_large_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    if pretrained: # load dino weights
        state_dict = load_state_dict(pretrained_weight)
        model.load_state_dict(state_dict)
    return model


@timm.models._registry.register_model
def vit_base_patch16_clip_224(pretrained: bool = False, **kwargs):
    """ ViT-B/16 CLIP image tower
  """
    pretrained_weight = kwargs['pretrained_weight']
    del kwargs['pretrained_weight'] 
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm
    )
    model = timm.models.vision_transformer._create_vision_transformer(
        'vit_base_patch16_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    if pretrained: # load dino weights
        state_dict = load_state_dict(pretrained_weight)
        model.load_state_dict(state_dict)
    return model


class ImageEncoder(nn.Module):
    def __init__(self, embedding_num, vision_model, vision_width, **kwargs):
        super(ImageEncoder, self).__init__()
        
        self.visual = vision_model

        scale = vision_width**-0.5
        self.proj = nn.Parameter(scale * torch.randn(vision_width, embedding_num))

    def forward(self, x):
        feat = self.visual(x)

        if self.proj is not None:
            feat = feat @ self.proj

        return feat


@MODELS.register_module(name='vit_encoder')
def vit_encoder(name, vision_width, pretrained, num_classes, **kwargs):
    vision_model = timm.create_model(name, pretrained=pretrained, num_classes=0, **kwargs)
    model = ImageEncoder(embedding_num=num_classes, vision_model=vision_model, vision_width=vision_width)
    return model

@MODELS.register_module(name='vit_encoder_feature_extractor')
def vit_encoder_feature_extractor(name, vision_width, pretrained, **kwargs):
    model = timm.create_model(name, pretrained=pretrained, num_classes=0, **kwargs)
    return model

def beit_encoder(**kwargs):
    pass
