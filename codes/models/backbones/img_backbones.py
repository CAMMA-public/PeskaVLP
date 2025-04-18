import torch.nn as nn
from codes.models.backbones.resnets import resnet_dict
from codes.registry import MODELS
import torch
from codes.models.backbones.resnets_openai import ModifiedResNet, AttentionPool2d

@MODELS.register_module(name='img_backbones/ImageEncoder')
class ImageEncoder(nn.Module):
    def __init__(self, num_classes, pretrained, backbone_name, img_norm):
        super(ImageEncoder, self).__init__()
        
        self.model, self.feature_dim, self.interm_feature_dim = resnet_dict[backbone_name](pretrained=pretrained)

        self.global_embedder = nn.Linear(self.feature_dim, num_classes)

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.norm = img_norm


    def forward(self, x, get_local=False):
        # --> fixed-size input: batch x 3 x 299 x 299
        global_ft, local_ft = self.resnet_forward(x, extract_features=True)
        global_emb = self.global_embedder(global_ft)

        if self.norm is True:
            global_emb = global_emb / torch.norm(
                global_emb, 2, dim=1, keepdim=True
            ).expand_as(global_emb)

        return global_emb

    def resnet_forward(self, x, extract_features=False):

        # --> fixed-size input: batch x 3 x 299 x 299
        # x = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)(x)

        x = self.model.conv1(x)  # (batch_size, 64, 150, 150)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)  # (batch_size, 64, 75, 75)
        x = self.model.layer2(x)  # (batch_size, 128, 38, 38)
        x = self.model.layer3(x)  # (batch_size, 256, 19, 19)
        local_features = x
        x = self.model.layer4(x)  # (batch_size, 512, 10, 10)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x, local_features

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)



@MODELS.register_module(name='img_backbones/ImageEncoder_attnpool')
class ImageEncoder_attnpool(nn.Module):
    def __init__(self, num_classes, pretrained, backbone_name, img_norm, input_resolution, heads, width=64):
        super(ImageEncoder_attnpool, self).__init__()
        
        self.model, self.feature_dim, self.interm_feature_dim = resnet_dict[backbone_name](pretrained=pretrained)

        self.global_embedder = nn.Linear(self.feature_dim, num_classes)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, num_classes)
        
        self.norm = img_norm


    def forward(self, x, get_local=False):
        # --> fixed-size input: batch x 3 x 299 x 299
        global_emb, local_ft = self.resnet_forward(x, extract_features=True)

        if self.norm is True:
            global_emb = global_emb / torch.norm(
                global_emb, 2, dim=1, keepdim=True
            ).expand_as(global_emb)

        return global_emb

    def resnet_forward(self, x, extract_features=False):

        # --> fixed-size input: batch x 3 x 299 x 299
        # x = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)(x)

        x = self.model.conv1(x)  # (batch_size, 64, 150, 150)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)  # (batch_size, 64, 75, 75)
        x = self.model.layer2(x)  # (batch_size, 128, 38, 38)
        x = self.model.layer3(x)  # (batch_size, 256, 19, 19)
        local_features = x
        x = self.model.layer4(x)  # (batch_size, 512, 10, 10)

        x = self.attnpool(x)
        x = x.view(x.size(0), -1)

        return x, local_features

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)



@MODELS.register_module(name='img_backbones/ImageEncoder_feature_extractor')
class ImageEncoder_feature_extractor(nn.Module):
    def __init__(self, num_classes, pretrained, backbone_name, img_norm):
        super(ImageEncoder_feature_extractor, self).__init__()
        
        self.model, self.feature_dim, self.interm_feature_dim = resnet_dict[backbone_name](pretrained=pretrained)

        self.global_embedder = nn.Linear(self.feature_dim, num_classes)

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.norm = img_norm


    def forward(self, x, get_local=False):
        # --> fixed-size input: batch x 3 x 299 x 299
        global_ft, local_ft = self.resnet_forward(x, extract_features=True)

        return global_ft

    def resnet_forward(self, x, extract_features=False):

        # --> fixed-size input: batch x 3 x 299 x 299
        # x = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)(x)

        x = self.model.conv1(x)  # (batch_size, 64, 150, 150)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)  # (batch_size, 64, 75, 75)
        x = self.model.layer2(x)  # (batch_size, 128, 38, 38)
        x = self.model.layer3(x)  # (batch_size, 256, 19, 19)
        local_features = x
        x = self.model.layer4(x)  # (batch_size, 512, 10, 10)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x, local_features

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)


@MODELS.register_module(name='img_backbones/ImageEncoder_CLIP')
class ImageEncoder_CLIP(nn.Module):
    def __init__(self, num_classes, img_norm, pretrained, layers, heads, input_resolution=224, width=64):
        super(ImageEncoder_CLIP, self).__init__()
        
        self.model = ModifiedResNet(
            layers=layers,
            output_dim=num_classes,
            heads=heads,
            input_resolution=input_resolution,
            width=width
        )
        
        self.norm = img_norm

        if pretrained != '':
            self.model.load_state_dict(torch.load(pretrained), strict=True)


    def forward(self, x, get_local=False):
        global_emb = self.model(x)

        if self.norm is True:
            global_emb = global_emb / torch.norm(
                global_emb, 2, dim=1, keepdim=True
            ).expand_as(global_emb)
            
        return global_emb