
import torch.nn as nn
from torchvision import models as models_2d
import clip
import torch

################################################################################
# ResNet Family
################################################################################

class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x

def resnet_18(pretrained=True):
    model = models_2d.resnet18(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_34(pretrained=True):
    model = models_2d.resnet34(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


def resnet_50(pretrained='imagenet'):
    if pretrained=='imagenet':
        model = models_2d.resnet50(pretrained=True)
    elif pretrained=='random':
        model = models_2d.resnet50(pretrained=False)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024

def resnet_50_CLIP(pretrained):
    model, preprocess = clip.load("RN50", device='cpu')
    visual = model.visual
    return visual, 1024, 1024

def resnet_101(pretrained=True):
    model = models_2d.resnet101(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, 1024


resnet_dict = {
    'resnet_18': resnet_18,
    'resnet_34': resnet_34,
    'resnet_50': resnet_50,
    'resnet_50_clip': resnet_50_CLIP,
    'resnet_101': resnet_101,
}

################################################################################
# DenseNet Family
################################################################################


def densenet_121(pretrained=True):
    model = models_2d.densenet121(pretrained=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims, None


def densenet_161(pretrained=True):
    model = models_2d.densenet161(pretrained=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims, None


def densenet_169(pretrained=True):
    model = models_2d.densenet169(pretrained=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims, None


################################################################################
# ResNextNet Family
################################################################################


def resnext_50(pretrained=True):
    model = models_2d.resnext50_32x4d(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, None


def resnext_100(pretrained=True):
    model = models_2d.resnext101_32x8d(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims, None
