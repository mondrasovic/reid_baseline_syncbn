import torch.nn as nn
import torchvision.models as models

from .baseline import Baseline

from .sync_bn import convert_model


def build_model(cfg, num_classes):
    if cfg.MODEL.NAME != 'resnet50':
        raise RuntimeError("unsupported backbone model")
    
    backbone = models.resnet50(pretrained=cfg.MODEL.PRETRAINED)
    # Remove the last FC classification layer.
    backbone_shrunk = nn.Sequential(*list(backbone.children())[:-1])

    model = Baseline(num_classes, backbone_shrunk)

    return model
