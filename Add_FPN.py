import torch.nn as nn
from typing import  Dict, Optional, List
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool, ExtraFPNBlock
from torch import nn, Tensor
from torchvision.models._utils import IntermediateLayerGetter






class Effnet_BackboneWithFPN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        return_layers: Dict[str, str],
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
    ) -> None:
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()
        
        self.backbone = backbone
        self.back = self.backbone.blocks
        

        self.body = IntermediateLayerGetter(self.back, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels
        
        

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x) 
        x = self.body(x)
        x = self.fpn(x)
        return x