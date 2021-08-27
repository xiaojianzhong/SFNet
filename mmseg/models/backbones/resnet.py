from mmseg.models.builder import BACKBONES
from mmseg.models.backbones import ResNetV1c


@BACKBONES.register_module()
class ResNetV1cc(ResNetV1c):
    def __init__(self, **kwargs):
        super(ResNetV1cc, self).__init__(**kwargs)
        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name)
            for block in res_layer:
                block.conv1.padding, block.conv2.padding = block.conv2.padding, block.conv1.padding
                block.conv1.dilation, block.conv2.dilation = block.conv2.dilation, block.conv1.dilation
