import torch
import numpy as np
import torch.nn as nn


# Tuples = (Out_channels, kernel_size, stride)
# List = [BLOCK, num_repeats]
# "S" = scale prediction
# "U" = Upscaling
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

# bn_act = batch normalization activation
# kwargs = kernel_size, stride etc

class CNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act = True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.batch_norm(self.conv(x)))
        else:
            return self.conv(x)

class Residual_block(nn.Module):
    def __init__(self, channels, use_residual = True, num_repeats = 1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats) :
            self.layers += [
                nn.Sequential(
                    CNN_block(channels,channels//2, kernel_size = 1),
                    CNN_block(channels//2, channels, kernel_size = 3, padding=1))
            ]
        self.residual = use_residual
        self.num_repeats = num_repeats

    def forward(self,x):
        for layer in self.layers:
            if self.residual:
                x = layer(x) + x
            else:
                x = layer(x)
        return x


# (num_classes+5)*3 is this size because for bounding box we want [p0, x, y, w, h], p0 = prob that there is an object,
# x,y, w and h bounding box dimensions.
class Scale_prediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.prediction = nn.Sequential(
            CNN_block(in_channels, 2*in_channels, kernel_size = 3, padding =1),
            CNN_block(2*in_channels, (num_classes+5)*3, bn_act=False, kernel_size = 1),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return(
            self.prediction(x).reshape(x.shape[0], 3, self.num_classes+5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2)
        )

class YOLOv3(nn.Module):
    def __init__(self, in_channels =3, num_classes = 2):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self,x):
        outputs = []
        route_connections = []

        for layer in self.layers:
            if isinstance(layer, Scale_prediction):
                outputs.append(layer(x))
                continue

            x = layer(x)
            if isinstance(layer, Residual_block) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                #print(x.size())
                #print(route_connections[-1].size())
                
                x = torch.cat([x,route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(CNN_block(
                                        in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding = 1 if kernel_size == 3 else 0,
                                        )
                            )
                in_channels = out_channels
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(Residual_block(in_channels, num_repeats=num_repeats))
                
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        Residual_block(in_channels, use_residual = False, num_repeats=1),
                        CNN_block(in_channels, in_channels//2, kernel_size=1),
                        Scale_prediction(in_channels//2, num_classes = self.num_classes)
                    ]
                    in_channels = in_channels//2


                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3
        return  layers