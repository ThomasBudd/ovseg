import torch.nn as nn
try:
    from iunets import iUNet as iUNetModel
except ModuleNotFoundError:
    print('iUNet not found.')


class iUNet(nn.Module):
    """Implements the invertible U-Net.

    The keyword arguments are the same as with the iUNet library.
    (github.com/cetmann/iunets)
    """

    def __init__(self,
                 in_channels: int,
                 intermediate_channels: int,
                 out_channels: int,
                 is_2d=False,
                 **kwargs
                 ):
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.out_channels = out_channels
        self.is_2d = is_2d

        conv_op = nn.Conv2d if is_2d else nn.Conv3d

        input_layer = conv_op(
            in_channels,
            intermediate_channels,
            kernel_size=3,
            padding=1
        )

        iunet = iUNetModel(
            in_channels=intermediate_channels,
            dim=2 if is_2d else 3,
        )

        output_layer = conv_op(
            intermediate_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.model = nn.Sequential(input_layer, iunet, output_layer)

    def forward(self, xb):
        self.model(xb)