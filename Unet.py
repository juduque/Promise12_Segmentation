import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.Conv3D_Block = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.Conv3D_Block(x)
        out = self.pool(x)
        return x, out


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=2, stride=2):
        super(Up, self).__init__()
        self.Deconv3D_Block = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel, stride=stride)

    def forward(self, x):
        x = self.Deconv3D_Block(x)  # up
        return x


class Unet3d(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=[32, 64, 128, 256, 512], has_dropout=False):
        super(Unet3d, self).__init__()

        # Down part of UNet
        self.down1 = Down(in_channels, features[0])
        self.down2 = Down(features[0], features[1])
        self.down3 = Down(features[1], features[2])
        self.down4 = Down(features[2], features[3])

        self.baseu = DoubleConv(features[3], features[4])

        # Up part of UNet
        self.up1 = Up(features[4], features[3])
        self.up2 = Up(features[3], features[2])
        self.up3 = Up(features[2], features[1])
        self.up4 = Up(features[1], features[0])

        self.up1_dec = DoubleConv(features[4], features[3])
        self.up2_dec = DoubleConv(features[3], features[2])
        self.up3_dec = DoubleConv(features[2], features[1])
        self.up4_dec = DoubleConv(features[1], features[0])

        # Final Image
        self.conv_class = nn.Conv3d(features[0], in_channels, kernel_size=1, stride=1, padding=0, bias=True)

        # Dropout
        self.has_dropout = has_dropout
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Down part of UNet
        conv1, x = self.down1(x)
        conv2, x = self.down2(x)
        conv3, x = self.down3(x)
        conv4, x = self.down4(x)

        # Base U
        x = self.baseu(x)

        # Up part of UNet
        x = torch.cat([self.up1(x), conv4], dim=1)
        x = self.up1_dec(x)

        x = torch.cat([self.up2(x), conv3], dim=1)
        x = self.up2_dec(x)

        x = torch.cat([self.up3(x), conv2], dim=1)
        x = self.up3_dec(x)

        x = torch.cat([self.up4(x), conv1], dim=1)
        x = self.up4_dec(x)

        # dropout
        if self.has_dropout:
            x = self.dropout(x)
        out = self.sigmoid(self.conv_class(x))

        return out


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand((1, 1, 128, 128, 64)).to(device)
    model = Unet3d().to(device)
    result = model(x)
    print(result.shape)
    print(x.shape)


if __name__ == "__main__":
    test()
