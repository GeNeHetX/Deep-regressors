import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """A block for the U-Net architecture consisting of two convolutional layers.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        """Forward pass through the UNetBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels, H, W).
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        """U-Net architecture for image segmentation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int, optional): Number of output channels. Defaults to 1.
        """
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = UNetBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = UNetBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = UNetBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = UNetBlock(512, 1024)
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = UNetBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = UNetBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = UNetBlock(128, 64)
        # Output
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        """Forward pass through the U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels, H, W).
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.out_conv(d1)
        return out.squeeze(1)  # shape: (batch, H, W)

# Example usage (optional, for testing this file directly)
if __name__ == '__main__':
    # Example: 2 input features, 1 output value
    model = UNet(in_channels=32, out_channels=1)
    print(model)