"""
Model architecture definitions for breast cancer segmentation (Attention U-Net)
and classification (CNN). Extracted from notebook, Colab dependencies removed.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(3, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.u1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.u2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.u3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.u4 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        b = self.bottleneck(self.pool(c4))
        x = self.u1(torch.cat([self.up1(b), c4], dim=1))
        x = self.u2(torch.cat([self.up2(x), c3], dim=1))
        x = self.u3(torch.cat([self.up3(x), c2], dim=1))
        x = self.u4(torch.cat([self.up4(x), c1], dim=1))
        return torch.sigmoid(self.out(x))


class AttentionBlock(nn.Module):
    def __init__(self, g_ch, x_ch, out_ch):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(g_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch)
        )
        self.Wx = nn.Sequential(
            nn.Conv2d(x_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(out_ch, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.Wg(g) + self.Wx(x))
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(UNet):
    def __init__(self):
        super().__init__()
        self.att1 = AttentionBlock(512, 512, 256)
        self.att2 = AttentionBlock(256, 256, 128)
        self.att3 = AttentionBlock(128, 128, 64)
        self.att4 = AttentionBlock(64, 64, 32)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        b = self.bottleneck(self.pool(c4))
        up1_b = self.up1(b)
        c4_att = self.att1(up1_b, c4)
        x = self.u1(torch.cat([up1_b, c4_att], dim=1))
        up2_x = self.up2(x)
        c3_att = self.att2(up2_x, c3)
        x = self.u2(torch.cat([up2_x, c3_att], dim=1))
        up3_x = self.up3(x)
        c2_att = self.att3(up3_x, c2)
        x = self.u3(torch.cat([up3_x, c2_att], dim=1))
        up4_x = self.up4(x)
        c1_att = self.att4(up4_x, c1)
        x = self.u4(torch.cat([up4_x, c1_att], dim=1))
        return torch.sigmoid(self.out(x))


class ClassificationCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(ClassificationCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_segmentation_model():
    return AttentionUNet()


def get_classification_model(num_classes=3):
    return ClassificationCNN(num_classes=num_classes)
