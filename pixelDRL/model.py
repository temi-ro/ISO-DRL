import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionModule, self).__init__()
        
        self.query_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.last_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        
        proj_query = self.query_conv(x).view(B, -1, H*W).permute(0, 2, 1) # B, N, C/2
        proj_key = self.key_conv(x).view(B, -1, H*W) # B, C/2, N
        proj_value = x.view(B, -1, H*W) # B, C, N
        
        energy = torch.bmm(proj_query, proj_key) # (B, N, N)
        attention = self.softmax(energy) # (B, N, N)
        
        out = torch.bmm(proj_value, attention) # (B, C, N)
        out = out.view(B, -1, H, W) # (B, C, H, W)
        out = self.last_conv(out) # (B, C, H, W)
        
        out = out + x
        return out

class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        return self.conv(x)

class HalfVGG16(nn.Module):
    def __init__(self, in_channels=1):
        super(HalfVGG16, self).__init__()
        
        # Block 1 (Out: 32 ch)
        self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1) 
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2 (Out: 64 ch)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3 (Out: 128 ch)
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4 (Out: 256 ch) - Paper mentions using up to conv4-3
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # No pool4 because conv4-3 is the final feature map s'(t)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1_1(x))
        c1_2 = F.relu(self.conv1_2(x)) # Save conv1-2
        x = self.pool1(c1_2)

        # Block 2
        x = F.relu(self.conv2_1(x))
        c2_2 = F.relu(self.conv2_2(x)) # Save conv2-2
        x = self.pool2(c2_2)

        # Block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        c3_3 = F.relu(self.conv3_3(x)) # Save conv3-3
        x = self.pool3(c3_3)
        
        # Block 4
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        s_prime = F.relu(self.conv4_3(x)) # Final feature s'(t)

        return s_prime, c1_2, c2_2, c3_3

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=1, n_out=2):
        super(UNet, self).__init__()
        
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(256, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # self.policy_head = nn.Conv2d(64, n_actions, kernel_size=1)
        # self.value_head = nn.Conv2d(64, 1, kernel_size=1)
        self.out_conv = nn.Conv2d(64, n_out, kernel_size=1)
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck
        b = self.bottleneck(p3)
        
        # Decoder
        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        out = self.out_conv(d1)
        return out

class PixelDRL(nn.Module):
    def __init__(self, in_channels=1, n_actions=2):
        super(PixelDRL, self).__init__()
        
        self.feature_extractor = HalfVGG16(in_channels=in_channels)

        self.sam = SelfAttentionModule(in_channels=256)
        
        cat_channels = 256 + 128 + 64 + 32  # Total: 480
        
        self.policy_net = nn.Sequential(
            DilatedConvBlock(cat_channels, 128, dilation=2),
            DilatedConvBlock(128, 64, dilation=2),
            nn.Conv2d(64, n_actions, kernel_size=1) 
        )
        
        self.value_net = nn.Sequential(
            DilatedConvBlock(cat_channels, 128, dilation=2),
            DilatedConvBlock(128, 64, dilation=2),
            nn.Conv2d(64, 1, kernel_size=1) 
        )

    def forward(self, x):
        # (H, W): Input image size (240x240)
        input_size = x.size()[2:] # (H, W)

        # s_prime: (B, 256, H/8, W/8)
        # c1_2: (B, 32, H, W)
        # c2_2: (B, 64, H/2, W/2)
        # c3_3: (B, 128, H/4, W/4)
        s_prime, c1_2, c2_2, c3_3 = self.feature_extractor(x)
        
        s_t = self.sam(s_prime) 
        
        s_t_up  = F.interpolate(s_t, size=input_size, mode='bilinear', align_corners=True)
        c3_3_up = F.interpolate(c3_3, size=input_size, mode='bilinear', align_corners=True)
        c2_2_up = F.interpolate(c2_2, size=input_size, mode='bilinear', align_corners=True)
        
        combined_features = torch.cat([s_t_up, c3_3_up, c2_2_up, c1_2], dim=1)
        
        policy_logits = self.policy_net(combined_features)
        state_value = self.value_net(combined_features)
        
        return policy_logits, state_value
