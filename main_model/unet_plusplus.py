import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()
        
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
            
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        
        nb_filter = [64, 128, 256, 512]

        self.conv0_0 = block(3, nb_filter[0])
        self.conv1_0 = block(nb_filter[0], nb_filter[1])
        self.conv2_0 = block(nb_filter[1], nb_filter[2])
        self.conv3_0 = block(nb_filter[2], nb_filter[3])

        self.conv0_1 = block(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = block(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = block(nb_filter[2]+nb_filter[3], nb_filter[2])

        self.conv0_2 = block(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = block(nb_filter[1]*2+nb_filter[2], nb_filter[1])

        self.conv0_3 = block(nb_filter[0]*3+nb_filter[1], nb_filter[0])

        self.out = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        return self.out(x0_3)
