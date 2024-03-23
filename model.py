import torch
from torch import nn
from layer import Conv2dCustom
#  [224, 224, 3] -> [224, 224, 64] -> [224,224,512] -> 224 * 224 * 512 -> 2
# [0.4, 0.6] -> 1 -> dogs
# Creating a CNN-based image classifier.
class ImageClassifier(nn.Module):
        def __init__(self, in_dim=3, n_class=2, h_dim=64, n_layer=1):
            super().__init__()
            
            self.conv_layer_1 = Conv2dCustom(in_dim=in_dim, out_dim=32, kernel_size=3, padding=1)
            
            self.conv_layer_2 = Conv2dCustom(in_dim=32, out_dim=h_dim, kernel_size=3, padding=1)
            
            self.conv_layer_3 = Conv2dCustom(in_dim=h_dim, out_dim=h_dim, kernel_size=3, padding=1)
            
            self.n_conv = nn.Sequential()
            for _ in range(n_layer):
                self.n_conv.append(self.conv_layer_3)
            
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=h_dim*3*3, out_features=n_class))
            self.softmax = nn.Softmax(dim=1)
            
        def forward(self, x: torch.Tensor):
            x = self.conv_layer_1(x)
            x = self.conv_layer_2(x)
            x = self.n_conv(x)
            x = self.classifier(x)
            x = self.softmax(x)
            return x


                