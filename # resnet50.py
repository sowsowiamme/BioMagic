# resnet50.py
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, strides=1, downsample=None):
        super().__init__()
        # 1x1 降维：不下采样
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        # 3x3：是否减半由传入的 strides 决定（与 downsample 保持一致）
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=strides, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        # 1x1 升维到 4x channels
        self.conv3 = nn.Conv2d(channels, 4 * channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(4 * channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out




class Resnet50(nn.Module):
    def __init__(self,block, layers, num_classess= 10000):
        super().__init__() 
        self.in_channels = 64
        # stem初始层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride =1, padding =3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 4个satge
        self.layer1 = self.make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layer(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layer(block, layers[3], intermediate_channels=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_classess)
        self._initialize_weights()

    def make_layer(self, block, num_residuals, intermediate_channels, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != intermediate_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, intermediate_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(intermediate_channels * block.expansion),
            )
        layers = []
        # 该 stage 的第一个 block：可能下采样 / 改通道
        layers.append(block(self.in_channels, intermediate_channels, stride, downsample))
        self.in_channels = intermediate_channels * block.expansion
        # 剩余 block：输入输出通道已对齐
        for _ in range(1, num_residuals):
            layers.append(block(self.in_channels, intermediate_channels))
        return nn.Sequential(*layers)
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        # 分类头向前传播
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x 

            


    
    


        #

            
def resnet50(num_classess=10000):
    return Resnet50(Bottleneck, [3,4,6,3], num_classess)


if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 224, 224)
    # 类别数在建模型时传入；forward 只接收输入张量
    model = resnet50(num_classess=2)
    output = model(dummy_input)
    print(output.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f} M")  # 约 23.5M（1000 类时）

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # [2, 2]

