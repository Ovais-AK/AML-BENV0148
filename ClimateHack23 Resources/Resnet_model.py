import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

layers = [4,4,4,4]
def conv_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_block(in_channels, out_channels, stride=stride)
        self.conv2 = conv_block(out_channels, out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return F.relu(out, inplace=False)

class ResNetModel(nn.Module):
    
    def __init__(self, block, layers, predict_length=48, pv=None,pv_inc=None, use_pv_inc=None):
        
        super(ResNetModel, self).__init__()
        self.in_channels = 12 #reduce the stride
        self.initial = nn.Identity()
        #self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.layer1 = self._make_layer(block, 12, layers[0])
        self.layer2 = self._make_layer(block, 24, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 48, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 96, layers[3], stride=1)
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        # Adjust this linear layer based on the concatenated size of HRV and PV features
        self.fc = nn.Linear(96  + 12, predict_length)  

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self,hrv, pv=pv, pv_inc=pv_inc,  use_pv_inc=use_pv_inc):
        
        x = self.initial(hrv)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if use_pv_inc:
            power = pv_inc[:, :, 0]
      
            angle = pv_inc[:, :, 1]
       
        
            combined = torch.cat((x, power, angle), dim=1)
        else: 
            x = torch.flatten(x, 1)
            pv = torch.flatten(pv, start_dim=1)
            if pv.dim() > 2:
                pv = torch.flatten(pv, start_dim=1)
            combined = torch.cat((x, pv), dim=1)

        #print(f"Sshape of x = {x.shape} shape of pv = {pv.shape}")
        #x = torch.concat((x, pv), dim=-1)
        #print("Shape after avgpool and flatten:", x.shape)

        
        
        #pv = pv.view(pv.size(0), -1)
        
        #print("Adjusted PV shape:", pv.shape)


        if self.fc.in_features != combined.shape[1]:
            self.fc = nn.Linear(combined.shape[1], predict_length).to(combined.device)

        out = self.fc(combined)
        return out
