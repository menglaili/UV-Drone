import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# so in the _init_ import the Resnet model part learned from ECE285;
# in the main.py load the weight of part of the model corresponding to 
# resnet and initialize other layers by hands https://discuss.pytorch.org/t/loading-a-few-layers-from-a-pretrained-mdnet/45221/4
# https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2;
# Give up all the FC in the resnet or Alexnet
class CorrResNet18(nn.Module):

    def __init__(self, num_actions = 6, fine_tuning = False):
        super(CorrResNet18, self).__init__()
        resnet = models.resnet18(pretrained=False)
        for param in resnet.parameters():
            param.requires_grad = fine_tuning
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool   #  output feature dimension 512 for input image dimension 224*224
        self.num_ftrs = resnet.fc.in_features
        self.linear_sensor = nn.Linear(5, 128)
        self.linear_final = nn.Linear(2*self.num_ftrs+2*128, num_actions)

    def forward(self, img, meta): # image pairs are cat along the channel dimension [batch, 6, width, height]
        siam1 = []
        for i in range(2): # the siamese network architecture 
            # for image
            x = self.relu(self.bn1(self.conv1(img[:,(i*3):(i+1)*3, :, :])))
            x = self.maxpool(x)
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            x = self.avgpool(x)
            x = x.view(-1, self.num_ftrs)
            siam1.append(x)
            # for sensor data
            y = self.linear_sensor(meta[:,:,i])
            siam2.append(y)
        out = torch.cat((siam1[0], siam1[1], siam2[0], siam2[1]), dim = 1)
        out = self.linear_final(out)
        return out