import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# so in the _init_ import the Resnet model part learned from ECE285;
# in the main.py load the weight of part of the model corresponding to 
# resnet and initialize other layers by hands https://discuss.pytorch.org/t/loading-a-few-layers-from-a-pretrained-mdnet/45221/4
# https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2

class Net(nn.Module):
   def __init__(self):
      super().__init__()
      
      self.conv1 = nn.Conv2d(1, 64, 7)
      self.pool1 = nn.MaxPool2d(2)
      self.conv2 = nn.Conv2d(64, 128, 5)
      self.conv3 = nn.Conv2d(128, 256, 5)
      self.linear1 = nn.Linear(2304, 512)
      
      self.linear2 = nn.Linear(512, 2)
      
   def forward(self, data):
      res = []
      for i in range(2): # Siamese nets; sharing weights
         x = data[i]
         x = self.conv1(x)
         x = F.relu(x)
         x = self.pool1(x)
         x = self.conv2(x)
         x = F.relu(x)
         x = self.conv3(x)
         x = F.relu(x)
         
         x = x.view(x.shape[0], -1)
         x = self.linear1(x)
         res.append(F.relu(x))
         
      res = torch.abs(res[1] - res[0])
      res = self.linear2(res)
      return res
