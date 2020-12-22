import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

def print_para(model):
    model = model()
    for name, param in model.named_parameters():
        print(name, param.data.shape, param.data)

def check_dataset(dataset, Net):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a = dataset()
    b = torch.utils.data.DataLoader(a, batch_size=2, shuffle=True, pin_memory=True)
    model = Net().to(device)
    for i, datas in enumerate(b):
        print(datas[0][1].size())
        print(model(datas[0][0].float().to(device), datas[0][1].float().to(device)))
        if i == 1:
            break

class OriCorrNet(nn.Module):
    def __init__(self, num_actions=1):
        super(OriCorrNet, self).__init__()
        self.fc0 = nn.Linear(2, 16)
        self.bn = nn.BatchNorm1d(16)
        self.fc1 = nn.Linear(16, num_actions)

    def forward(self, ori_height):  # [ori in theta, height] batch*channel
        out = F.relu(self.bn(self.fc0(ori_height)))
        print(out.size())
        out = self.fc1(out)
        print(out.size())
        return out

class CorrResNet34(nn.Module):
    def __init__(self, fine_tuning=False):
        super(CorrResNet34, self).__init__()
        self.resnet = models.resnet34(pretrained=False)
        self.resnet.fc = nn.Identity()
        self.fc = nn.Linear(1024, 1024)
        self.bn = nn.BatchNorm1d(1024)
        self.fc_t = nn.Linear(1024, 3)
        for param in self.resnet.parameters():
            param.requires_grad = fine_tuning

    def forward(self, img):  # image pairs are cat along the channel dimension [batch, 6, width, height]
        siam = []
        for i in range(2):  # the siamese network architecture
            x = self.resnet(img[:, (i * 3):(i + 1) * 3, :, :])
            x = x.view(-1, 512)
            siam.append(x)
        out_siam = torch.cat((siam[0], siam[1]), dim=1)
        self.out_fc_siam = F.relu(self.bn(self.fc(out_siam)))
        self.out_rel = self.fc_t(self.out_fc_siam)

        return self.out_rel


class CorrCtrlNet(nn.Module):
    def __init__(self, num_actions=2, fine_tuning=False, drop_p = 0.5):
        super(CorrCtrlNet, self).__init__()
        self.resnet = models.resnet34(pretrained=False)
        self.resnet.fc = nn.Identity()
        self.fc = nn.Linear(1024, 1024)
        self.bn = nn.BatchNorm1d(1024)
        self.drop_layer1 = nn.Dropout(p=drop_p)
        for param in self.resnet.parameters(): # enable gradient changing
            param.requires_grad = fine_tuning
        self.fc_ctrl1 = nn.Linear(1024, 4)
        self.bn_ctrl1 = nn.BatchNorm1d(4)
        self.drop_layer2 = nn.Dropout(p=drop_p)
        self.fc_conca1 = nn.Linear(8, 4)
        self.bn_conca1 = nn.BatchNorm1d(4)
        self.fc_conca2 = nn.Linear(4, 4)
        self.bn_conca2 = nn.BatchNorm1d(4)
        self.fc_final = nn.Linear(4, num_actions)


    def forward(self, img, direction):  # direction is one-hot coding for an eight dimension vector
        siam = []
        for i in range(2):  # the siamese network architecture
            x = self.resnet(img[:, (i * 3):(i + 1) * 3, :, :])
            x = x.view(-1, 512)
            siam.append(x)
        out_siam = torch.cat((siam[0], siam[1]), dim=1)
        out_fc_siam = self.drop_layer1(F.relu(self.bn(self.fc(out_siam))))
        out_fc_ctrl1 = self.drop_layer2(F.relu(self.bn_ctrl1(self.fc_ctrl1(out_fc_siam))))
        out_conca = torch.cat((out_fc_ctrl1, direction), dim = 1)
        out_fc_conca1 = F.relu(self.bn_conca1(self.fc_conca1(out_conca)))
        out_fc_conca2 = F.relu(self.bn_conca2(self.fc_conca2(out_fc_conca1)))
        out_ctrl = self.fc_final(out_fc_conca2)
        return out_ctrl  #, self.out_rel


if "__name__" == "__main__":
    from data_provider import DroneCRDataset
    check_dataset(DroneCRDataset, CorrCtrlNet)
    print_para(CorrCtrlNet)

    from data_provider import DroneOriHeiSet
    check_dataset(DroneOriHeiSet, OriCorrNet)
    print_para(OriCorrNet)

    DroneCorrNet = CorrCtrlNet()
    state_dict = torch.load('./pretrained/CorrCtrlNet.pth')
    DroneCorrNet.load_state_dict(state_dict)
    check_dataset(DroneCRDataset, DroneCorrNet)

