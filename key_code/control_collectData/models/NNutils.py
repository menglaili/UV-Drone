import os.path
import h5py
import math
import torch
import torch.nn as nn
import torchvision.models as models
from NNmodels import CorrResNet34, CorrCtrlNet
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'



def set_weight(savemodel = False):
    # the lua resnet34 model, has additional tensor after conv is the zero bias
    # The down-sampling layer does not have BN but only one conv weight + zero bias
    model = CorrResNet34()
    myFile = h5py.File('weight_lua.h5', 'r')
    runnning = h5py.File('batchrunning.h5', 'r')
    def mean_var(inte):
        mean = torch.from_numpy(runnning[str(inte) + "_mean"].value)
        var = torch.from_numpy(runnning[str(inte) + "_var"].value)
        return mean, var
    with torch.no_grad():
        model.resnet.conv1.weight = nn.Parameter(torch.from_numpy(myFile["1"].value))
        model.resnet.bn1.weight = nn.Parameter(torch.from_numpy(myFile["3"].value))
        model.resnet.bn1.bias = nn.Parameter(torch.from_numpy(myFile["4"].value))
        model.resnet.bn1.running_mean, model.resnet.bn1.running_var = mean_var(1)
        model.resnet.layer1[0].conv1.weight = nn.Parameter(torch.from_numpy(myFile["5"].value))
        model.resnet.layer1[0].bn1.weight = nn.Parameter(torch.from_numpy(myFile["7"].value))
        model.resnet.layer1[0].bn1.bias = nn.Parameter(torch.from_numpy(myFile["8"].value))
        model.resnet.layer1[0].bn1.running_mean, model.resnet.layer1[0].bn1.running_var = mean_var(2)
        model.resnet.layer1[0].conv2.weight = nn.Parameter(torch.from_numpy(myFile["9"].value))
        model.resnet.layer1[0].bn2.weight = nn.Parameter(torch.from_numpy(myFile["11"].value))
        model.resnet.layer1[0].bn2.bias = nn.Parameter(torch.from_numpy(myFile["12"].value))
        model.resnet.layer1[0].bn2.running_mean, model.resnet.layer1[0].bn2.running_var = mean_var(3)
        model.resnet.layer1[1].conv1.weight = nn.Parameter(torch.from_numpy(myFile["13"].value))
        model.resnet.layer1[1].bn1.weight = nn.Parameter(torch.from_numpy(myFile["15"].value))
        model.resnet.layer1[1].bn1.bias = nn.Parameter(torch.from_numpy(myFile["16"].value))
        model.resnet.layer1[1].bn1.running_mean, model.resnet.layer1[1].bn1.running_var = mean_var(4)
        model.resnet.layer1[1].conv2.weight = nn.Parameter(torch.from_numpy(myFile["17"].value))
        model.resnet.layer1[1].bn2.weight = nn.Parameter(torch.from_numpy(myFile["19"].value))
        model.resnet.layer1[1].bn2.bias = nn.Parameter(torch.from_numpy(myFile["20"].value))
        model.resnet.layer1[1].bn2.running_mean, model.resnet.layer1[1].bn2.running_var = mean_var(5)
        model.resnet.layer1[2].conv1.weight = nn.Parameter(torch.from_numpy(myFile["21"].value))
        model.resnet.layer1[2].bn1.weight = nn.Parameter(torch.from_numpy(myFile["23"].value))
        model.resnet.layer1[2].bn1.bias = nn.Parameter(torch.from_numpy(myFile["24"].value))
        model.resnet.layer1[2].bn1.running_mean, model.resnet.layer1[2].bn1.running_var = mean_var(6)
        model.resnet.layer1[2].conv2.weight = nn.Parameter(torch.from_numpy(myFile["25"].value))
        model.resnet.layer1[2].bn2.weight = nn.Parameter(torch.from_numpy(myFile["27"].value))
        model.resnet.layer1[2].bn2.bias = nn.Parameter(torch.from_numpy(myFile["28"].value))
        model.resnet.layer1[2].bn2.running_mean, model.resnet.layer1[2].bn2.running_var = mean_var(7)
        model.resnet.layer2[0].conv1.weight = nn.Parameter(torch.from_numpy(myFile["29"].value))
        model.resnet.layer2[0].bn1.weight = nn.Parameter(torch.from_numpy(myFile["31"].value))
        model.resnet.layer2[0].bn1.bias = nn.Parameter(torch.from_numpy(myFile["32"].value))
        model.resnet.layer2[0].bn1.running_mean, model.resnet.layer2[0].bn1.running_var = mean_var(8)
        model.resnet.layer2[0].conv2.weight = nn.Parameter(torch.from_numpy(myFile["33"].value))
        model.resnet.layer2[0].bn2.weight = nn.Parameter(torch.from_numpy(myFile["35"].value))
        model.resnet.layer2[0].bn2.bias = nn.Parameter(torch.from_numpy(myFile["36"].value))
        model.resnet.layer2[0].bn2.running_mean, model.resnet.layer2[0].bn2.running_var = mean_var(9)
        model.resnet.layer2[0].downsample[0].weight = nn.Parameter(torch.from_numpy(myFile["37"].value))
        model.resnet.layer2[1].conv1.weight = nn.Parameter(torch.from_numpy(myFile["39"].value))
        model.resnet.layer2[1].bn1.weight = nn.Parameter(torch.from_numpy(myFile["41"].value))
        model.resnet.layer2[1].bn1.bias = nn.Parameter(torch.from_numpy(myFile["42"].value))
        model.resnet.layer2[1].bn1.running_mean, model.resnet.layer2[1].bn1.running_var = mean_var(10)
        model.resnet.layer2[1].conv2.weight = nn.Parameter(torch.from_numpy(myFile["43"].value))
        model.resnet.layer2[1].bn2.weight = nn.Parameter(torch.from_numpy(myFile["45"].value))
        model.resnet.layer2[1].bn2.bias = nn.Parameter(torch.from_numpy(myFile["46"].value))
        model.resnet.layer2[1].bn2.running_mean, model.resnet.layer2[1].bn2.running_var = mean_var(11)
        model.resnet.layer2[2].conv1.weight = nn.Parameter(torch.from_numpy(myFile["47"].value))
        model.resnet.layer2[2].bn1.weight = nn.Parameter(torch.from_numpy(myFile["49"].value))
        model.resnet.layer2[2].bn1.bias = nn.Parameter(torch.from_numpy(myFile["50"].value))
        model.resnet.layer2[2].bn1.running_mean, model.resnet.layer2[2].bn1.running_var = mean_var(12)
        model.resnet.layer2[2].conv2.weight = nn.Parameter(torch.from_numpy(myFile["51"].value))
        model.resnet.layer2[2].bn2.weight = nn.Parameter(torch.from_numpy(myFile["53"].value))
        model.resnet.layer2[2].bn2.bias = nn.Parameter(torch.from_numpy(myFile["54"].value))
        model.resnet.layer2[2].bn2.running_mean, model.resnet.layer2[2].bn2.running_var = mean_var(13)
        model.resnet.layer2[3].conv1.weight = nn.Parameter(torch.from_numpy(myFile["55"].value))
        model.resnet.layer2[3].bn1.weight = nn.Parameter(torch.from_numpy(myFile["57"].value))
        model.resnet.layer2[3].bn1.bias = nn.Parameter(torch.from_numpy(myFile["58"].value))
        model.resnet.layer2[3].bn1.running_mean, model.resnet.layer2[3].bn1.running_var = mean_var(14)
        model.resnet.layer2[3].conv2.weight = nn.Parameter(torch.from_numpy(myFile["59"].value))
        model.resnet.layer2[3].bn2.weight = nn.Parameter(torch.from_numpy(myFile["61"].value))
        model.resnet.layer2[3].bn2.bias = nn.Parameter(torch.from_numpy(myFile["62"].value))
        model.resnet.layer2[3].bn2.running_mean, model.resnet.layer2[3].bn2.running_var = mean_var(15)
        model.resnet.layer3[0].conv1.weight = nn.Parameter(torch.from_numpy(myFile["63"].value))
        model.resnet.layer3[0].bn1.weight = nn.Parameter(torch.from_numpy(myFile["65"].value))
        model.resnet.layer3[0].bn1.bias = nn.Parameter(torch.from_numpy(myFile["66"].value))
        model.resnet.layer3[0].bn1.running_mean, model.resnet.layer3[0].bn1.running_var = mean_var(16)
        model.resnet.layer3[0].conv2.weight = nn.Parameter(torch.from_numpy(myFile["67"].value))
        model.resnet.layer3[0].bn2.weight = nn.Parameter(torch.from_numpy(myFile["69"].value))
        model.resnet.layer3[0].bn2.bias = nn.Parameter(torch.from_numpy(myFile["70"].value))
        model.resnet.layer3[0].bn2.running_mean, model.resnet.layer3[0].bn2.running_var = mean_var(17)
        model.resnet.layer3[0].downsample[0].weight = nn.Parameter(torch.from_numpy(myFile["71"].value))
        model.resnet.layer3[1].conv1.weight = nn.Parameter(torch.from_numpy(myFile["73"].value))
        model.resnet.layer3[1].bn1.weight = nn.Parameter(torch.from_numpy(myFile["75"].value))
        model.resnet.layer3[1].bn1.bias = nn.Parameter(torch.from_numpy(myFile["76"].value))
        model.resnet.layer3[1].bn1.running_mean, model.resnet.layer3[1].bn1.running_var = mean_var(18)
        model.resnet.layer3[1].conv2.weight = nn.Parameter(torch.from_numpy(myFile["77"].value))
        model.resnet.layer3[1].bn2.weight = nn.Parameter(torch.from_numpy(myFile["79"].value))
        model.resnet.layer3[1].bn2.bias = nn.Parameter(torch.from_numpy(myFile["80"].value))
        model.resnet.layer3[1].bn2.running_mean, model.resnet.layer3[1].bn2.running_var = mean_var(19)
        model.resnet.layer3[2].conv1.weight = nn.Parameter(torch.from_numpy(myFile["81"].value))
        model.resnet.layer3[2].bn1.weight = nn.Parameter(torch.from_numpy(myFile["83"].value))
        model.resnet.layer3[2].bn1.bias = nn.Parameter(torch.from_numpy(myFile["84"].value))
        model.resnet.layer3[2].bn1.running_mean, model.resnet.layer3[2].bn1.running_var = mean_var(20)
        model.resnet.layer3[2].conv2.weight = nn.Parameter(torch.from_numpy(myFile["85"].value))
        model.resnet.layer3[2].bn2.weight = nn.Parameter(torch.from_numpy(myFile["87"].value))
        model.resnet.layer3[2].bn2.bias = nn.Parameter(torch.from_numpy(myFile["88"].value))
        model.resnet.layer3[2].bn2.running_mean, model.resnet.layer3[2].bn2.running_var = mean_var(21)
        model.resnet.layer3[3].conv1.weight = nn.Parameter(torch.from_numpy(myFile["89"].value))
        model.resnet.layer3[3].bn1.weight = nn.Parameter(torch.from_numpy(myFile["91"].value))
        model.resnet.layer3[3].bn1.bias = nn.Parameter(torch.from_numpy(myFile["92"].value))
        model.resnet.layer3[3].bn1.running_mean, model.resnet.layer3[3].bn1.running_var = mean_var(22)
        model.resnet.layer3[3].conv2.weight = nn.Parameter(torch.from_numpy(myFile["93"].value))
        model.resnet.layer3[3].bn2.weight = nn.Parameter(torch.from_numpy(myFile["95"].value))
        model.resnet.layer3[3].bn2.bias = nn.Parameter(torch.from_numpy(myFile["96"].value))
        model.resnet.layer3[3].bn2.running_mean, model.resnet.layer3[3].bn2.running_var = mean_var(23)
        model.resnet.layer3[4].conv1.weight = nn.Parameter(torch.from_numpy(myFile["97"].value))
        model.resnet.layer3[4].bn1.weight = nn.Parameter(torch.from_numpy(myFile["99"].value))
        model.resnet.layer3[4].bn1.bias = nn.Parameter(torch.from_numpy(myFile["100"].value))
        model.resnet.layer3[4].bn1.running_mean, model.resnet.layer3[4].bn1.running_var = mean_var(24)
        model.resnet.layer3[4].conv2.weight = nn.Parameter(torch.from_numpy(myFile["101"].value))
        model.resnet.layer3[4].bn2.weight = nn.Parameter(torch.from_numpy(myFile["103"].value))
        model.resnet.layer3[4].bn2.bias = nn.Parameter(torch.from_numpy(myFile["104"].value))
        model.resnet.layer3[4].bn2.running_mean, model.resnet.layer3[4].bn2.running_var = mean_var(25)
        model.resnet.layer3[5].conv1.weight = nn.Parameter(torch.from_numpy(myFile["105"].value))
        model.resnet.layer3[5].bn1.weight = nn.Parameter(torch.from_numpy(myFile["107"].value))
        model.resnet.layer3[5].bn1.bias = nn.Parameter(torch.from_numpy(myFile["108"].value))
        model.resnet.layer3[5].bn1.running_mean, model.resnet.layer3[5].bn1.running_var = mean_var(26)
        model.resnet.layer3[5].conv2.weight = nn.Parameter(torch.from_numpy(myFile["109"].value))
        model.resnet.layer3[5].bn2.weight = nn.Parameter(torch.from_numpy(myFile["111"].value))
        model.resnet.layer3[5].bn2.bias = nn.Parameter(torch.from_numpy(myFile["112"].value))
        model.resnet.layer3[5].bn2.running_mean, model.resnet.layer3[5].bn2.running_var = mean_var(27)
        model.resnet.layer4[0].conv1.weight = nn.Parameter(torch.from_numpy(myFile["113"].value))
        model.resnet.layer4[0].bn1.weight = nn.Parameter(torch.from_numpy(myFile["115"].value))
        model.resnet.layer4[0].bn1.bias = nn.Parameter(torch.from_numpy(myFile["116"].value))
        model.resnet.layer4[0].bn1.running_mean, model.resnet.layer4[0].bn1.running_var = mean_var(28)
        model.resnet.layer4[0].conv2.weight = nn.Parameter(torch.from_numpy(myFile["117"].value))
        model.resnet.layer4[0].bn2.weight = nn.Parameter(torch.from_numpy(myFile["119"].value))
        model.resnet.layer4[0].bn2.bias = nn.Parameter(torch.from_numpy(myFile["120"].value))
        model.resnet.layer4[0].bn2.running_mean, model.resnet.layer4[0].bn2.running_var = mean_var(29)
        model.resnet.layer4[0].downsample[0].weight = nn.Parameter(torch.from_numpy(myFile["121"].value))
        model.resnet.layer4[1].conv1.weight = nn.Parameter(torch.from_numpy(myFile["123"].value))
        model.resnet.layer4[1].bn1.weight = nn.Parameter(torch.from_numpy(myFile["125"].value))
        model.resnet.layer4[1].bn1.bias = nn.Parameter(torch.from_numpy(myFile["126"].value))
        model.resnet.layer4[1].bn1.running_mean, model.resnet.layer4[1].bn1.running_var = mean_var(30)
        model.resnet.layer4[1].conv2.weight = nn.Parameter(torch.from_numpy(myFile["127"].value))
        model.resnet.layer4[1].bn2.weight = nn.Parameter(torch.from_numpy(myFile["129"].value))
        model.resnet.layer4[1].bn2.bias = nn.Parameter(torch.from_numpy(myFile["130"].value))
        model.resnet.layer4[1].bn2.running_mean, model.resnet.layer4[1].bn2.running_var = mean_var(31)
        model.resnet.layer4[2].conv1.weight = nn.Parameter(torch.from_numpy(myFile["131"].value))
        model.resnet.layer4[2].bn1.weight = nn.Parameter(torch.from_numpy(myFile["133"].value))
        model.resnet.layer4[2].bn1.bias = nn.Parameter(torch.from_numpy(myFile["134"].value))
        model.resnet.layer4[2].bn1.running_mean, model.resnet.layer4[2].bn1.running_var = mean_var(32)
        model.resnet.layer4[2].conv2.weight = nn.Parameter(torch.from_numpy(myFile["135"].value))
        model.resnet.layer4[2].bn2.weight = nn.Parameter(torch.from_numpy(myFile["137"].value))
        model.resnet.layer4[2].bn2.bias = nn.Parameter(torch.from_numpy(myFile["138"].value))
        model.resnet.layer4[2].bn2.running_mean, model.resnet.layer4[2].bn2.running_var = mean_var(33)
        model.fc.weight = nn.Parameter(torch.from_numpy(myFile["277"].value))
        model.fc.bias = nn.Parameter(torch.from_numpy(myFile["278"].value))
        model.bn.weight = nn.Parameter(torch.from_numpy(myFile["279"].value))
        model.bn.bias = nn.Parameter(torch.from_numpy(myFile["280"].value))
        model.bn.running_mean, model.bn.running_var = mean_var(34)
        model.fc_t.weight = nn.Parameter(torch.from_numpy(myFile["281"].value))
        model.fc_t.bias = nn.Parameter(torch.from_numpy(myFile["282"].value))
    if savemodel:
        torch.save(model.state_dict(), './pretrained/pretrained.pth')
    return model

def setup_cornet(savemodel = False):
    model = CorrCtrlNet()
    state_dict = torch.load('./pretrained/pretrained.pth')
    state_dict["fc_ctrl1.weight"] = nn.init.kaiming_uniform_(model.fc_ctrl1.weight).float()
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.fc_ctrl1.weight)
    bound = 1 / math.sqrt(fan_in)
    state_dict["fc_ctrl1.bias"] = nn.init.uniform_(model.fc_ctrl1.bias, -bound, bound).float()
    state_dict["fc_final.weight"] = nn.init.kaiming_uniform_(model.fc_final.weight).float()
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.fc_final.weight)
    bound = 1 / math.sqrt(fan_in)
    state_dict["fc_final.bias"] = nn.init.uniform_(model.fc_final.bias, -bound, bound).float()
    state_dict["bn_ctrl1.weight"] = model.bn_ctrl1.weight.data.fill_(1).float()
    state_dict["bn_ctrl1.bias"] = model.bn_ctrl1.bias.data.fill_(0).float()
    state_dict["bn_ctrl1.running_var"] = model.bn_ctrl1.running_var.data.fill_(1).float()
    state_dict["bn_ctrl1.running_mean"] = model.bn_ctrl1.running_mean.data.fill_(0).float()
    model.load_state_dict(state_dict)
    if savemodel:
        torch.save(model.state_dict(), './pretrained/CorrCtrlNet.pth')
    return model

def test_resnet34():
    resnet = models.resnet34(pretrained=True)
    return resnet

def test_resnet34_state_dict():
    resnet = models.resnet34(pretrained=False)
    resn_model = torch.load("resnet34-333f7ec4.pth")
    resnet.load_state_dict(resn_model)
    return resn_model

def print_para(model):
    for param_tensor in model.state_dict():
        print(param_tensor, model.state_dict()[param_tensor].shape)


if "__name__" == "__main__":
    import os.path
    import torch
    import torch.utils.data as data
    from data_provider import DroneCRDataset


    model = CorrResNet34()
    myFile = h5py.File('weight_lua.h5', 'r')

    DroneCorrNet = set_weight()
    DroneCorrNet.eval()

    DroneData = DroneCRDataset(mode='test')
    DroneDataLoader = torch.utils.data.DataLoader(DroneData, batch_size=1, shuffle=False, pin_memory=False)
    criterion = nn.MSELoss()

