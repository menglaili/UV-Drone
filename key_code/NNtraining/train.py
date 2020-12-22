import os.path
import torch
import torch.utils.data as data
import numpy as np
from data_provider import gt_action_data, DroneCRDataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from NNutils import *
from torch.utils.tensorboard import SummaryWriter
from models import CorrCtrlNet


# BETA = 0.5
curr_lr = 0.01
num_epochs = 100
BatchSize = 32
gt_action_data()
with open("./data/train_action.txt", "r") as f:
    dcount = 0
    for line in f:
        dcount += 1
DatasetLen = dcount * 3
TrainValRatio = [0.8, 0.2]
TrainValLen = [int(TrainValRatio[0]*DatasetLen), DatasetLen - int(TrainValRatio[0]*DatasetLen)]
output_model_file = './output_model7'  # The new dataset result start from 2
writer = SummaryWriter('./runs/exp7')  # adding direction from 4   5 use mse   5.1 use l1loss  5.2 use loss add up
if not os.path.isdir(output_model_file):  # model5 gives the best performance before the wing broken
    os.mkdir(output_model_file)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

DroneCorrNet = CorrCtrlNet()
state_dict = torch.load('./output_model6final.pth') #('./pretrained/CorrCtrlNet_dir.pth') #('./output_model4final.ckpt') #('./pretrained/CorrCtrlNet_dir.pth')  #('./output_model2final.ckpt')
DroneCorrNet.load_state_dict(state_dict)
DroneCorrNet.to(device)

DroneData_train, DroneData_val = torch.utils.data.random_split(DroneCRDataset(mode='train', action = True), TrainValLen)
DroneDataLoader__train = torch.utils.data.DataLoader(DroneData_train, batch_size=BatchSize, shuffle=True, pin_memory=True)
DatasetLen_train = len(DroneDataLoader__train.dataset)
DroneDataLoader__val = torch.utils.data.DataLoader(DroneData_val, batch_size=BatchSize, shuffle=False, pin_memory=True)
DatasetLen_val = len(DroneDataLoader__val.dataset)

# criterion_rel = nn.MSELoss()
criterion_ctrl = nn.MSELoss() #nn.MSELoss() # nn.L1Loss
optimizer = torch.optim.Adam(DroneCorrNet.parameters(), lr = curr_lr)


for epoch in range(num_epochs):
    running_loss_train = 0.0
    DroneCorrNet.train()
    for i, Data in enumerate(DroneDataLoader__train):
        image = Data[0][0].float().to(device)
        direction = Data[0][1].float().to(device)
        gt_ctrl = Data[1].float().to(device)
        # gt_ctrl = Data[2].to(device)

        est_ctrl = DroneCorrNet(image, direction)  # gt_ctrl
        # loss_rel = criterion_rel(est_rel, gt_rel)
        loss_ctrl = criterion_ctrl(est_ctrl, gt_ctrl)
        loss = loss_ctrl #+ BETA*loss_rel

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss_train += loss.item() * Data[0][0].size(0)

    # update learning rate
    if ((epoch) % 30 == 0) and (epoch < 100):
        curr_lr /= 10
        update_lr(optimizer, curr_lr)

    loss_train = running_loss_train/DatasetLen_train
    print("Train: Epoch [{}/{}] Loss: {:.4f}"
          .format(epoch + 1, num_epochs, loss_train))
    writer.add_scalar('training_loss',
                      loss_train,
                      epoch)


    if (epoch) % 50 == 0:
        torch.save(DroneCorrNet.state_dict(), output_model_file + '/%s_model.pth' % (epoch))

    # test on the validation set at every epochs
    DroneCorrNet.eval()
    running_loss_val = 0.0
    for i, Data in enumerate(DroneDataLoader__val):
        image = Data[0][0].float().to(device)
        direction = Data[0][1].float().to(device)
        gt_ctrl = Data[1].float().to(device)
        # gt_ctrl = Data[2].to(device)

        est_ctrl = DroneCorrNet(image, direction)  # gt_ctrl
        # loss_rel = criterion_rel(est_rel, gt_rel)
        loss_ctrl = criterion_ctrl(est_ctrl, gt_ctrl)
        loss = loss_ctrl #+ BETA * loss_rel

        running_loss_val += loss.item() * Data[0][0].size(0)


    loss_val = running_loss_val / DatasetLen_val

    print("Validation: Epoch [{}/{}] Loss: {:.4f}"
          .format(epoch + 1, num_epochs, loss_val))
    writer.add_scalar('validation_loss',
                      loss_val,
                      epoch)

# Save the model checkpoint
torch.save(DroneCorrNet.state_dict(), output_model_file + 'final.ckpt')



