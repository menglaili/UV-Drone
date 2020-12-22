import os.path
import torch
import torch.utils.data as data
from data_provider import DroneCRDataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from NNutils import *
from utils.cumulative_err import *
from utils.plot_scatter_orientation import *

# original network
DroneData = DroneCRDataset(mode='test', action = False)
DroneDataLoader = torch.utils.data.DataLoader(DroneData, batch_size=1, shuffle=False, pin_memory=False)
criterion = nn.MSELoss()

DroneCorrNet = CorrResNet34()
state_dict = torch.load('./pretrained/pretrained.pth')
DroneCorrNet.load_state_dict(state_dict)
DroneCorrNet.to(device).eval()

with torch.no_grad():
    est_translation = []
    loss_list = []
    gt_list = []
    for i, Data in enumerate(DroneDataLoader):
        image = Data[0].to(device).float()
        labels = Data[1].to(device).float()
        cur_pose = Data[2].numpy()
        cur_R = Quaternion(cur_pose[0, 3:]).rotation_matrix
        outputs = DroneCorrNet(image)
        loss = criterion(outputs, labels)
        gt_list.append(labels.cpu().numpy())
        est_translation.append(cur_R.T.dot(outputs.cpu().numpy().T))
        loss_list.append(loss.item())

gt_list = np.squeeze(np.array(gt_list))[:, :2]
est_translation = np.squeeze(np.array(est_translation))[:, :2]

# self network
DroneData = DroneCRDataset(mode='test')
DroneDataLoader = torch.utils.data.DataLoader(DroneData, batch_size=1, shuffle=False, pin_memory=False)
criterion = nn.MSELoss()

DroneCorrNet = CorrCtrlNet()
state_dict = torch.load('./output_model7final.ckpt')
DroneCorrNet.load_state_dict(state_dict)
DroneCorrNet.to(device).eval()

with torch.no_grad():
    est_translation = []
    loss_list = []
    gt_list = []
    for i, Data in enumerate(DroneDataLoader):
        image = Data[0][0].to(device).float()
        direction = Data[0][1].to(device).float()
        labels = Data[1].to(device).float()
        outputs = DroneCorrNet(image, direction)
        loss = criterion(outputs, labels)
        gt_list.append(labels.cpu().numpy())
        est_translation.append(outputs.cpu().numpy())
        loss_list.append(loss.item())


# plot cumulative error
from scipy.special import softmax
gt_list = np.squeeze(np.array(gt_list))
est_translation = np.squeeze(np.array(est_translation))
gt_list1 = gt_list # *4
est_translation1 = est_translation
diff = np.abs(gt_list1) - np.abs(est_translation1)
relative_accuracy = np.linalg.norm(diff, axis = 1) / np.linalg.norm(gt_list1, axis = 1)
# relative_accuracy[np.argmax(relative_accuracy)] = 1
counts, bins = np.histogram(relative_accuracy, bins=20)
plt.bar(bins[:-1], counts/np.sum(counts), 0.15)
plt.hist(bins[:-1], bins, weights = counts, density=False, stacked = False, cumulative = False) #
plt.xlabel("normalized error")
plt.ylabel("probability")
# plt.locator_params(nbins=100, axis='x')
print("mean relative error rate: ", np.sum(relative_accuracy < 1))


xs = np.arange(diff.shape[0])
diff = np.hstack((xs.reshape(-1, 1), diff, est_translation)).tolist()
diff_xmax = sorted(diff, key = lambda x:x[1], reverse = True)


trans_error_plot_new_xyz(gt_list1, est_translation, savefig = False)
# plot xyz error bar
plotxyzerrbar(gt_list1, est_translation)

