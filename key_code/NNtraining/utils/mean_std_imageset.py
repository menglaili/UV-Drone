import os
import numpy as np
import matplotlib.pyplot as plt
filepath = "./utils/office"
pathDir = os.listdir(filepath)

R_channel = 0
G_channel = 0
B_channel = 0

total_pixel = 0
for idx in range(len(pathDir)):
    filename = pathDir[idx]
    img = plt.imread(os.path.join(filepath, filename))
    R_channel += np.sum(img[:, :, 0])
    G_channel += np.sum(img[:, :, 1])
    B_channel += np.sum(img[:, :, 2])

num = len(pathDir) * img.shape[0] * img.shape[1]
R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num

R_total = 0
G_total = 0
B_total = 0
for idx in range(len(pathDir)):
    filename = pathDir[idx]
    img = plt.imread(os.path.join(filepath, filename))

    R_total += np.sum((img[:, :, 0] - R_mean) ** 2)
    G_total += np.sum((img[:, :, 1] - G_mean) ** 2)
    B_total += np.sum((img[:, :, 2] - B_mean) ** 2)

R_std = np.sqrt(R_total / num)
G_std = np.sqrt(G_total / num)
B_std = np.sqrt(B_total / num)

print("%1.3f, %1.3f, %1.3f, %1.3f, %1.3f, %1.3f" % (R_mean, G_mean, B_mean, R_std, G_std, B_std))


from torchvision import datasets, transforms
from torch.utils import data

filepath = "./data/train_data_for_drone_ctrl"
dataset = datasets.ImageFolder(root = filepath,
                transform = transforms.ToTensor())
loader = data.DataLoader(dataset, batch_size = 1, shuffle = False)

R_channel = 0
G_channel = 0
B_channel = 0

total_pixel = 0
for ind, datas in enumerate(loader):
    img = np.moveaxis(np.squeeze(datas[0].numpy()), 0, -1)
    R_channel += np.sum(img[:, :, 0])
    G_channel += np.sum(img[:, :, 1])
    B_channel += np.sum(img[:, :, 2])

num = len(dataset) * img.shape[0] * img.shape[1]
R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num

R_total = 0
G_total = 0
B_total = 0
for ind, datas in enumerate(loader):
    img = datas[0].numpy()
    R_total += np.sum((img[:, :, 0] - R_mean) ** 2)
    G_total += np.sum((img[:, :, 1] - G_mean) ** 2)
    B_total += np.sum((img[:, :, 2] - B_mean) ** 2)

R_std = np.sqrt(R_total / num)
G_std = np.sqrt(G_total / num)
B_std = np.sqrt(B_total / num)

print("%1.3f, %1.3f, %1.3f, %1.3f, %1.3f, %1.3f" % (R_mean, G_mean, B_mean, R_std, G_std, B_std))
