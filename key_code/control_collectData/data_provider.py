import os
import torch
import random
import torch.utils.data as data
import numpy as np
from PIL import Image
import torchvision as tv
from utils.cumulative_err import gt_rel_translation, qua2rad
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# create ground truth image_pair-label txt file
def gt_data_txt(dataset_dir = "./data", mode = "test"):
    with open(os.path.join(dataset_dir, "%s.txt" % mode), "r") as f:
        data_addr = [line.split()[:2] for line in f]
    data_pose = np.loadtxt(os.path.join(dataset_dir, "%s_pose.txt" % mode))
    assert len(data_addr) == (data_pose).shape[0], "data_addr and data_pose size mismatch"
    with open("%s.txt" % mode, "w") as f:
        for i in range(len(data_addr)):
            pose1 = data_pose[i, :3]
            pose2 = data_pose[i, 7:10]
            gt_rel_trans = np.around((pose2 - pose1), 4).astype(np.str).tolist()
            f.write(data_addr[i][0] + " " + data_addr[i][1] + " " + " ".join(gt_rel_trans) + "\n")
            # f.write(data_addr[i][0] + " " + data_addr[i][1] + " " + " ".join(pose1) + " " + " ".join(pose2) \
            #         + " " + " ".join(rel_trans) + "\n")

# create txt file for action training
def gt_action_data(data_ref_dir = "./data/Ref_Data_free", mode = "train"):
    # Train and Val: img_addr1, img_addr2, action1, action2
    # Test: img_addr1, img_addr2, action1, action2
    data_cor_dir = "./data/%s_data_for_drone_ctrl" % mode
    traj_dir_list = os.listdir(data_cor_dir)
    string_list = []
    for i, file in enumerate(traj_dir_list):
        cur_traj_dir = os.path.join(data_cor_dir, file)
        if not checkdir(cur_traj_dir):
            continue
        ctrl_list = load_ctrltxt(os.path.join(cur_traj_dir, "ref_ctrl_current.txt"))
        startind = 0
        while startind < ctrl_list.shape[0]:
            p1 = ctrl_list[startind, 0]
            count = 0
            img_addr_list = []
            indx_list = []
            while startind < ctrl_list.shape[0] and ctrl_list[startind, 0] == p1:
                if ctrl_list[startind, 0] not in indx_list:
                    indx_list.append(ctrl_list[startind, 0])
                    cur_str = str(int(ctrl_list[startind, 0])) + "current.png"
                    all_str = os.path.join(data_cor_dir, file, "image", cur_str)
                    img_addr_list.append(all_str)
                    startind += 1
                else:
                    cur_str = str(int(ctrl_list[startind, 0])) + "_" + str(count) + "_" + "current.png"
                    all_str = os.path.join(data_cor_dir, file, "image", cur_str)
                    img_addr_list.append(all_str)
                    count += 1
                    startind += 1
            img_addr_list.append(os.path.join(data_ref_dir, "image", str(int(ctrl_list[startind-1, 0])) + ".png"))
            # with open(mode+"_action.txt", "a") as f:
            for j in range(len(img_addr_list)-1):
                dirnum = finddirection(img_addr_list[j])
                string_list.append(img_addr_list[j] + " " + img_addr_list[j+1] + " " + \
                                   " ".join(np.around(ctrl_list[startind - len(img_addr_list) + 1 + j, 1:], 4).astype(np.str).tolist()) + \
                                   " " + dirnum + "\n")
                    # f.write(img_addr_list[j] + " " + img_addr_list[j+1] + " " \
                    #           + " ".join(np.around(ctrl_list[startind - len(img_addr_list) + 1 + j, 1:], 4).astype(np.str).tolist()) + "\n")
    with open("./data/%s_action.txt" % mode, "w") as f:
        for k in range(len(string_list)):
            xy = np.array(string_list[k].split()[2:4], dtype = np.float)
            if np.any(np.abs(xy) > 0.05):
                f.write(string_list[k])


def checkdir(cur_traj_dir):
    sign = True
    img_dir = os.path.join(cur_traj_dir, "image")
    # pose_dir = os.path.join(cur_traj_dir, "pose")
    ctrl_dir = os.path.join(cur_traj_dir, "ref_ctrl_current.txt")
    if not os.path.isdir(cur_traj_dir):
        sign = False
    if (not os.path.isdir(img_dir)):
        sign = False
    # if (not os.path.isdir(pose_dir)):
    #     sign = False
    if (not os.path.isfile(ctrl_dir)):
        sign = False
    return sign

def load_ctrltxt(ctrltxt_dir):
    ctrl_list = []
    with open(ctrltxt_dir, "r") as f:
        for line in f:
            if line != 'None\n':
                ctrl_list.append(line.split())
    ctrl_list = np.array(ctrl_list, dtype = np.float)
    return ctrl_list

def finddirection(imgaddr):
    '''
    0,1,2,3,4: 0
    5,6,7: 90
    8,9,10,11,12: 180
    '''
    def direction2num(string):
        # string is 2 char
        if string.isnumeric():
            return str(180)
        else:
            if int(string[1])<5:
                return str(0)
            elif int(string[1])<8:
                return str(90)
            else:
                return str(180)

    if imgaddr[-12].isnumeric():
        return direction2num(imgaddr[-13:-11])
    else:
        return direction2num(imgaddr[-16:-14])





# a = load_ctrltxt("./Training_data_for_drone_ctrl/0/ref_ctrl_current.txt")
# gt_action_data()


class DroneOriHeiSet(data.Dataset):
    # (reference)qw, qx, qy, qz, (current)qw, qx, qy, qz, height, height, action_ori, action_height
    # the action is from current to reference similar to base to query
    def __init__(self, dataset_dir = "./data"):
        super(DroneOriHeiSet, self).__init__()
        with open(os.path.join(dataset_dir, "orihei.txt"), "r") as f:
            self.data_addr = [line.split() for line in f]
        self.data_size = len(self.data_addr)
        self.dataset_dir = dataset_dir

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        rad_ref = qua2rad(np.array(self.data_addr[idx][:4], dtype = np.float))
        rad_cur = qua2rad(np.array(self.data_addr[idx][4:8], dtype = np.float))
        height_ref = float(self.data_addr[idx][8])
        height_cur = float(self.data_addr[idx][9])
        action = np.array(self.data_addr[idx][-2:], dtype = np.float)

        return np.array(((rad_cur - rad_ref), (height_cur - height_ref))), action


def trainval_test_split(data_path = "origin.txt", debug = False, TrainVal_Test_ratio = 0.8):
    with open(data_path, "r") as f:
        data_addr = [line.split() for line in f]
    if debug:
        random.seed(42)
        random.shuffle(data_addr)
    else:
        random.shuffle(data_addr)
    data_addr_trainval = data_addr[:int(TrainVal_Test_ratio * len(data_addr))]
    data_addr_test = data_addr[int(TrainVal_Test_ratio * len(data_addr)):]
    with open("train.txt", "w") as f1:
        for ele in data_addr_trainval:
            f1.write(" ".join(ele) + "\n")
    with open("test.txt", "w") as f2:
        for ele in data_addr_test:
            f2.write(" ".join(ele) + "\n")


class DroneCRDataset(data.Dataset):
    def __init__(self, action = True, mode='train', image_size=(224, 224), dataset_dir = "./data"):
        # No action: Train and Val: img_addr1, img_addr2, reltranx, reltrany, reltranz
        # No action: Test: img_addr1, img_addr2, reltranx, reltrany, reltranz, gt_pose1, gt_pose2
        # Action: img_addr1, img_addr2, action
        super(DroneCRDataset, self).__init__()
        if action:
            with open(os.path.join(dataset_dir, "%s_action.txt" % mode), "r") as f:
                self.data_addr = [line.split() for line in f]
        else:
            self.img_dir = os.path.join(dataset_dir, "image")
            with open(os.path.join(dataset_dir, "%s.txt" % mode), "r") as f:
                if mode == "train" or mode == "val":
                    self.data_addr = [line.split()[:5] for line in f]
                elif mode == "test":
                    self.data_addr = [line.split() for line in f]
        self.data_size = len(self.data_addr)
        self.image_size = image_size
        self.mode = mode
        self.action = action

    def __len__(self):
        if self.mode == "train":
            return 3 * self.data_size  # augment the dataset for 3 times, randomness from randomcrop
        else:
            return self.data_size

    def __getitem__(self, idx):
        idx = idx % self.data_size
        # read in images
        if self.action:
            image1 = Image.open(self.data_addr[idx][0])
            image2 = Image.open(self.data_addr[idx][1])
        else:
            image1 = Image.open(os.path.join(self.img_dir, self.data_addr[idx][0]))
            image2 = Image.open(os.path.join(self.img_dir, self.data_addr[idx][1]))

        # img transformation
        transform = self.transform_img()
        image1 = transform(image1)
        image2 = transform(image2)
        image_pair = torch.cat((image1, image2), dim=0).float()  # The dimension 0 is the RGB channel

        if self.action:
            action = np.array(self.data_addr[idx][2:4], dtype=np.float)
            direction = self.direction2onehot(self.data_addr[idx][-1])
            return (image_pair, direction), action
        else:
            rel_trans = np.array(self.data_addr[idx][2:5], dtype=np.float)  # query base query base
            if self.mode == "test":
                # gt_pose = np.array(self.data_addr[idx][5:19], dtype = np.float)
                return image_pair, rel_trans #, gt_pose
            else:
                return image_pair, rel_trans

    def transform_img(self):
        if self.mode == "train":
            transform = tv.transforms.Compose([
                tv.transforms.Resize(256),  # fix the ratio and keep the smaller edge to 224 according to Relative **
                tv.transforms.RandomCrop(self.image_size),
                tv.transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4),  # randomly change hue, contrast illumination
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.440, 0.435, 0.381],
                                        std=[0.011, 0.011, 0.013])])
        else:
            transform = tv.transforms.Compose([
                tv.transforms.CenterCrop(self.image_size),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.440, 0.435, 0.381],
                                        std=[0.011, 0.011, 0.013])])
        return transform

    def direction2onehot(self, direction):
        '''
        +x: indicate function(0), +y: indicate function(1),
        -x: indicate function(2), -y: indicate function(3)

        '''
        a = np.zeros(4, dtype = np.float)
        a[int(int(direction)/90)] = 1.0
        return a

if "__name__" == "__main__":

    a = DroneCRDataset(mode='train')
    b = torch.utils.data.DataLoader(a, batch_size=2, shuffle=False, pin_memory=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i, datas in enumerate(b):
        print(datas[0])
        if i == 1:
            break

    a1 = DroneOriHeiSet()
    b1 = torch.utils.data.DataLoader(a1, batch_size=1, shuffle=False, pin_memory=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for i, datas in enumerate(b1):
        print(datas)
        if i == 1:
            break

    trainval_test_split()

    a = np.zeros((1,2,3))
    b = np.swapaxes(a, 0,1)
    print(b.shape)


    # create dataset
    gt_action_data(mode = "train")
    gt_action_data(mode="test")
