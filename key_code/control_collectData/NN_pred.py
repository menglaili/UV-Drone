import cv2
import os.path
import torch
import torch.utils.data as data
import torchvision as tv
from data_provider import DroneCRDataset
from models.NNutils import *
from utils.cumulative_err import *
from models.NNmodels import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def finddirection(direction):
    '''
    0,1,2,3,4: 0
    5,6,7: 90
    8,9,10,11,12: 180
    '''
    def dir2onehot(ori):
        a = np.zeros((1, 4), dtype = np.float)
        a[0, int(int(ori)/90)] = 1.0
        return a

    if direction<5:
        return dir2onehot(str(0))
    elif direction<8:
        return dir2onehot(str(90))
    else:
        return dir2onehot(str(180))



transform = tv.transforms.Compose([
    tv.transforms.ToPILImage(),
    tv.transforms.CenterCrop(224),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.440, 0.435, 0.381],
                            std=[0.011, 0.011, 0.013])])

def NNtrans0(cur_img, ref_img):
    DroneCorrNet0 = set_weight().to(device)
    DroneCorrNet0.eval()
    cur_img = np.swapaxes(cur_img, 0, 1)
    ref_img = np.swapaxes(ref_img, 0, 1)
    cur_img = transform(cur_img)
    ref_img = transform(ref_img)
    image_pair = torch.cat((cur_img, ref_img), dim=0)
    image_pair = image_pair[None, :, :, :]
    image = image_pair.to(device)
    with torch.no_grad():
        outputs = DroneCorrNet0(image)
    return np.squeeze(outputs.cpu().numpy())


DroneCorrNet1 = CorrCtrlNet() 
state_dict = torch.load('./output_model7final.ckpt')
DroneCorrNet1.load_state_dict(state_dict)
DroneCorrNet1.to(device).eval()
def NNtrans1(cur_img, ref_img, direction):
    # predict x-y
    cur_img = np.swapaxes(cur_img, 0, 1)
    ref_img = np.swapaxes(ref_img, 0, 1)
    cur_img = transform(cur_img)
    ref_img = transform(ref_img)
    image_pair = torch.cat((cur_img, ref_img), dim=0)
    image_pair = image_pair[None, :, :, :]
    image = image_pair.float().to(device)
    direction = torch.from_numpy(finddirection(direction)).float().to(device)
    with torch.no_grad():
        outputs = DroneCorrNet1(image, direction)
    return np.squeeze(outputs.cpu().numpy())


def NNtrans01(cur_img, ref_img):
    # predict x-y
    cur_img = np.swapaxes(cur_img, 0, 1)
    ref_img = np.swapaxes(ref_img, 0, 1)
    cur_img = transform(cur_img)
    ref_img = transform(ref_img)
    image_pair = torch.cat((cur_img, ref_img), dim=0)
    image_pair = image_pair[None, :, :, :]
    image = image_pair.to(device)
    with torch.no_grad():
        outputs = DroneCorrNet1(image)
    return np.squeeze(outputs.cpu().numpy())



