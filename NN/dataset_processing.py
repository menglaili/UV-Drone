import os
import csv
import numpy as np 


train_data_dir = './train_data'
train_img_dir = train_data_dir + '/img'
train_meta_dir = train_data_dir + '/metadata'
train_ctrl_dir = train_data_dir + '/ctrl'

scenes = [f.train_ctrl_dir for f in os.scandir(train_ctrl_dir) if f.is_dir()]
scenes_num = len(scenes)

eval_list = [3]  

def minmax(vec, maxval, minval):
    return (vec-minval)/(maxval - minval)

ctrl_vec_list = []
# create a csv to document the file
with open('train.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=['ctrl', 'img_exp', 'img_cur', 'meta_exp', 'meta_cur'])    
    writer.writeheader()

    for scene_idx in range(scenes_num):
        if scene_idx in eval_list:
            continue
        filename = scenes[scene_idx][len(train_ctrl_dir)+1:]
        ctrl_subdir = scenes[scene_idx]
        img_subdir = os.path.join(train_img_dir, filename)
        meta_subdir = os.path.join(train_meta_dir, filename)
        sub_ctrl_dir = os.listdir(ctrl_subdir)
        # sub_img_dir = os.listdir(img_subdir)
        # sub_meta_dir = os.listdir(meta_subdir)
        for i in range(len(sub_ctrl_dir)):
            # calculate the vector of ctrl
            ctrl = np.loadtxt(os.path.join(ctrl_subdir, sub_ctrl_dir[i]))
            ctrl_vec_list.append(np.sum(ctrl, 0)[-4:]) # how to normalize the label to 0-1?? use minmax
            writer.writerow(
            {'ctrl' : os.path.join(ctrl_subdir, str(i)+'vec.txt'), 'img_exp': os.path.join(img_subdir, str(i)+'after.jpg'), \
            'img_cur': os.path.join(img_subdir, str(i)+'err.jpg'), 'meta_exp': os.path.join(meta_subdir, str(i)+'after.txt'), \
            'meta_cur': os.path.join(meta_subdir, str(i)+'err.txt')})


with open('val.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=['ctrl', 'img_exp', 'img_cur', 'meta_exp', 'meta_cur'])    
    writer.writeheader()
    for scene_idx in range(scenes_num):
        if scene_idx in eval_list:
            filename = scenes[scene_idx][len(train_ctrl_dir)+1:]
            ctrl_subdir = scenes[scene_idx]
            img_subdir = os.path.join(train_img_dir, filename)
            meta_subdir = os.path.join(train_meta_dir, filename)
            sub_ctrl_dir = os.listdir(ctrl_subdir)
            # sub_img_dir = os.listdir(img_subdir)
            # sub_meta_dir = os.listdir(meta_subdir)
            for i in range(len(sub_ctrl_dir)):
                ctrl = np.loadtxt(os.path.join(ctrl_subdir, sub_ctrl_dir[i]))
                ctrl_vec_list.append(np.sum(ctrl, 0)[-4:]) # how to normalize the label to 0-1?? use minmax
                writer.writerow(
                {'ctrl' : os.path.join(ctrl_subdir, str(i)+'vec.txt'), 'img_exp': os.path.join(img_subdir, str(i)+'after.jpg'), \
                'img_cur': os.path.join(img_subdir, str(i)+'err.jpg'), 'meta_exp': os.path.join(meta_subdir, str(i)+'after.txt'), \
                'meta_cur': os.path.join(meta_subdir, str(i)+'err.txt')})


ctrl_vec_list = np.array(ctrl_vec_list)
maxval = np.max(ctrl_vec_list, 0)
minval = np.min(ctrl_vec_list, 0)
for scene_idx in range(scenes_num):
    ctrl_subdir = scenes[scene_idx]
    sub_ctrl_dir = os.listdir(ctrl_subdir)
    for i in range(len(sub_ctrl_dir)):
        ctrl = np.loadtxt(os.path.join(ctrl_subdir, sub_ctrl_dir[i]))
        np.savetxt(os.path.join(ctrl_subdir, str(i)+'vec.txt'), minmax(np.sum(ctrl, 0)[-4:], maxval, minval))



