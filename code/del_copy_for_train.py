# -*- coding=utf-8 -*-

import os
import shutil

source_split_bad_pic_path = '../data/data_split/bad'
source_split_good_pic_path = '../data/data_split/good'
source_aug_bad_pic_path = '../data/data_augment/bad'
source_aug_good_pic_path = '../data/data_augment/good'
target_bad_pic_path = '../data/data_for_train/bad'
target_good_pic_path = '../data/data_for_train/good'
if os.path.exists(target_bad_pic_path):
    shutil.rmtree(target_bad_pic_path)
if os.path.exists(target_good_pic_path):
    shutil.rmtree(target_good_pic_path)
os.makedirs(target_bad_pic_path)
os.makedirs(target_good_pic_path)

#1# 复制原始图片到目标位置
# 复制图片到目标bad文件夹下
for parent, _, files in os.walk(source_split_bad_pic_path):
    for file in files:
        # print(os.path.join(parent, file))
        shutil.copyfile(os.path.join(source_split_bad_pic_path, file), os.path.join(target_bad_pic_path, file))

# 复制图片到目标good文件夹下
for parent, _, files in os.walk(source_split_good_pic_path):
    for file in files:
        # print(os.path.join(parent, file))
        shutil.copyfile(os.path.join(source_split_good_pic_path, file), os.path.join(target_good_pic_path, file))

#2# 复制增强后的图片到目标位置
# 复制图片到目标bad文件夹下
for parent, _, files in os.walk(source_aug_bad_pic_path):
    for file in files:
        # print(os.path.join(parent, file))
        shutil.copyfile(os.path.join(source_aug_bad_pic_path, file), os.path.join(target_bad_pic_path, file))

# 复制图片到目标good文件夹下
for parent, _, files in os.walk(source_aug_good_pic_path):
    for file in files:
        # print(os.path.join(parent, file))
        shutil.copyfile(os.path.join(source_aug_good_pic_path, file), os.path.join(target_good_pic_path, file))
