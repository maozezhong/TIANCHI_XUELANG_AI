# -*- coding=utf-8 -*-
'''
根据原始数据data
得到分类的数据data_split
data_split文件夹下包括两个文件夹
    - good， 无瑕疵图片
    - bad， 有瑕疵图片
'''

import os
import shutil
from tqdm import tqdm

ori_data_root = '../data'
target_bad_root_path = '../data/data_split/bad'   #存放有瑕疵图片的根目录
target_good_root_path = '../data/data_split/good' #存放无瑕疵图片的根目录
if os.path.exists(target_bad_root_path):
    shutil.rmtree(target_bad_root_path)
if os.path.exists(target_good_root_path):
    shutil.rmtree(target_good_root_path)
os.makedirs(target_bad_root_path)
os.makedirs(target_good_root_path)

for parent, _, files in os.walk(ori_data_root):
    # 跳过测试数据
    if 'test' in parent.split('_') or 'data_split' in parent.split('/'):
        continue
    print(parent)
    for file in tqdm(files):
        file_name = os.path.join(parent, file)
        if file_name[-3:] == 'jpg': #只拷贝图片
            temp_name = file_name.split('/')[-2]	#比如'正常'
            if temp_name == '正常':
                target_pic_path = os.path.join(target_good_root_path, file)
            else:
                target_pic_path = os.path.join(target_bad_root_path, file)
            shutil.copyfile(file_name, target_pic_path)