# -*- coding=utf-8 -*-
'''
讲所有xml文件copy到./data/xml文件夹下
'''
import os
import shutil
from tqdm import tqdm

ori_data_root = '../data'
target_xml_path = '../data/xml'
if os.path.exists(target_xml_path):
    shutil.rmtree(target_xml_path)
os.makedirs(target_xml_path)

for parent, _, files in os.walk(ori_data_root):
    # 跳过test数据以及无瑕疵即正常的数据
    if 'test' in parent.split('_') or '正常' in parent.split('/') or 'xml' in parent.split('/'):
        continue
    for file in tqdm(files):
        file_name = os.path.join(parent, file)
        if file_name[-3:] == 'xml':
	        shutil.copyfile(file_name, os.path.join(target_xml_path, file))
