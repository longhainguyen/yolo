import xml.etree.ElementTree as ET
from glob import glob
import pandas as pd
from shutil import copyfile
import pandas as pd
from sklearn import preprocessing, model_selection
import os
import cv2
import torch

import config



def load_blood_cell_detection_csv():
    annotations = sorted(glob('./BCCD_Dataset/BCCD/Annotations/*.xml'))

    df = []
    cnt = 0
    for file in annotations:
        prev_filename = file.split('/')[-1].split('.')[0] + '.jpg'
        filename = str(cnt) + '.jpg'
        row = []
        parsedXML = ET.parse(file)
        for node in parsedXML.getroot().iter('object'):
            blood_cells = node.find('name').text
            xmin = int(node.find('bndbox/xmin').text)
            xmax = int(node.find('bndbox/xmax').text)
            ymin = int(node.find('bndbox/ymin').text)
            ymax = int(node.find('bndbox/ymax').text)

            row = [prev_filename, filename, blood_cells, xmin, xmax, ymin, ymax]
            df.append(row)
        cnt += 1

    data = pd.DataFrame(df, columns=['prev_filename', 'filename', 'cell_type', 'xmin', 'xmax', 'ymin', 'ymax'])

    data[['prev_filename','filename', 'cell_type', 'xmin', 'xmax', 'ymin', 'ymax']].to_csv('./BCCD_Dataset/blood_cell_detection.csv', index=False)

def read_blood_cell_dectection_csv():
    img_width = 640
    img_height = 480

    def width(df):
        return int(df.xmax - df.xmin)
    def height(df):
        return int(df.ymax - df.ymin)
    def x_center(df):
        return int(df.xmin + (df.width/2))
    def y_center(df):
        return int(df.ymin + (df.height/2))
    def w_norm(df):
        return df/img_width
    def h_norm(df):
        return df/img_height

    df = pd.read_csv('./BCCD_Dataset/blood_cell_detection.csv')

    le = preprocessing.LabelEncoder()
    le.fit(df['cell_type'])
    print(le.classes_)
    labels = le.transform(df['cell_type'])
    df['labels'] = labels

    df['width'] = df.apply(width, axis=1)
    df['height'] = df.apply(height, axis=1)

    df['x_center'] = df.apply(x_center, axis=1)
    df['y_center'] = df.apply(y_center, axis=1)

    df['x_center_norm'] = df['x_center'].apply(w_norm)
    df['width_norm'] = df['width'].apply(w_norm)

    df['y_center_norm'] = df['y_center'].apply(h_norm)
    df['height_norm'] = df['height'].apply(h_norm)
    return df

def split_dataset_csv():
    df = read_blood_cell_dectection_csv()
    df_train, df_valid = model_selection.train_test_split(df, test_size=0.1, random_state=13, shuffle=True)
    return df_train, df_valid

def load_data_to_train():
    file_list = os.listdir(config.DATA_IMG_DIR)
    x_train = torch.zeros((len(file_list), config.img_size_H, config.img_size_W, 3))
    for idx ,file_name in enumerate(file_list):
        x_train[idx] = torch.from_numpy(cv2.imread(f"{config.DATA_IMG_DIR}{file_name}"))
    x_train = x_train.permute(0, 3, 1, 2)/255

    file_list_label = os.listdir(config.DATA_LABLE_DIR)
    y_train = torch.zeros((len(file_list), config.out_W, config.out_H, config.nclass + 5))
    for idx ,file_name in enumerate(file_list_label):
        with open(f"{config.DATA_LABLE_DIR}{file_name}", 'r') as file:
        # Đọc nội dung của tệp và gán nó cho một biến
            file_content = file.read()
            lines = file_content.splitlines()
            for line in lines:
                values = line.split()
                class_label, x_center_norm, y_center_norm, width_norm, height_norm = map(float, values)
                cl = torch.zeros((3))
                cl[int(class_label)] = 1
                # x_center, y_center, w, h = int(x_center_norm * config.img_size_W),  int(y_center_norm * config.img_size_H), int(width_norm * config.img_size_W), int(height_norm * config.img_size_H)
                # index của object trên ma trận ô vuông 10x8
                x_idx, y_idx = int(x_center_norm * config.out_W), int(y_center_norm * config.out_H)
                y_tensor = torch.zeros(config.nclass + 5)
                y_tensor[0] = 1
                y_tensor[1] = x_center_norm
                y_tensor[2] = y_center_norm
                y_tensor[3] = width_norm
                y_tensor[4] = height_norm
                y_tensor[5:] = cl
                y_train[idx][x_idx][y_idx] = y_tensor
    return x_train, y_train







