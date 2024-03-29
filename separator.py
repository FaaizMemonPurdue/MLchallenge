import os
import pandas as pd
from constants import name_map

def separate(img_folder, label_path):
    # sep_parent = os.path.join(os.path.dirname(img_folder), "separate_" + os.path.basename(img_folder))
    # if not os.path.exists(sep_parent):
    #     print('creating {sep_parent}')
    #     os.makedirs(sep_parent)
    labels = pd.read_csv(label_path)
    for index, row in labels.iterrows():
    # access data using column names
        print(row['Category'], row['File Name'])
        if index > 10:
            break
    files = os.listdir(img_folder)
    i = 0
    for file in files:
        num_label = 
        if i > 15:
            break
    


#     # o_name = os.path.basename(train_folder)
#     # s_name = "bar_" + folder_name

separate("/home/ubuntu/fs6/data/train_small", "purdue-face-recognition-challenge-2024/train_small.csv")

# print()