import os
import pandas as pd
from constants import name_strind
import shutil
import sys

def separate(src_folder, label_path):
    sep_parent = os.path.join(os.path.dirname(src_folder), "separate_" + os.path.basename(src_folder))
    if not os.path.exists(sep_parent):
        print(f'creating {sep_parent}')
        os.makedirs(sep_parent)
    labels = pd.read_csv(label_path)
    for index, row in labels.iterrows():
    # access data using column names
        fname = row['File Name']
        src_path = os.path.join(src_folder, fname)
        if not os.path.exists(src_path):
            print(f"{src_path} dne, stopping")
            break

        cname = row['Category']
        cnum = name_strind[cname]
        dest_folder = os.path.join(sep_parent, cnum)
        if not os.path.exists(dest_folder):
            print(f'creating {dest_folder}')
            os.makedirs(dest_folder)
        dest_path = os.path.join(dest_folder, fname)
        if not os.path.exists(dest_path):
            # print(f"placing {dest_path}")
            shutil.copy(src_path, dest_path)

if __name__ == '__main__':
    separate(str(sys.argv[1]), str(sys.argv[2]))

# print()