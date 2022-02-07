import os
import pandas as pd
import csv
import shutil

coco_txt_files_path = r'D:/Dataset/COCO/train2014/train2014/'
coco_list_file = r'D:/my_repos/Computer_Vision/yolo_scripts/annotation_conversion_scripts/coco_xml_to_yolo.csv'  # File where coco list is

# creating a directory where required data can be copied
new_dir = r'D:\Dataset\COCO\train2014/req_data/'
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

reader = csv.reader(open(coco_list_file, 'r'))
name_id = {}
id_name = {}
for row in reader:
   k, v,w = row
   id_name[v] = w
   name_id[w] = v

del [k,v,w, row]

ctr = 0
df = pd.DataFrame(columns=['file_name', 'data', 'data_req'])
f_name = []
data = []
for f in os.listdir(coco_txt_files_path):
    if f.endswith('.txt') and f != 'classes.txt':
        f_name.append(coco_txt_files_path+f)
        file1 = open(coco_txt_files_path+f, 'r')
        Lines = file1.readlines()
        data.append(Lines)

df['file_name'] = f_name
df['data'] = data


for i in range(len(df)):
    new_data = []
    temp = df.loc[i, 'data']
    temp_nam = df.loc[i,'file_name']
    for j in range(len(temp)):
        temp_data = int(temp[j].split(' ')[0])
        nam = id_name[str(temp_data + 1)]
        if nam not in ('car', 'truck', 'bus', 'motorcycle'):
            pass
        else:
            new_data.append(temp[j])
    df.loc[i, 'data_req'] = new_data

    # moving files now
    if len(df.loc[i, 'data_req']) > 0:
        old_image_file_path = temp_nam.split('.')[0]+'.jpg'
        new_image_file_path = new_dir + ((temp_nam.split('/')[-1]).split('.')[0]+'.jpg')
        old_txt_file_path = temp_nam
        new_txt_file_path = new_dir + (temp_nam.split('/')[-1])
        shutil.copyfile(old_image_file_path, new_image_file_path)
        shutil.copyfile(old_txt_file_path, new_txt_file_path)

        # writing new data to txt file
        with open(new_txt_file_path, "w") as output:
            output.writelines(df.loc[i, 'data_req'])

