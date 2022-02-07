import os
import xml.etree.ElementTree as ET
import csv

dirpath = 'saved_xml'         # Path to find XML Files
newdir = 'saved_yolo'     # Path to Save TXT Files
coco_list_file = 'coco_xml_to_yolo.csv'  # File where coco list is

reader = csv.reader(open(coco_list_file, 'r'))
name_id = {}
id_name = {}
for row in reader:
   k, v,w = row
   id_name[v] = w
   name_id[w] = v


if not os.path.exists(newdir):
    os.makedirs(newdir)

for fp in os.listdir(dirpath):
    root = ET.parse(os.path.join(dirpath, fp)).getroot()

    xmin, ymin, xmax, ymax = 0, 0, 0, 0
    sz = root.find('size')

    width = float(sz[0].text)
    height = float(sz[1].text)

    filename = fp.split('.')[0]
    for child in root.findall('object'):  # find all the boxes in the picture
        # print(child.find('name').text)

        sub = child.find('bndbox')  # find the dimension values ​​and reading frame
        label = child.find('name').text
        id = int(name_id[label])-1
        xmin = float(sub[0].text)
        ymin = float(sub[1].text)
        xmax = float(sub[2].text)
        ymax = float(sub[3].text)
        # yolov3 label format is converted into the required normalized to (0-1):
        try:
            x_center = (xmin + xmax) / (2 * width)
            y_center = (ymin + ymax) / (2 * height)
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
        except ZeroDivisionError:
            print(filename, 'the width in question')

        with open(os.path.join(newdir, filename + '.txt'), 'a+') as f:
            f.write(' '.join([str(id), str(x_center), str(y_center), str(w), str(h) + '\n']))

print('ok')