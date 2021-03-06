import os
import pandas as pd
import xml.etree.ElementTree as ET
import cv2
import json

# XML Template Format File in which to save
template_path = 'pascal_voc_template.xml'
# COCO annotated JSON File
annotations_path = r'D:\Dataset\COCO\annotations_trainval2014\annotations\instances_train2014.json'
# COCO Category List CSV File
df = pd.read_csv('coco.csv')
df.set_index('id', inplace=True)
# Image Paths
image_folder = r'D:\Dataset\COCO\train2014/train2014'    # Path where COCO images are placed
savepath = 'saved_xml'                                   # Path where XML Files must be saved


def write_to_xml(image_name,img_name, image_dict, data_folder, save_folder, xml_template=template_path):
    # get bboxes
    bboxes = image_dict[image_name]

    # read xml file
    tree = ET.parse(xml_template)
    root = tree.getroot()

    # modify
    folder = root.find('folder')
    folder.text = 'Annotations'

    fname = root.find('filename')
    fname.text = image_name.split('.')[0]

    src = root.find('source')
    database = src.find('database')
    database.text = 'COCO2014'

    # size
    img = cv2.imread(os.path.join(data_folder, img_name))
    h, w, d = img.shape

    size = root.find('size')
    width = size.find('width')
    width.text = str(w)
    height = size.find('height')
    height.text = str(h)
    depth = size.find('depth')
    depth.text = str(d)

    for box in bboxes:
        # append object
        obj = ET.SubElement(root, 'object')

        name = ET.SubElement(obj, 'name')
        name.text = box[0]

        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'

        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = str(0)

        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = str(0)

        bndbox = ET.SubElement(obj, 'bndbox')

        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(int(box[1]))

        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(int(box[2]))

        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(int(box[3]))

        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(int(box[4]))

    # save .xml to anno_path
    anno_path = os.path.join(save_folder, img_name.split('.')[0] + '.xml')
    print(anno_path)
    tree.write(anno_path)


if not os.path.exists(savepath):
    os.makedirs(savepath)

# read in .json format
with open(annotations_path, 'rb') as file:
    doc = json.load(file)

# get annotations
annotations = doc['annotations']

# iscrowd allowed? 1 for ok, else set to 0
iscrowd_allowed = 1

# initialize dict to store bboxes for each image
image_dict = {}

# loop through the annotations in the subset
for anno in annotations:
    # get annotation for image name
    image_id = anno['image_id']
    image_name = '{0:012d}.jpg'.format(image_id)

    # get category
    category = df.loc[anno['category_id']]['name']

    # add as a key to image_dict
    if not image_name in image_dict.keys():
        image_dict[image_name] = []

    # append bounding boxes to it
    box = anno['bbox']
    # since bboxes = [xmin, ymin, width, height]:
    image_dict[image_name].append([category, box[0], box[1], box[0] + box[2], box[1] + box[3]])

# generate .xml files
for image_name in image_dict.keys():
    img_name= 'COCO_train2014_'+image_name
    write_to_xml(image_name,img_name, image_dict, image_folder, savepath)
    print('generated for: ', image_name)