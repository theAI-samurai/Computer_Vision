import os
import cv2
from darknet import *
import  numpy as np

net = model_load(cfgPath='cfg/yolov3_full_tables.cfg', wgtPath='cfg/yolov3_full_tables_last_3062.weights')
table_detect =  ObjectDetection(dataPath='cfg/tables.data', netwrk= net)

# test data
test_data = []
for f in os.listdir('images'):
    if f.endswith('.jpg'):
        temp_name = f.split('.')[0]+'.xml'
        if os.path.isfile('images/'+temp_name):
            pass
        else:
            #print(f)
            test_data.append('images/'+f)

for f in os.listdir('test'):
    test_data.append('test/' + f)



def get_index(lst, val):
    return lst.index(val)


def correction_borders(xmin_lst, ymin_lst, xmax_lst, ymax_lst):

    print(xmin_lst)
    print(xmax_lst)
    print(ymin_lst)
    print(ymax_lst)
    ind = get_index(xmin_lst, min(xmin_lst))
    if cls_lst[ind] == 'borderless' or 'bordered':
        # check y-min and readjust
        if ymin_lst[ind] < 0:
            # find new y-min
            for row in range(img.shape[1]):
                if not np.all(img[row,:,0]==255):
                    ymin_lst[ind] = row-2
                    return xmin_lst, ymin_lst, xmax_lst, ymax_lst



for img_p in test_data:
    xmin_lst = []
    xmax_lst = []
    ymin_lst = []
    ymax_lst = []
    cls_lst = []

    img = cv2.imread(img_p)
    res = table_detect.detect(img_p)
    img_ = img.copy()

    if len(res) != 0:
        for ele in res:
            cls, confi, coordi = ele
            xmin,ymin,xmax,ymax = bbox2points(coordi)
            xmin_lst.append(xmin)
            ymin_lst.append(ymin)
            xmax_lst.append(xmax)
            ymax_lst.append(ymax)
            cls_lst.append(cls.decode())

            img_ = cv2.rectangle(img_, (xmin, ymin), (xmax,ymax),(255, 255, 0), 2)

    #xmin_lst, ymin_lst, xmax_lst, ymax_lst = correction_borders(xmin_lst, ymin_lst, xmax_lst, ymax_lst)


    #img2_ = img.copy()
    #for i in range(len(xmin_lst)):
    #    img2_ = cv2.rectangle(img2_, (xmin_lst[i], ymin_lst[i]), (xmax_lst[i],xmax_lst[i]),(255, 0, 0), 2)


    cv2.imwrite('results/'+img_p.split('/')[1], img_)

    #cv2.imshow('orignal', img)
    #cv2.imshow('detected', img_)
    #cv2.imshow('improved', img2_)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()





