'''
This Script takes a YOLO format txt as input crop an object of a class (here its class 0)
then, translates coordinates of all objects inside the cropped image, and write the same to a new Txt

'''
import math
import os
import cv2

image_folder = r'D:\IndataLabs\GaugeDetection\data\tmp\images/'
label_folder = r'D:\IndataLabs\GaugeDetection\data\tmp\labels/'
save_folder = r'D:\IndataLabs\GaugeDetection\data\tmp\crop_image/'
os.makedirs(save_folder, mode=0o777, exist_ok=True)


def convert_yolo_cood_XY_format(lst, img_W, img_H):
    cx = float(lst[0])
    cy = float(lst[1])
    cw = float(lst[2])
    ch = float(lst[3])

    X1 = int(cx * img_W) - (cw * img_W / 2)
    X2 = int(cx * img_W) + (cw * img_W / 2)
    Y1 = int(cy * img_H) - (ch * img_H / 2)
    Y2 = int(cy * img_H) + (ch * img_H / 2)

    return int(X1), int(Y1), int(X2), int(Y2)


def crop_object_save(image, X1, Y1, X2, Y2, savepath):
    crop_image = image[Y1:Y2, X1:X2]
    cv2.imwrite(savepath, crop_image)
    return crop_image


def translate_coordinates_after_crop(main_object_coord, sub_object_coor, img=None):
    x1 = sub_object_coor[0][0]-main_object_coord[0][0]
    y1 = sub_object_coor[0][1]-main_object_coord[0][1]
    x2 = sub_object_coor[0][2]-main_object_coord[0][0]
    y2 = sub_object_coor[0][3]-main_object_coord[0][1]
    if img != None:
        cv2.rectangle(img, )
    return [x1,y1,x2,y2]


def coordinate_to_yoloformat(cropW, cropH, trns_lst):

    yolo_x = round((trns_lst[0] + trns_lst[2])/ (2*cropW), 3)
    yolo_y = round((trns_lst[1] + trns_lst[3])/ (2*cropH), 3)
    yolo_w = round((trns_lst[2] - trns_lst[0])/(cropW), 3)
    yolo_h = round((trns_lst[3] - trns_lst[1])/(cropH), 3)
    return [str(yolo_x), str(yolo_y), str(yolo_w), str(yolo_h)]


for f in os.listdir(image_folder):
    if f.endswith('.png'):
        nm = f.split('.')[0]
        img = cv2.imread(image_folder+f)
        if os.path.exists(label_folder+nm+'.txt'):
            print(nm, img.shape)                    # 1024 - 1024
            txtf = open(label_folder+nm+'.txt', 'r')
            txtread = txtf.readlines()

            dial = []
            C = []
            A = []
            B = []
            M = []
            C_H = None
            C_W = None
            crop_txt_list = []
            c_img = None

            for item in txtread:
                item = item.strip()
                itemlist = item.split(' ')
                if itemlist[0] == '0':
                    yolocoor = itemlist[1:]
                    # converting YOLO format to COORDINATE system
                    x1,y1,x2,y2 = convert_yolo_cood_XY_format(lst=yolocoor, img_W=1024, img_H=1024)
                    dial.append([x1,y1,x2,y2])
                    save_path = save_folder + nm + '.jpg'
                    # CROP SEGMENT of IMAGE
                    # c_img = crop_object_save(image=img, X1=x1, Y1=y1, X2=x2, Y2=y2, savepath=save_path)
                    try:
                        c_img = cv2.imread(save_path)
                    except:
                        c_img = crop_object_save(image=img, X1=x1, Y1=y1, X2=x2, Y2=y2, savepath=save_path)
                    try:
                        C_H, C_W, C_chan = c_img.shape   # was coming as none
                    except:
                        pass
                else:
                    if itemlist[0] == '1':
                        yolocoor = itemlist[1:]
                        x1, y1, x2, y2 = convert_yolo_cood_XY_format(lst=yolocoor, img_W=1024, img_H=1024)
                        A.append([x1, y1, x2, y2])
                    if itemlist[0] == '2':
                        yolocoor = itemlist[1:]
                        x1, y1, x2, y2 = convert_yolo_cood_XY_format(lst=yolocoor, img_W=1024, img_H=1024)
                        B.append([x1, y1, x2, y2])
                    if itemlist[0] == '3':
                        yolocoor = itemlist[1:]
                        x1, y1, x2, y2 = convert_yolo_cood_XY_format(lst=yolocoor, img_W=1024, img_H=1024)
                        C.append([x1, y1, x2, y2])
                    if itemlist[0] == '4':
                        yolocoor = itemlist[1:]
                        x1, y1, x2, y2 = convert_yolo_cood_XY_format(lst=yolocoor, img_W=1024, img_H=1024)
                        M.append([x1, y1, x2, y2])

            At = translate_coordinates_after_crop(dial, A)
            Bt = translate_coordinates_after_crop(dial, B)
            Ct = translate_coordinates_after_crop(dial, C)
            Mt = translate_coordinates_after_crop(dial, M)

            # convert translated Coordinates to YOLO
            if C_W is not None and C_H is not None:
                # drawing bboxes
                # for i in range(len(At)):
                #     c_img = cv2.rectangle(c_img, (At[0], At[1]), (At[2], At[3]), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
                #     c_img = cv2.rectangle(c_img, (Bt[0], Bt[1]), (Bt[2], Bt[3]), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
                #     c_img = cv2.rectangle(c_img, (Ct[0], Ct[1]), (Ct[2], Ct[3]), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
                #     c_img = cv2.rectangle(c_img, (Mt[0], Mt[1]), (Mt[2], Mt[3]), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
                # cv2.imwrite(save_folder + nm + '_draw.jpg', c_img)

                At_yolo = ['1']
                Bt_yolo = ['2']
                Ct_yolo = ['3']
                Mt_yolo = ['4']
                At_yolo.extend(coordinate_to_yoloformat(cropW=C_W, cropH=C_H, trns_lst=At))
                Bt_yolo.extend(coordinate_to_yoloformat(cropW=C_W, cropH=C_H, trns_lst=Bt))
                Ct_yolo.extend(coordinate_to_yoloformat(cropW=C_W, cropH=C_H, trns_lst=Ct))
                Mt_yolo.extend(coordinate_to_yoloformat(cropW=C_W, cropH=C_H, trns_lst=Mt))

                crop_txt_list.append(' '.join(At_yolo)+'\n')
                crop_txt_list.append(' '.join(Bt_yolo)+'\n')
                crop_txt_list.append(' '.join(Ct_yolo)+'\n')
                crop_txt_list.append(' '.join(Mt_yolo)+'\n')

                with open(save_folder+nm+'.txt', 'w') as out:
                    out.writelines(crop_txt_list)



"""

# ------------------- TRANSLATING THE COORDINATES FROM CROPPED --------------------------
img_orig = r'D:\IndataLabs\GaugeDetection\data\tmp\images/scale_0_meas_0.png'
img_crop = r'D:\IndataLabs\GaugeDetection\data\tmp\crop_image/scale_0_meas_0.jpg'
txt_file = r'D:\IndataLabs\GaugeDetection\data\tmp\labels/scale_0_meas_0.txt'

txtf = open(txt_file, 'r')
txtread = txtf.readlines()

imgo = cv2.imread(img_orig)
imgc = cv2.imread(img_crop)

for item in txtread:
    item = item.strip()
    itemlist = item.split(' ')
    itemlist = itemlist[1 :]
    print(itemlist)
    x1, y1, x2, y2 = convert_yolo_cood_XY_format(lst=itemlist, img_W=1024, img_H=1024)
    print(x1, y1, x2, y2)

#"""