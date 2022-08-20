import math
import cv2
import time
import os

from yolo_detection_v5 import *

ROOTDIR = r'D:\IndataLabs\AnalogGuagReading_yolov5/'
res_save_dir = ROOTDIR + '/res_save_dir2/'
os.makedirs(name=res_save_dir, mode=0o777, exist_ok=True)
gauge_weight = ROOTDIR + 'weight_files/gaugeDet_stage1.pt'
gauge_yaml = ROOTDIR + 'weight_files/needle.yaml'
meter_weight = ROOTDIR + 'weight_files/needleDet_stage2.pt'
src_path = r'D:\IndataLabs\GaugeDetection\yolov5\demoOeignal.mp4'


def detection_cleanup(prediction):
    '''
    This function is currently used for 2nd - stage model.
    It takes all detections done by model and remove duplicate detections of a given class.
    Args:
        meter_pred: prediction results from 2md -stage Model that may / may not
                    have duplicate detection for each class

    Returns:
        update_pred_lst : An updated list that only has
        dec_dict_npdated : dictionary of all cleaned up classes detected {cls_x:[coordinates, conf]}

    '''
    update_pred_lst = []
    det_class = {}
    for i in range(len(prediction)):
        cx1, cy1, cx2, cy2, confid, cls_n = prediction[i]
        if int(cls_n) in det_class.keys():
            if confid > det_class[int(cls_n)][-1]:
                det_class.update({int(cls_n): [int(cx1), int(cy1), int(cx2), int(cy2),confid]})
            else:
                pass
        else:
            det_class.update({int(cls_n):[int(cx1), int(cy1), int(cx2), int(cy2),confid]})

    for k in det_class.keys():
        val = det_class[k]
        val.extend([k])
        update_pred_lst.append(val)
    return update_pred_lst
def draw_detection(raw_img, prediction, colr, put_text= False):
    z = raw_img.copy()
    if isinstance(prediction, list):
        for i in range(len(prediction)):
            cx1, cy1, cx2, cy2, confid, cls_n = prediction[i]
            z = cv2.rectangle(z, (cx1,cy1), (cx2,cy2), color=colr, thickness=2)

    if isinstance(prediction, dict):
        for i in prediction.keys():
            if prediction[i] != None:
                cx1, cy1, cx2, cy2, confid = prediction[i]
                if put_text:
                    z = cv2.rectangle(z, (cx1, cy1), (cx2, cy2), color=colr, thickness=2)
                    z = cv2.putText(z, text=i, org=(cx1+5, cy1+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,0), thickness=1, lineType=cv2.LINE_AA)
                else:
                    z = cv2.rectangle(z, (cx1, cy1), (cx2, cy2), color=colr, thickness=2)
    return z
def cropped_pred_translate_orignal(prediction, orignal_pred):
    updated_lst = []
    for i in range(len(prediction)):
        nx1 = prediction[i][0] + orignal_pred[0]
        ny1 = prediction[i][1] + orignal_pred[1]
        nx2 = prediction[i][2] + orignal_pred[0]
        ny2 = prediction[i][3] + orignal_pred[1]
        tmp_lst = [nx1,ny1,nx2,ny2, prediction[i][4], prediction[i][5]]
        updated_lst.append(tmp_lst)
    return updated_lst
def detect_dic_update(prediction, det_dic):
    det_dic_c = det_dic.copy()
    for i in range(len(prediction)):
        cx1, cy1, cx2, cy2, confid, cls_n = prediction[i]
        if int(cls_n) == 1:
            det_dic_c.update({'A':[int(cx1), int(cy1), int(cx2), int(cy2), conf]})
        if int(cls_n) == 2:
            det_dic_c.update({'B':[int(cx1), int(cy1), int(cx2), int(cy2), conf]})
        if int(cls_n) == 3:
            det_dic_c.update({'C':[int(cx1), int(cy1), int(cx2), int(cy2), conf]})
        if int(cls_n) == 4:
            det_dic_c.update({'M':[int(cx1), int(cy1), int(cx2), int(cy2), conf]})
    return det_dic_c
def get_center_rect(prediction):
    cx1 = int((prediction[0]+prediction[2])/2)
    cy1 = int((prediction[1]+prediction[3])/2)
    return (cx1,cy1)
def get_slope(pt1,pt2):
    '''
    this function calculates the slope of a Line.
    slope of a Line (m) = (y2-y1)/(x2-x1)
    Args:
        pt1 : in our case let pt1 be the center point of needle
        pt2 : let p2 be the other points like --> A, B, M
    Returns:
        Slope
    '''
    slope = round((pt2[1]-pt1[1])/(pt2[0]-pt1[0]), 2)
    return slope
def getAngle(pointsList):
    '''
    calculates Angle using slope of 2 lines.
    theta = tan-1((m2 - m1)/(1 + m1*m2))
    Args:
        pointsList : A list or dict of at Least 3 points [pt1, pt2, pt3]
        pt1 ---> point of center C
        pt2 ---> point of Meter M
        pt3 ---> point of Min/Max position : A or B
    Returns:
        Angle in Degrees
    '''
    pt1, pt2, pt3 = pointsList
    m1 = get_slope(pt1, pt2)
    m2 = get_slope(pt1, pt3)
    angR = math.atan((m2-m1)/(1+(m1*m2)))
    angD = math.degrees(angR)
    return angD
def meter_deflection_calculate(all_detections_dict, inp_image):

    imgd = inp_image.copy()
    angMin = None
    angMax = None
    center_dict = {}
    # --- STEP 1 : get center point of All detections Made ----------------
    for i in all_detections_dict.keys():
        val = all_detections_dict[i]
        if val is not None:
            center_dict.update({i: get_center_rect(val)})
    # ---------------------------------------------------------------------

    # ---------- STEP 2 : Calculating Angle using centers indentified above -------------------------
    # Note : We need 3 points atleast to draw 2 Lines , here  C, M is cumpolsory and A/ B can be optional
    if ('C' in center_dict.keys()) and ('M' in center_dict.keys()) and (('A' in center_dict.keys()) or ('B' in center_dict.keys())):
        # Draw line between center and meter - pointer
        imgd = cv2.arrowedLine(imgd, pt1=center_dict['C'], pt2=center_dict['M'],color=(0,0,255),thickness=2)

        if ('A' in center_dict.keys()) and ('B' in center_dict.keys()):
            imgd = cv2.arrowedLine(imgd, pt1=center_dict['C'], pt2=center_dict['A'], color=(0, 0, 0), thickness=2)
            angMin = getAngle([center_dict['C'], center_dict['M'], center_dict['A']])
            textA = 'angle from A : ' + str(round(angMin,2))
            imgd = cv2.putText(imgd, text=textA, org=(center_dict['BBox'][0], center_dict['BBox'][1]-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 255), thickness=1, lineType=cv2.LINE_AA)

            imgd = cv2.arrowedLine(imgd, pt1=center_dict['C'], pt2=center_dict['B'], color=(0, 0, 0), thickness=2)
            angMax = getAngle([center_dict['C'], center_dict['M'], center_dict['B']])
            textB = 'angle from B : ' + str(round(angMax, 2))
            imgd = cv2.putText(imgd, text=textB, org=(center_dict['BBox'][0], center_dict['BBox'][1] + 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 255), thickness=1, lineType=cv2.LINE_AA)

        else:
            try:
                imgd = cv2.arrowedLine(imgd, pt1=center_dict['C'], pt2=center_dict['A'], color=(0, 0, 0), thickness=2)
                angMin = getAngle([center_dict['C'], center_dict['M'], center_dict['A']])
                textA = 'angle from A : ' + str(round(angMin,2))
                imgd = cv2.putText(imgd, text=textA, org=(center_dict['BBox'][0], center_dict['BBox'][1] - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 255), thickness=1, lineType=cv2.LINE_AA)
            except:
                    imgd = cv2.arrowedLine(imgd, pt1=center_dict['C'], pt2=center_dict['B'], color=(0, 0, 0), thickness=2)
                    angMax = getAngle([center_dict['C'], center_dict['M'], center_dict['B']])
                    textB = 'angle from B : ' + str(round(angMax,2))
                    imgd = cv2.putText(imgd, text=textB, org=(center_dict['BBox'][0], center_dict['BBox'][1] + 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 255), thickness=1, lineType=cv2.LINE_AA)

    else:
        text = 'Deflection can not be calculated'
        print(text)
        imgd = cv2.putText(imgd, text=text, org=center_dict['BBox'], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    return center_dict, inp_image, imgd, angMin, angMax


gauge = YoloDetection(weight_path=gauge_weight, yaml_path=gauge_yaml)
meter = YoloDetection(weight_path=meter_weight, yaml_path=gauge_yaml)

run_flag = True
ret_time_ctr = time.time()

cap = cv2.VideoCapture(src_path)
ret, frame = cap.read()
ctr = 1
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(ROOTDIR + 'video.mp4',fourcc,20,(frame.shape[1], frame.shape[0]))

while cap.isOpened() and (run_flag is True): #and ctr < 8:
    ret, frame = cap.read()
    draw_img_orig = None
    if ret:
        all_detections_in_frame = []
        ret_time_ctr = time.time()
        framec = frame.copy()       # H, W, C

        imgsz = check_img_size(frame.shape[:2], s=gauge.stride)  # check image size
        frame = cv2.resize(frame, (imgsz[1], imgsz[0]))

        # shape format: Opencv ---> H, W, C  # shape format: torch ---> b:batch, ch, h, w
        # we rearrange the position of C in opencv --> transpose to shape as (Ch, H, W)
        frame = frame.transpose((2, 0, 1))
        pred = gauge.predict(image=frame, num_det=5, conf_thresh=0.9)
        pred = gauge.scale_prediction(prediction=pred, output_shape_tuple=framec.shape[:2], input_shape_tuple=gauge.torch_image_tensor[2:])

        for i in range(len(pred)):
            x1, y1, x2, y2, conf, cls = pred[i]
            if int(cls) == 0 and conf > 0.9:
                all_detections_dict = {key: None for key in gauge.names}
                all_detections_in_frame.append([int(x1), int(y1), int(x2), int(y2), conf, int(cls)])
                all_detections_dict['BBox'] = [int(x1), int(y1), int(x2), int(y2), conf]

                # cropping frame for 2md model input
                crop_gauge_frame = framec[int(y1):int(y2), int(x1):int(x2)]
                crop_frame = crop_gauge_frame.copy()

                imgsz = check_img_size(crop_frame.shape[:2], s=meter.stride)
                crop_frame = cv2.resize(crop_frame, (imgsz[1], imgsz[0]))
                crop_frame = crop_frame.transpose((2, 0, 1))
                pred_met = meter.predict(image=crop_frame, num_det=5, conf_thresh=0.5)
                pred_met = meter.scale_prediction(prediction=pred_met, output_shape_tuple=crop_gauge_frame.shape[:2], input_shape_tuple=meter.torch_image_tensor[2:])
                pred_met = detection_cleanup(pred_met)

                # tranlating detection to orignal image
                new_pred_met = cropped_pred_translate_orignal(prediction=pred_met, orignal_pred=[int(x1), int(y1), int(x2), int(y2)])
                all_detections_in_frame.extend(new_pred_met)

                # dictionary update
                all_detections_dict = detect_dic_update(prediction=all_detections_in_frame, det_dic=    all_detections_dict)
                draw_img_orig = draw_detection(framec, all_detections_dict, colr=(0, 225, 0), put_text=True)
                centeroid_center, img_b_def, draw_img_orig, angA, angB = meter_deflection_calculate(all_detections_dict, inp_image=draw_img_orig)
                cv2.imwrite(res_save_dir+str(ctr)+'.jpg', draw_img_orig)

    else:
        if time.time() - ret_time_ctr > 20:
            run_flag = False

    video.write(draw_img_orig)
    ctr += 1

video.release()



