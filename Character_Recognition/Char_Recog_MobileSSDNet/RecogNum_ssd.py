from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np
import os

class character_recognition():

    def __init__(self, folder_name):  #, ssd_model_path, wt_path, cls_path):
        self.m_p = folder_name + '/MobileNets_character_recognition.json'
        self.w_p = folder_name + '/License_character_recognition.h5'
        self.c_p = folder_name + '/license_character_classes.npy'
        self.model = None
        self.label = None
        self.plate_num = ""

    def load_model_arch(self):
        json_file = open(self.m_p, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(self.w_p)
        print("OCR model loaded successfully")
        self.label = LabelEncoder()
        self.label.classes_ = np.load(self.c_p)
        print("OCR Labels loaded successfully...")

    def get_countor(self, img_path):
        """
        Use this function when Path of Number plate is given
        Return : this returns the countours
        """
        img = cv2.imread(img_path)
        gray = cv2.imread(img_path, 0)  # 0: Gray scale loading
        binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # 180,255
        try:
            _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # gets all the countours
        except:
            contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # gets all the countours
        return contours, img, binary

    def get_countor_2(self, image_rgb):
        """
        Use this function when RGB image of Number plate is given
        Return : this returns the countours
        """
        img = image_rgb
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 0: Gray scale loading
        binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  # 180,255
        try:
            _, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # gets all the countours
        except:
            contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # gets all the countours

        return contours, img, binary

    def sequencing_countour(self, list_contour):
        dic = {}
        for i, c in enumerate(list_contour):
            x = []
            for e in c:
                x.append(e[0][0])
            dic.update({i: min(x)})
        dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1])}
        return list(dic.keys())

    def character_secmentation(self, single_countour, image_bin):
        (x, y, w, h) = cv2.boundingRect(single_countour)
        if w / h < 2 and h > 8:
            try:
                crop = image_bin[y - 1:y + h + 1, x - 1:x + w + 1]
                return crop
            except:
                pass
        else:
            return None

    def predict_from_model(self, image, model, labels):
        image = cv2.resize(image, (80, 80), interpolation=cv2.INTER_NEAREST)
        image = np.stack((image,) * 3, axis=-1)
        prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis, :]))])
        return prediction

    def plate_cleaning(self, string):

        char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                     'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                     'O', 'P', 'Q', 'R', 'S',
                     'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        num_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        state_list = ['AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ', 'HR', 'HP', 'JH',
                      'JK', 'KA', 'KL', 'LD', 'MH', 'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN',
                      'TR', 'TS', 'UK', 'UP', 'WB']

        temp = list(string)
        if len(temp) < 4:  # clearly no numberpalte can be less than 4
            return ""
        elif len(temp) > 9:  # when extra character comes
            if len(temp) == 10:  # when last detect is a Char not a Digit
                if temp[-1] in char_list:  # Removing last element if Char
                    temp = temp[:-1]
                    str = "".join(temp)
                    return str
                elif temp[0] in num_list:  # Removing First element if number
                    temp = temp[1:]
                    str = "".join(temp)
                    return str
                elif temp[0] in char_list and temp[1] in char_list and temp[2] in char_list:  # removig 1st state list
                    if "".join(temp[1:3]) in state_list and "".join(temp[0:2]) not in state_list:
                        temp = temp[1:]  # Removing First element
                        str = "".join(temp)
                        return str
                    elif temp[-1] in num_list and temp[-2] in num_list and temp[-3] in num_list and temp[
                        -4] in num_list and temp[-5] in num_list:
                        temp = temp[:-1]  # Removing First element
                        str = "".join(temp)
                        return str
                    else:
                        return string
                else:
                    return string
            elif temp[-1] in char_list:
                temp = temp[:-1]
                str = "".join(temp)
                return str
            else:
                return string
        elif temp[-1] in char_list:
            temp = temp[:-1]
            str = "".join(temp)
            return str
        else:
            return string

    def get_plate_number(self, img):
        plate = ""
        cnt, orig_img, binary_img = self.get_countor_2(img)
        seq_ind_cnt = self.sequencing_countour(cnt)
        for index in seq_ind_cnt:
            c = cnt[index]  # get countour of character
            if len(c) > 5:
                cropped_img = self.character_secmentation(single_countour=c, image_bin=binary_img)
                if cropped_img is not None:
                    try:
                        pred = self.predict_from_model(cropped_img, self.model, self.label)
                        plate += pred[0].strip("'[]")
                    except Exception as e:
                        pass
        string_corected = self.plate_cleaning(plate)
        self.plate_num = string_corected
        return self.plate_num


