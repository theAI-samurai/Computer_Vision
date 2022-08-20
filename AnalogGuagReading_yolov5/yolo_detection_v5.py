import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from torchvision.utils import save_image


class YoloDetection:
    def __init__(self, weight_path, yaml_path, cuda_device=None, dnn=False, fp16=False, fuse=True):
        self.weights = weight_path
        self.data_yaml = yaml_path
        self.dnn = dnn
        self.fp16 = fp16
        self.fuse = fuse
        if cuda_device is None:
            self.device = self.get_device()
        else:
            self.device = cuda_device
        self.model = self.get_model()
        self.stride = self.model.stride
        self.names = self.model.names
        self.num_class = len(self.names)
        self.torch_image_tensor = None

    def get_model(self):
        model = DetectMultiBackend(weights=self.weights, device=self.device,
                                   dnn=self.dnn, data=self.data_yaml,
                                   fuse=self.fuse, fp16=self.fp16)
        return model

    def get_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def predict(self, image, num_det, iou_thresh=0.45, conf_thresh=0.25):
        im = torch.from_numpy(image).to(self.device)
        im = im.float()
        im /= 255

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.torch_image_tensor = im.shape
        prediction = self.model(im)
        prediction = non_max_suppression(prediction, conf_thresh, iou_thresh, max_det=num_det)
        return prediction

    def scale_prediction(self, prediction, input_shape_tuple, output_shape_tuple):
        prediction[0][:, :4] = scale_coords(input_shape_tuple, prediction[0][:, :4], output_shape_tuple).round()
        prediction = prediction[0].tolist()
        return prediction

