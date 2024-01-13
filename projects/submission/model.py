import os
import cv2
import torch
from mmpretrain.apis import init_model, ImageClassificationInferencer


class model:
    def __init__(self):
        self.checkpoint = "epoch_36.pth"
        self.config_path = "my_swin_base_in1k_384.py"
        # The model is evaluated using CPU, please do not change to GPU to avoid error reporting.
        self.device = torch.device("cpu")

    def load(self, dir_path):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """
        # join paths
        checkpoint_path = os.path.join(dir_path, self.checkpoint)
        config_path = os.path.join(dir_path, self.config_path)
        self.model = init_model(config_path,checkpoint=checkpoint_path,device="cpu")
        self.model.eval()
        self.inference = ImageClassificationInferencer(self.model)

    def predict(self, input_image, patient_info_dict):
        """
        perform the prediction given an image and the metadata.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).
        note that the order of the three channels of the input_image read by cv2 is BGR.
        :param input_image: the input image to the model.
        :param patient_info_dict: a dictionary with the metadata for the given image,
        such as {'age': 52.0, 'sex': 'male', 'height': nan, 'weight': 71.3},
        where age, height and weight are of type float, while sex is of type str.
        :return: an int value indicating the class for the input image.
        """

        with torch.no_grad():
            predict = self.inference([input_image], batch_size=1)
        pred_class = predict[0]["pred_label"]

        return int(pred_class)
