from tflite_runtime.interpreter import Interpreter
from PIL import Image

import numpy as np

import re

class ObjectDetector():
    def __init__(self, model_path, label_path):
        self.__labels = self.load_labels(label_path)
        self.__interpreter = Interpreter(model_path)
        self.__interpreter.allocate_tensors()

    def load_labels(self, path):
        """
        ラベルをリストに読み込む
        """
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            labels = {}
            for row_number, content in enumerate(lines):
                pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
                if len(pair) == 2 and pair[0].strip().isdigit():
                    labels[int(pair[0])] = pair[1].strip()
                else:
                    labels[row_number] = pair[0].strip()
        
        return labels


    def set_input_tensor(self, interpreter, image):
        tensor_index = interpreter.get_input_details()[0]['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image


    def get_output_tensor(self, interpreter, index):
        output_details = interpreter.get_output_details()[index]
        tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
        return tensor


    def detect_object(self, image, threshold):
        """
        物体検出を行う
        """

        resized = self.resize(self.__interpreter, image)
        self.set_input_tensor(self.__interpreter, resized)
        self.__interpreter.invoke()

        boxes = self.get_output_tensor(self.__interpreter, 0)
        classes = self.get_output_tensor(self.__interpreter, 1)
        scores = self.get_output_tensor(self.__interpreter, 2)
        count = int(self.get_output_tensor(self.__interpreter, 3))

        results = []
        for i in range(count):
            if scores[i] >= threshold:
                result = {
                    'bounding_box': boxes[i],
                    'class_id': classes[i],
                    'score': scores[i]
                }
                results.append(result)
        return results

    def resize(self, interpreter, image_array):
        _, input_width, input_height, _ = interpreter.get_input_details()[0]['shape']
        with Image.fromarray(image_array).convert('RGB') as image:
            resized_image = image.resize((input_width, input_height), Image.ANTIALIAS)

            return resized_image

    def get_labels(self):
        return self.__labels