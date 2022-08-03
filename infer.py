import cv2
import numpy as np
import queue
from keras.models import load_model


class ViolencePredictor(object):
    def __init__(self,
                 model_path: str = "model.h5",
                 imgsz: int = 128,
                 queue_num: int = 30,
                 conf_thres: float = 0.25,
                 num_thres: int = 20) -> None:
        self.model = load_model(model_path)
        self.results = queue.Queue()
        self.imgsz = imgsz
        self.queue_num = queue_num
        self.conf_thres = conf_thres
        self.num_thres = num_thres

    def __call__(self, img):
        # pre-process image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz)).astype("float32")
        img = img.reshape(self.imgsz, self.imgsz, 3) / 255
        # run infer
        preds = self.model.predict(np.expand_dims(img, axis=0))[0]
        # coount result
        if preds > self.conf_thres:
            self.results.put(True)
        else:
            self.results.put(False)
        if self.results.qsize() > self.queue_num:
            self.results.get()
        # judge
        if sum(self.results.queue) > self.num_thres:
            return True
        else:
            return False
