import cv2
import numpy as np
import queue
from keras.models import load_model

MODEL_SETTING_DICT = {'violenceModel_baseOn_inceptionV3.h5':{'imgsz':224,'normalize':False,'mean_subtraction':True},
                      'violenceModel_baseOn_MobileNetV2.h5':{'imgsz':128,'normalize':True,'mean_subtraction':True}}
MEAN = np.array([123.68, 116.779, 103.939][::1], dtype="float32")


class ViolencePredictor(object):
    '''
    model_name: ['violenceModel_baseOn_inceptionV3.h5','violenceModel_baseOn_MobileNetV2.h5']
    '''
    def __init__(self,
                 model_name: str = "violenceModel_baseOn_inceptionV3.h5",
                 #imgsz: int = 128,
                 queue_num: int = 30,
                 conf_thres: float = 0.25,
                 num_thres: int = 20,
                 video_output_path: str = './test.avi') -> None:
        self.model_name = model_name
        self.model = load_model(f"models/{model_name}")
        self.results = queue.Queue()
        self.imgsz = MODEL_SETTING_DICT[model_name]['imgsz']
        self.queue_num = queue_num
        self.conf_thres = conf_thres
        self.num_thres = num_thres
        self.writer = None
        (self.W, self.H) = (None, None)
        self.video_output_path = video_output_path

    def process(self, img, is_export=False):
        original_img = img.copy()
        if self.W is None or self.H is None:
            (self.H, self.W) = img.shape[:2]
        # pre-process image
        img = self._preprocess(img)
        # run infer
        preds = self.model.predict(np.expand_dims(img, axis=0))[0]
        if self.model_name == 'violenceModel_baseOn_inceptionV3.h5':
            preds = preds[1]
        # count result
        if preds > self.conf_thres:
            self.results.put(True)
        else:
            self.results.put(False)
        if self.results.qsize() > self.queue_num:
            self.results.get()
        # judge
        judge_result = sum(self.results.queue) > self.num_thres

        if is_export:
            self._save_infer_image(original_img,judge_result)
        if judge_result:
            return judge_result
        else:
            return judge_result
        
    def _preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz)).astype("float32")
        img = img.reshape(self.imgsz, self.imgsz, 3)
        img = img / 255 if MODEL_SETTING_DICT[self.model_name]['normalize'] else img
        img = img - MEAN if MODEL_SETTING_DICT[self.model_name]['mean_subtraction'] else img
        return img
    def _save_infer_image(self,img,judge_result):
        if judge_result:
            label = 'violence'
            text_color = (0, 0, 255)
        else:
            label = 'nonviolence'
            text_color = (0, 255, 0)
        
        text = "State : {:8}".format(label)
        FONT = cv2.FONT_HERSHEY_SIMPLEX 

        cv2.putText(img, text, (35, 50), FONT,1.25, text_color, 3)
        cv2.putText(img, text, (35, 50), FONT,1.25, text_color, 3)
        output = cv2.rectangle(img, (35, 80), (35,100), text_color,-1)
        if self.writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.writer = cv2.VideoWriter(self.video_output_path, fourcc, 30,(self.W, self.H), True)
        self.writer.write(output)
