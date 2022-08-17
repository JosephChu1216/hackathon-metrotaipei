import cv2
import infer

i = 0
cap = cv2.VideoCapture("dataset/Violence/V_129.mp4")
# cap = cv2.VideoCapture("datasets/videos/NonViolence/NV_129.mp4")

model_name_list = ['violenceModel_baseOn_inceptionV3.h5','violenceModel_baseOn_MobileNetV2.h5']
vp_baseOn_inceptionV3 = ViolencePredictor(model_name = model_name_list[0], video_output_path='./infer_V_129_baseOn_inceptionV3.avi')
vp_baseOn_MobileNetV2 = ViolencePredictor(model_name = model_name_list[1], video_output_path='./infer_V_129_baseOn_MobileNetV2.avi')

while i < 180:
    i += 1
    grabbed, img = cap.read()
    if not grabbed:
	    break
    violence_1 = vp_baseOn_inceptionV3.process(img)
    violence_2 = vp_baseOn_MobileNetV2.process(img)
    print(violence_1,violence_2)
