import cv2
import infer

violence_predictor = infer.ViolencePredictor()
i = 0
cap = cv2.VideoCapture("datasets/videos/Violence/V_129.mp4")
# cap = cv2.VideoCapture("datasets/videos/NonViolence/NV_129.mp4")
while i < 90:
    i += 1
    _, img = cap.read()
    violence = violence_predictor(img)
    print(violence)
