import cv2
from detector import BirdDetector

cap = cv2.VideoCapture("sample_videos/poultry_cctv.mp4")
det = BirdDetector()

while True:
    ret,frame = cap.read()
    if not ret:
        break

    dets = det.detect(frame)
    for x1,y1,x2,y2,_ in dets:
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)

    cv2.imshow("DETECTION DEBUG", frame)
    if cv2.waitKey(1)==27:
        break
