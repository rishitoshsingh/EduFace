import cv2
import numpy as np
import json
from models import Camera
from datetime import datetime


with open('cameras.json','r') as file:
    cameras_dicts = json.load(file)

with open('motion_configs.json','r') as file:
    motion_configs = json.load(file)

RESOLUTION = (1280, 720)

cameras = []
for camera_d in cameras_dicts:
    c = Camera(**camera_d)
    cameras.append(c)

CAMERA_N = 1
camera = cameras[CAMERA_N]
motion_config = motion_configs[str(camera.get_id())]
cap = cv2.VideoCapture(camera.get_stream())

width = abs(motion_config['corner_1'][0] - motion_config['corner_2'][0]) 
height = abs(motion_config['corner_1'][1] - motion_config['corner_2'][1]) 
motion_area = width*height
print('Motion area ',motion_area)
detection_area_threshold = abs(0.5 * motion_area)
print('Trigger area ',detection_area_threshold)
_, frame1 = cap.read()
frame1 = cv2.resize(frame1, RESOLUTION)
_, frame2 = cap.read()
frame2 = cv2.resize(frame1, RESOLUTION)

def detect_motion(frame1, frame2):
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        area = w*h
        if area < detection_area_threshold:
            continue
        frame_cp = frame1.copy()
        cv2.rectangle(frame_cp, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imwrite('motion_detected/{}.jpg'.format(datetime.now()), frame_cp)
        return True
    

while True:
    x1, y1, x2, y2 = motion_config['corner_1'][0], motion_config['corner_1'][1], motion_config['corner_2'][0], motion_config['corner_2'][1]
    frame1_detect, frame2_detect = frame1[y1:y2, x1:x2 ], frame2[y1:y2, x1:x2 ]
    detect_motion(frame1_detect, frame2_detect)
    
    # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    cv2.rectangle(frame1, tuple(motion_config['corner_1']), tuple(motion_config['corner_2']), (255, 0, 0), 2)
    cv2.imshow(str(camera.get_id()), frame1)
    frame1 = frame2
    _, frame2 = cap.read()
    frame2 = cv2.resize(frame2, RESOLUTION)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()