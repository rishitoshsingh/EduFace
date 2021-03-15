import cv2
from imutils.video import VideoStream
import numpy as np
import json
from models import Camera


with open('cameras.json','r') as file:
    cameras_dicts = json.load(file)

UPDATE_NUM = 1
RESOLUTION = (1280, 720)

cameras = []
for camera_d in cameras_dicts:
    c = Camera(**camera_d)
    cameras.append(c)

motion_boxes = {}
def get_new_motion_box(corner_1, corner_2):
    motion_box = {
        'corner_1': corner_1,
        'corner_2': corner_2
    }
    return motion_box
motion_box = get_new_motion_box(corner_1=[0,0], corner_2=list(RESOLUTION))
        
for camera in cameras:
    cap = VideoStream(camera.get_stream()).start()
    try:
        frame = cap.read()
        frame = cv2.resize(frame, RESOLUTION)
        (height, width) = frame.shape[:2]
        # motion_box['corner_2'] = [width, height]
        while True:
            frame = cap.read()
            frame = cv2.resize(frame, RESOLUTION)
            
            frame = cv2.putText(frame, str(camera), (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (225,0,0),2)
            frame = cv2.putText(frame, str(motion_box['corner_1']), (0, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0,225,0),2)
            frame = cv2.putText(frame, str(motion_box['corner_2']), (0, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0,225,0),2)
            
            frame = cv2.rectangle(frame, tuple(motion_box['corner_1']), tuple(motion_box['corner_2']), (225,0,0),2)
            cv2.imshow(str(camera), frame)
            key = cv2.waitKey(1)
            if key == ord('w'):
                motion_box['corner_1'][1] = (motion_box['corner_1'][1] - UPDATE_NUM) % height
            elif key == ord('a'):
                motion_box['corner_1'][0] = (motion_box['corner_1'][0] - UPDATE_NUM) % width
            elif key == ord('s'):
                motion_box['corner_1'][1] = (motion_box['corner_1'][1] + UPDATE_NUM) % height
            elif key == ord('d'):
                motion_box['corner_1'][0] = (motion_box['corner_1'][0] + UPDATE_NUM) % width
            
            elif key == ord('i'):
                motion_box['corner_2'][1] = (motion_box['corner_2'][1] - UPDATE_NUM) % height
            elif key == ord('j'):
                motion_box['corner_2'][0] = (motion_box['corner_2'][0] - UPDATE_NUM) % width
            elif key == ord('k'):
                motion_box['corner_2'][1] = (motion_box['corner_2'][1] + UPDATE_NUM) % height
            elif key == ord('l'):
                motion_box['corner_2'][0] = (motion_box['corner_2'][0] + UPDATE_NUM) % width
            elif key == 27:
                motion_boxes.update({camera.get_id(): motion_box})
                motion_box = get_new_motion_box(motion_box['corner_1'].copy(), motion_box['corner_2'].copy())
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
        cv2.destroyAllWindows()

with open('motion_configs_temp.json','w') as file:
    json.dump(motion_boxes, file, indent=4)