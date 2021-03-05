from datetime import datetime
import logging
logging.basicConfig(filename='eduface.log', format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('EduFace started at {}'.format(datetime.now()))

import multiprocessing
# logger = multiprocessing.get_logger()
# # formatter = logging.Formatter('[%(levelname)s/%(processName)s] %(message)s')
# handler = logging.StreamHandler()
# # handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.setLevel(logging.INFO)

from pipeline import producer, consumer
from models import Camera
import time
import json

# 24fps * seconds = 14400
buffer = multiprocessing.JoinableQueue(14400)
# buffer = multiprocessing.Queue(14400)
consumer_process_ready = multiprocessing.Event()
producer_process_ready = multiprocessing.Event()
consumer_quit_event = multiprocessing.Event()

with open('detection_config.json','r') as file:
    detection_config = json.load(file)
consumer_process = consumer.RecognitionModel(buffer, consumer_process_ready, producer_process_ready, encoder_model_path='data/model/facenet_keras.h5', quit_event=consumer_quit_event, **detection_config)
consumer_process.start()

while not consumer_process_ready.is_set():
    pass

# reading cameras details
with open('cameras.json','r') as file:
    cameras_dicts = json.load(file)

# creating cameras objects and producer process
cameras = []
producer_processes = {}
producer_quit_events = {}
for camera_d in cameras_dicts:
    c = Camera(**camera_d)
    cameras.append(c)
    quit_event = multiprocessing.Event()
    producer_processes[c.get_id] = producer.StreamCamera(c, buffer, quit_event,max_idle=60, daemon=True)
    producer_quit_events[c.get_id] = quit_event

# consumer_process_ready = multiprocessing.Event()

# creating and starting consumer process 
# consumer_quit_event = multiprocessing.Event()
# consumer_process = consumer.SaveFrames(buffer, quit_event=consumer_quit_event, directory='frames/')
# consumer_process = consumer.ViewStream(buffer, consumer_quit_event)


# starting all producer processes
for _, process in producer_processes.items():
    process.start()
producer_process_ready.set()

time.sleep(5)

# terminating producers and consumer processs
for _, event in producer_quit_events.items():
    event.set()

# buffer.close()
buffer.join()
print('Buffer Cleared')
consumer_quit_event.set()
print('consumes_quit_event set')
buffer.close()


# for _, process in producer_processes.items():
#     process.join()
# consumer_process.join()

exit(1)
# logging.info('Produced Frames: {}'.format(buffer.produced))
# logging.info('Consumed Frames: {}'.format(buffer.consumed))
# logging.info('Current in buffer: {}'.format(buffer.qsize()))