import logging
logging.basicConfig(filename='eduface.log', format='%(asctime)s - %(message)s', level=logging.INFO)

from pipeline import producer, consumer
from models import Camera
import time
import json
import multiprocessing
# from multiprocessing import Process, Queue, Event

# from mtcnn.mtcnn import MTCNN
# from tensorflow.keras.models import load_model, Model
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import warnings
# warnings.filterwarnings("ignore", message = "Passing", category = FutureWarning)

# # loading models

# detector_model = MTCNN()
# encoder_model = load_model(encoder_model)

# buffer to store frames from all cameras
# buffer = buffer.Buffer()
buffer = multiprocessing.JoinableQueue(14400)

# reading cameras details
# with open('cameras.json','r') as file:
with open('yt_videos.json','r') as file:
    cameras_dicts = json.load(file)

# creating cameras objects and producer process
cameras = []
producer_processes = {}
producer_quit_events = {}
for camera_d in cameras_dicts:
    c = Camera(**camera_d)
    cameras.append(c)
    quit_event = multiprocessing.Event()
    producer_processes[c.get_id] = producer.StreamCamera(c, buffer, quit_event)
    producer_quit_events[c.get_id] = quit_event

# consumer_process_ready = multiprocessing.Event()

# creating and starting consumer process 
consumer_quit_event = multiprocessing.Event()
consumer_process = consumer.SaveFrames(buffer, quit_event=consumer_quit_event, directory='frames/')
# consumer_process = consumer.ViewStream(buffer, consumer_quit_event)
# with open('detection_config.json','r') as file:
#     detection_config = json.load(file)
# consumer_process = consumer.RecognitionModel(buffer, consumer_process_ready, encoder_model_path='data/model/facenet_keras.h5', consumer_quit_event, **detection_config)

# while not consumer_process_ready.is_set():
    # pass
# starting all producer processes
for _, process in producer_processes.items():
    process.start()
time.sleep(5)

consumer_process.start()

time.sleep(60)
# terminating producers and consumer processs
for _, event in producer_quit_events.items():
    event.set()
consumer_quit_event.set()

buffer.join()
# for _, process in producer_processes.items():
#     process.join()
# consumer_process.join()

# logging.info('Produced Frames: {}'.format(buffer.produced))
# logging.info('Consumed Frames: {}'.format(buffer.consumed))
# logging.info('Current in buffer: {}'.format(buffer.qsize()))