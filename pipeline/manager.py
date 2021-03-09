import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pipeline.consumer import RecognitionModel, BatchGeneratorAndPiclker
from pipeline.producer import StreamCamera
from models import Camera
from datetime import datetime, timedelta
import time
import json

import multiprocessing
import threading

class RecognitionManager:
    def __init__(self, cameras_dicts, detection_config, encoder_model_path, MAX_BUFFER=14400):
        self.MAX_BUFFER = MAX_BUFFER
        self.buffer = multiprocessing.JoinableQueue(MAX_BUFFER)

        # creating events
        self.consumer_process_ready = multiprocessing.Event()
        self.producer_process_ready = multiprocessing.Event()
        self.consumer_quit_event = multiprocessing.Event()
        self.update_encodings = multiprocessing.Event()
        self.kill_encoding_updater_thread = threading.Event()
        current_datetime = datetime.now()
        self.next_update_time = current_datetime + timedelta(minutes=1)
        self.encoding_updater_thread = threading.Thread(target= self.set_update_encoding_event)

        # creating producers
        self.producer_processes = {}
        self.producer_quit_events = {}
        self.cameras = []
        for camera_d in cameras_dicts:
            c = Camera(**camera_d)
            self.cameras.append(c)
            quit_event = multiprocessing.Event() # process to be quit
            self.producer_processes[c.get_id] = StreamCamera(c, self.buffer, quit_event, max_idle=60)
            self.producer_quit_events[c.get_id] = quit_event
            
        self.consumer_process = RecognitionModel(self.buffer, self.consumer_process_ready, self.producer_process_ready, self.update_encodings, encoder_model_path, quit_event=self.consumer_quit_event, **detection_config)
        self.consumer_process.start()
        while not self.consumer_process_ready.is_set():
            pass
        
    def set_update_encoding_event(self):
        while not self.kill_encoding_updater_thread.is_set():
            if datetime.now() >= self.next_update_time:
                self.update_encodings.set()
                self.next_update_time = self.next_update_time + timedelta(minutes=10)
    
    def start(self):
        # starting all producer processes
        for _, process in self.producer_processes.items():
            process.start()
        self.producer_process_ready.set()
        
        self.encoding_updater_thread.start()
    
    def kill(self):
        self.buffer.close()
        pass
    
    def terminate(self):
        # terminatinf encoding updater thread
        self.kill_encoding_updater_thread.set()
        self.encoding_updater_thread.join()
        
        # terminating producers and consumer processs
        for _, event in self.producer_quit_events.items():
            event.set()
        self.consumer_quit_event.set()
        
        
class BatchRecognitionManager:
    def __init__(self, cameras_dicts, detection_config, encoder_model_path, MAX_BUFFER=14400, BATCH_SIZE=128):
        self.MAX_BUFFER = MAX_BUFFER
        self.BATCH_SIZE = BATCH_SIZE
        self.camera_batcher_buffer = multiprocessing.JoinableQueue(MAX_BUFFER)
        self.batcher_recognition_buffer = multiprocessing.JoinableQueue(MAX_BUFFER)

        # creating events
        self.recognition_consumer_process_ready = multiprocessing.Event()
        self.producers_process_ready = multiprocessing.Event()
        self.recognition_consumer_quit_event = multiprocessing.Event()
        self.batcher_consumer_quit_event = multiprocessing.Event()
        
        self.update_encodings = multiprocessing.Event()
        self.kill_encoding_updater_thread = threading.Event()
        current_datetime = datetime.now()
        self.next_update_time = current_datetime + timedelta(minutes=1)
        self.encoding_updater_thread = threading.Thread(target= self.set_update_encoding_event)

        # creating producers
        self.producer_processes = {}
        self.producer_quit_events = {}
        self.cameras = []
        for camera_d in cameras_dicts:
            c = Camera(**camera_d)
            self.cameras.append(c)
            quit_event = multiprocessing.Event()
            self.producer_processes[c.get_id] = StreamCamera(c, self.camera_batcher_buffer, quit_event, max_idle=60)
            self.producer_quit_events[c.get_id] = quit_event

        self.batcher_consumer_process = BatchGeneratorAndPiclker(self.camera_batcher_buffer,
                                                                 self.BATCH_SIZE,
                                                                 self.batcher_recognition_buffer,
                                                                 self.batcher_consumer_quit_event)
            
        self.recognition_consumer_process = RecognitionModel(self.batcher_recognition_buffer,
                                                 self.recognition_consumer_process_ready,
                                                 self.producers_process_ready,
                                                 self.update_encodings,
                                                 encoder_model_path,
                                                 self.BATCH_SIZE,
                                                 quit_event=self.recognition_consumer_quit_event,
                                                 **detection_config)
        
        self.recognition_consumer_process.start()
        while not self.recognition_consumer_process_ready.is_set():
            pass
        
    def set_update_encoding_event(self):
        while not self.kill_encoding_updater_thread.is_set():
            if datetime.now() >= self.next_update_time:
                self.update_encodings.set()
                self.next_update_time = self.next_update_time + timedelta(minutes=10)
    
    def start(self):
        # starting all producer processes
        for _, process in self.producer_processes.items():
            process.start()
        self.producers_process_ready.set()
        self.batcher_consumer_process.start()
        
        self.encoding_updater_thread.start()
    
    def kill(self):
        self.buffer.close()
    
    def terminate(self):
        # terminatinf encoding updater thread
        self.kill_encoding_updater_thread.set()
        self.encoding_updater_thread.join()
        
        # terminating producers and consumer processs
        for _, event in self.producer_quit_events.items():
            event.set()
        self.batcher_consumer_quit_event.set()
        self.recognition_consumer_quit_event.set()