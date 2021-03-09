from pipeline.consumer import RecognitionModel
from pipeline.producer import StreamCamera
from models import Camera
from datetime import datetime, timedelta
import time
import json

import multiprocessing
import threading

class RecognitionManager:
    def __init__(self, cameras_dicts, detection_config, encoder_model_path, MAX_BUFFER=500):
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
            quit_event = multiprocessing.Event()
            self.producer_processes[c.get_id] = StreamCamera(c, self.buffer, quit_event, max_idle=60)
            self.producer_quit_events[c.get_id] = quit_event
            
        self.consumer_process = RecognitionModel(self.buffer, self.consumer_process_ready, self.producer_process_ready, self.update_encodings, encoder_model_path, quit_event=self.consumer_quit_event, **detection_config)
        self.consumer_process.start()
        while not self.consumer_process_ready.is_set():
            pass
        
    def set_update_encoding_event(self):
        while not self.kill_encoding_updater_thread.is_set():
            if datetime.now() == self.next_update_time:
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