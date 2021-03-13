import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pipeline.consumer import RecognitionModel, BatchGeneratorAndPiclker
from pipeline.producer import StreamCamera
from models import Camera
from datetime import datetime, timedelta
import time
import signal
import json

import multiprocessing
import threading

class RecognitionManager:
    """Manager to create, manage and terminate producer, buffer and consumer. (currently not used in EduFace pipeline)
    """
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
        """function to set update encoding event which is used in recognition consumer
        """
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
    """This manager will create a pipeline consisting of StreamCamera producers, BatchGeneratorAndPiclker consumer and RecognitionModel consumer
    """
    def __init__(self,
                 cameras_dicts: dict,
                 motion_configs: dict,
                 detection_config: dict,
                 encoder_model_path: str,
                 MAX_BUFFER=14400,
                 BATCH_SIZE=128,
                 BATCH_TIMEOUT_MINUTES=1):
        """Create manager object

        Args:
            cameras_dicts (dict): camera details (read from cameras.json)
            motion_configs (dict): motion detection configs (read from motion_configs.json)
            detection_config (dict): detection config (read from detection_config.json)
            encoder_model_path (str): trained encoder model path
            MAX_BUFFER (int, optional): buffer size. Defaults to 14400.
            BATCH_SIZE (int, optional): batch size. Defaults to 128.
            BATCH_TIMEOUT_MINUTES (int, optional): batch timeout in minutes. Defaults to 1.
        """
        self.MAX_BUFFER = MAX_BUFFER
        self.BATCH_SIZE = BATCH_SIZE
        self.BATCH_TIMEOUT_MINUTES = BATCH_TIMEOUT_MINUTES
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
            motion_config = motion_configs[str(c.get_id())]
            self.cameras.append(c)
            quit_event = multiprocessing.Event()
            self.producer_processes[c.get_id] = StreamCamera(c, motion_config, self.camera_batcher_buffer, quit_event, max_idle=60)
            self.producer_quit_events[c.get_id] = quit_event

        self.batcher_consumer_process = BatchGeneratorAndPiclker(self.camera_batcher_buffer,
                                                                 self.BATCH_SIZE,
                                                                 self.BATCH_TIMEOUT_MINUTES,
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
        self.camera_batcher_buffer.close()
        self.batcher_recognition_buffer.close()
    
    def terminate(self):
        # terminatinf encoding updater thread
        self.kill_encoding_updater_thread.set()
        self.encoding_updater_thread.join()
        
        # terminating producers and consumer processs
        for _, event in self.producer_quit_events.items():
            event.set()
        # wait till all frames are saved in batches
        self.camera_batcher_buffer.join()
        # kill when frames are saved
        self.batcher_consumer_quit_event.set()
        # wait till all recognizing batches is completed
        self.batcher_recognition_buffer.join()
        # kill when all batcher are analyzed
        self.recognition_consumer_quit_event.set()
        
class BatchPicklingManager:
    """Manager which will create a pipeline consisting of StreamCamera producers and BatchGeneratorAndPiclker consumer
    """
    def __init__(self,
                 cameras_dicts: dict,
                 motion_configs: dict,
                 MAX_BUFFER=14400,
                 BATCH_SIZE=128,
                 BATCH_TIMEOUT_MINUTES=1):
        """Create manager object

        Args:
            cameras_dicts (dict): camera details (read from cameras.json)
            motion_configs (dict): motion detection configs (read from motion_configs.json)
            MAX_BUFFER (int, optional): buffer size. Defaults to 14400. Defaults to 14400.
            BATCH_SIZE (int, optional): batch size. Defaults to 128. Defaults to 128.
            BATCH_TIMEOUT_MINUTES (int, optional): batch timeout in minutes. Defaults to 1.
        """
        self.MAX_BUFFER = MAX_BUFFER
        self.BATCH_SIZE = BATCH_SIZE
        self.BATCH_TIMEOUT_MINUTES = BATCH_TIMEOUT_MINUTES
        self.camera_batcher_buffer = multiprocessing.JoinableQueue(MAX_BUFFER)

        # creating events
        self.recognition_consumer_process_ready = multiprocessing.Event()
        self.producers_process_ready = multiprocessing.Event()
        self.batcher_consumer_quit_event = multiprocessing.Event()
        
        # creating producers
        self.producer_processes = {}
        self.producer_quit_events = {}
        self.cameras = []
        for camera_d in cameras_dicts:
            c = Camera(**camera_d)
            motion_config = motion_configs[str(c.get_id())]
            self.cameras.append(c)
            quit_event = multiprocessing.Event()
            self.producer_processes[c.get_id] = StreamCamera(c, motion_config, self.camera_batcher_buffer, quit_event, max_idle=60)
            self.producer_quit_events[c.get_id] = quit_event

        self.batcher_consumer_process = BatchGeneratorAndPiclker(self.camera_batcher_buffer,
                                                                 self.BATCH_SIZE,
                                                                 self.BATCH_TIMEOUT_MINUTES,
                                                                 None,
                                                                 self.batcher_consumer_quit_event)
            
    def start(self):
        # starting all producer processes
        for _, process in self.producer_processes.items():
            process.start()
        self.producers_process_ready.set()
        self.batcher_consumer_process.start()
    
    def kill(self):
        self.camera_batcher_buffer.close()
    
    def terminate(self):
        # terminatinf encoding updater thread
        self.kill_encoding_updater_thread.set()
        self.encoding_updater_thread.join()
        
        # terminating producers and consumer processs
        for _, event in self.producer_quit_events.items():
            event.set()
        # wait till all frames are saved in batches
        self.camera_batcher_buffer.join()
        # kill when frames are saved            
        self.batcher_consumer_quit_event.set()