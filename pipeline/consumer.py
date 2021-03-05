import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing
import queue
from datetime import datetime
import numpy as np
import logging
import json
from models import Frame
# from pipeline.buffer import Buffer

# for SaveFrames
from PIL import Image
# for ViewStreams
import cv2

# for RecognitionModel
from scipy.spatial.distance import cosine
from sklearn.preprocessing import Normalizer
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model, Model
# import mysql.connector

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", message = "Passing", category = FutureWarning)
import pickle

from tensorflow.compat import v1 as tfv1
from tensorflow.compat.v1.keras import backend as K

class SaveFrames(multiprocessing.Process):
    def __init__(self, buffer: multiprocessing.JoinableQueue, quit_event: multiprocessing.Event, directory='frames/'):
        """Iniitalize Consumer object

        Args:
            buffer (multiprocessing.JoinableQueue): Used to insert captured frame to get processed
            quit_event (multiprocessing.Event): used to trigger process to stop
        """
        super(SaveFrames,self).__init__()
        logging.info('Creating SaveFrames process')
        self.buffer = buffer
        self.directory = directory
        self.quit_event = quit_event
        

    def run(self):
        logging.info('Running SaveFrames process')
        while not self.quit_event.is_set():
            frame = self.buffer.get(60)
            print('Consuming {} in SaveFrames'.format(frame))
            # data = np.zeros(160,240,3)

            # img = Image.fromarray(frame.numpy().reshape(frame.get_shape()))
            img = Image.fromarray(frame.numpy())
            img.save(self.directory+'{}.jpg'.format(frame.get_timestamp_fmt()))
            # img.show()
            self.buffer.task_done()
        logging.info('Terminated SaveFrames process')


class ViewStream(multiprocessing.Process):
    def __init__(self, buffer: multiprocessing.JoinableQueue, quit_event: multiprocessing.Event, directory='frames/'):
        """Iniitalize Consumer object

        Args:
            buffer (multiprocessing.JoinableQueue): Used to insert captured frame to get processed
            quit_event (multiprocessing.Event): used to trigger process to stop
        """
        super(ViewStream,self).__init__()
        logging.info('Creating ViewStream proces')
        self.buffer = buffer
        self.quit_event = quit_event

    def run(self):
        logging.info('Running ViewStream proces')
        while not self.quit_event.is_set():
            frame = self.buffer.get(60)
            # img = Image.fromarray(frame.numpy().reshape(frame.get_shape()))
            img = Image.fromarray(frame.numpy())
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow('{}'.format(frame.get_camera()),img)
            self.buffer.task_done()
            k = cv2.waitKey(1)
            if k==27:
                self._running = False
                self.buffer.clear()
                break
        cv2.destroyAllWindows()
        logging.info('Terminated ViewStream proces')            


class RecognitionModel(multiprocessing.Process):
    
    def __init__(self, buffer: multiprocessing.JoinableQueue, consumer_process_ready: multiprocessing.Event, producer_process_ready: multiprocessing.Event, encoder_model_path: str, quit_event: multiprocessing.Event, **detection_config):
        super(RecognitionModel, self).__init__()
        logging.info('Creating RecognitionModel process')
        self.buffer = buffer
        
        self.encoder_model_path = encoder_model_path
        self.detection_config = detection_config
        
        with open('file_encoding.pickle', 'rb') as handle:
            self.encoding_dict = pickle.load(handle)
        
        self.detection_config = detection_config
        self.consumer_process_ready = consumer_process_ready
        self.producer_process_ready = producer_process_ready

        self.quit_event = quit_event
        logging.info('Created RecognitionModel process')

    def _recognize(self, detector_model, encoder_model, img: np.ndarray):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with self.sess.graph.as_default():
            results = detector_model.detect_faces(img_rgb)
        for res in results:
            if res['confidence'] < self.detection_config['confidence_threshold']:
                continue
            face, pt_1, pt_2 = self._get_face(img_rgb, res['box'])
            encoding = self._get_encoding(encoder_model, face)
            encoding = Normalizer('l2').transform(encoding.reshape(1, -1))[0]
            name = 'Unknown'
            distance = float("inf")
            for db_name, db_encoding in self.encoding_dict.items():
                db_encoding = json.loads(db_encoding)
                dist = cosine(db_encoding, encoding)
                if dist < self.detection_config['recognition_threshold'] and dist < distance:
                    name = db_name
                    distance = dist
            cv2.rectangle(img_rgb, pt_1, pt_2, (255, 0, 0), 2)
            cv2.putText(img_rgb, name, (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
        return img_rgb, len(results)
    
    def _get_face(self, frame: Frame, box: tuple):
        x1, y1, width, height = box
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = frame[y1:y2, x1:x2]
        return face, (x1, y1), (x2, y2)
    
    def _get_encoding(self, encoder_model, face):
        face = self._normalize(face)
        face = cv2.resize(face, tuple(self.detection_config['recognized_face_shape']))
        with self.sess.graph.as_default():
            encoding = encoder_model.predict(np.expand_dims(face, axis=0))[0]
        return encoding

    def _normalize(self, img):
        mean, std = img.mean(), img.std()
        return (img - mean) / std

    def run(self):
        logging.info('Loading models on GPU')
        config = tfv1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tfv1.Session(config=config)
        
        with self.sess.graph.as_default():
            K.set_session(tfv1.Session(config = config))
            detector_model = MTCNN()
            encoder_model = load_model(self.encoder_model_path, compile=False)
        logging.info('Running RecognitionModel thread')
        self.consumer_process_ready.set()
        
        while not self.producer_process_ready.is_set():
            pass
        
        while not self.quit_event.is_set():
            frame = self.buffer.get(60)
            # img = Image.fromarray(frame.numpy().reshape(frame.get_shape()))
            img = Image.fromarray(frame.numpy())
            img = np.array(img)
            img, results = self._recognize(detector_model, encoder_model, img)
            if not results:
                cv2.imwrite('frames/{}.jpg'.format(frame.get_timestamp_fmt()), img)
            else:
                pass
            self.buffer.task_done()
            # cv2.imshow('{}'.format(frame.get_camera()),img)
            # k = cv2.waitKey(1)
            # if k==27:
            #     self._running = False
            #     self.buffer.clear()
            #     break
        # cv2.destroyAllWindows()
        logging.info('Terminated ViewStream process')
