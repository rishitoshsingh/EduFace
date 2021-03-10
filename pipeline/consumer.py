import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat import v1 as tfv1
import pickle
import warnings
import mysql.connector
from tensorflow.keras.models import load_model, Model
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine
import cv2
from PIL import Image
import pipeline.recognize_utils as utils
from models import Frame, Camera
import json
import logging
import numpy as np
from datetime import datetime, timedelta
import queue
import multiprocessing
import pickle
warnings.filterwarnings("ignore", message = "Passing", category = FutureWarning)


class SaveFrames(multiprocessing.Process):
    def __init__(self, buffer: multiprocessing.JoinableQueue, quit_event: multiprocessing.Event, directory='frames/'):
        """Iniitalize Consumer object

        Args:
            buffer (multiprocessing.JoinableQueue): Used to insert captured frame to get processed
            quit_event (multiprocessing.Event): used to trigger process to stop
        """
        super(SaveFrames, self).__init__()
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

class BatchGeneratorAndPiclker(multiprocessing.Process):
    """This consumer read all frames from inBuffer, generate batches and pickel them into disks. These pickeled batch file names are inserted in outBuffer.
    """
    def __init__(self, inBuffer: multiprocessing.JoinableQueue, batch_size: int, outBuffer: multiprocessing.JoinableQueue, quit_event: multiprocessing.Event, directory='batches/'):
        """Iniitalize Consumer object

        Args:
            inBuffer (multiprocessing.JoinableQueue): Used to insert captured frame to get processed
            batch_size (int): batch size 
            outBuffer (multiprocessing.JoinableQueue): Used to insert file names of batches saved in disk
            quit_event (multiprocessing.Event): used to trigger process to stop
            diectory (str): directory to save pickled objects 
        """
        super(BatchGeneratorAndPiclker, self).__init__()
        logging.info('Creating BatchGeneratorAndPiclker process')
        self.inBuffer = inBuffer
        self.BATCH_SIZE = batch_size
        self.quit_event = quit_event
        self.outBuffer = outBuffer
        self.directory = directory
        
    def _generate_batch(self,batch_id:int):
        batch = {}
        batch['id'] = batch_id
        batch['timestamp'] = datetime.now()
        batch['batch_size'] = self.BATCH_SIZE
        batch['num_frames'] = 0
        batch['frames'] = []
        return batch

    def run(self):
        logging.info('Running BatchGeneratorAndPiclker process')
        batch_count = 1
        batch = self._generate_batch(batch_count)
        
        FPS_COUNTER_INTERVAL = 1 # minutes
        current_fps_counter_time = datetime.now()
        next_fps_counter_time = current_fps_counter_time + timedelta(minutes=FPS_COUNTER_INTERVAL)
        consumed_frames = 0
        
        while not self.quit_event.is_set():
            frame = self.inBuffer.get(60)
            batch['frames'].append(frame)
            batch['num_frames'] += 1
            self.inBuffer.task_done()
            
            consumed_frames += 1
            if datetime.now() >= next_fps_counter_time:
                delta_seconds = (datetime.now() - current_fps_counter_time).seconds 
                print('BatchGeneratorAndPiclker consuming {:4.1f} fps'.format(consumed_frames / delta_seconds))
                current_fps_counter_time = datetime.now()
                next_fps_counter_time = current_fps_counter_time + timedelta(minutes=FPS_COUNTER_INTERVAL)
                consumed_frames = 0
            
            if batch['num_frames'] == self.BATCH_SIZE:
                file_name = self.directory + 'batch-{}'.format(batch['id']) + ' ({})'.format(batch['timestamp'].strftime('%-d %b %-y, %-I:%-M %p (%f)')) + '.pickle'
                with open(file_name, 'wb') as file:
                    pickle.dump(batch, file)
                if self.outBuffer is not None:
                    self.outBuffer.put(file_name)
                batch_count += 1
                batch = self._generate_batch(batch_count)
                
        logging.info('Terminated BatchGeneratorAndPiclker process')


class ViewStream(multiprocessing.Process):
    def __init__(self, buffer: multiprocessing.JoinableQueue, quit_event: multiprocessing.Event, directory='frames/'):
        """Iniitalize Consumer object

        Args:
            buffer (multiprocessing.JoinableQueue): Used to insert captured frame to get processed
            quit_event (multiprocessing.Event): used to trigger process to stop
        """
        super(ViewStream, self).__init__()
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
            cv2.imshow('{}'.format(frame.get_camera()), img)
            self.buffer.task_done()
            k = cv2.waitKey(1)
            if k == 27:
                self._running = False
                self.buffer.clear()
                break
        cv2.destroyAllWindows()
        logging.info('Terminated ViewStream proces')


class RecognitionModel(multiprocessing.Process):

    def __init__(self,
                 buffer: multiprocessing.JoinableQueue,
                 consumer_process_ready: multiprocessing.Event,
                 producer_process_ready: multiprocessing.Event, 
                 update_encodings: multiprocessing.Event, 
                 encoder_model_path: str,
                 batch_siz: int, 
                 quit_event: multiprocessing.Event, 
                 **detection_config):
        
        super(RecognitionModel, self).__init__()
        logging.info('Creating RecognitionModel process')
        self.buffer = buffer
        self.BATCH_SIZE = batch_siz

        self.encoder_model_path = encoder_model_path
        self.detection_config = detection_config

        self.detection_config = detection_config
        
        self.consumer_process_ready = consumer_process_ready
        self.producer_process_ready = producer_process_ready
        self.update_encodings = update_encodings 
        self.quit_event = quit_event

        self.mysql_host = 'localhost'
        self.mysql_user = 'root'
        self.mysql_password = 'edugrad@1234'
        self.mysql_name = 'asbl'

        # self.known_attendance_flag = {}
        self.known_encoding = {}
        self.known_attendance_data = {}
        self.known_attendance_time = {}
        self.known_start_time = ''
        self.unknown_counter = 0
        self.unknown_encoding = {}
        self.unkown_attendance_data = {}
        self.unknown_attendance_time = {}
        self.unknown_start_time = ''

        logging.info('Created RecognitionModel process')

    def _get_mysql_connection(self):
        return mysql.connector.connect(host=self.mysql_host, user=self.mysql_user, password=self.mysql_password, database=self.mysql_name)

    def _read_known_encodings(self):
        known_encoding_dict = {}
        mydb = self._get_mysql_connection()
        my_cursor = mydb.cursor()
        sql = """select id, name from labours where deleted_at is NULL"""
        my_cursor.execute(sql)
        labour_id = my_cursor.fetchall()
        for i in labour_id:
            my_cursor = mydb.cursor()
            sql = """select embeding from embedings where labour_id = %s"""
            my_cursor.execute(sql, (int(i[0]), ))
            emb = my_cursor.fetchall()
            known_encoding_dict[i[1]] = json.loads(emb[0][0])
        mydb.commit()
        self.known_encoding = known_encoding_dict
        logging.info('Known encodings updated from database')

    def _read_unknown_encodings(self):
        unknown_encoding_dict = {}
        mydb = self._get_mysql_connection()
        mycursor = mydb.cursor()
        sql = """SELECT name, un_embedings from unkown_att"""
        mycursor.execute(sql)
        result = mycursor.fetchall()
        for a1, a2 in result:
            unknown_encoding_dict[a1] = json.loads(a2)
        mydb.commit()
        self.unknown_encoding = unknown_encoding_dict
        logging.info('Unknown encodings updated from database')

    def _update_unknown_attendance(self, unknown_name, encoding):
        mydb = self._get_mysql_connection()
        mycursor = mydb.cursor()
        sql = """INSERT INTO unkown_att (name , un_embedings) VALUES (%s, %s)"""
        val = (unknown_name, json.dumps(encoding, cls=utils.NumpyEncoder))
        mycursor.execute(sql, val)
        mydb.commit()

    def _update_known_attendance(self, name: str, camera: Camera, file_name: str):
        mydb = self._get_mysql_connection()
        mycursor = mydb.cursor()
        sql = "SELECT id FROM labours id where name = %s"
        mycursor.execute(sql, (name, ))
        labour_id = int(mycursor.fetchall()[0][0])
        mydb.commit()

        sql = """INSERT INTO attendances (labour_id, camera, location, timestamp, photo, created_at) VALUES (%s, %s, %s, %s, %s, %s)"""
        val = (labour_id, camera.get_id(), camera.get_location(),
               datetime.now(), file_name, datetime.now())
        mycursor.execute(sql, val)
        mydb.commit()

        mycursor.execute(sql, val)
        mydb.commit()

    def _recognize(self, detector_model, encoder_model, img: np.ndarray, camera: Camera):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with self.sess.graph.as_default():
            results = detector_model.detect_faces(img_rgb)
        for res in results:
            if res['confidence'] < self.detection_config['confidence_threshold']:
                continue
            face, width, corner_1, corner_2 = utils.get_face_width(img_rgb, res['box'])
            # cv2.imwrite('frames/temp.jpg',face)
            cam_dist = 6421 / width
            cam_dist = float(cam_dist)
            if cam_dist < 80 and cam_dist > 21:
                encoding = self._get_encoding(encoder_model, face)
                encoding = Normalizer('l2').transform(
                    encoding.reshape(1, -1))[0]
                name = 'unknown'
                distance = float("inf")
                for db_name, db_encode in self.known_encoding.items():
                    dist = cosine(db_encode, encoding)
                    if dist < self.detection_config['recognition_threshold'] and dist < distance:
                        name = db_name
                        distance = dist 
                print(name)
                if name == 'unknown':
                    logging.info('Unknown recognized in frame, marking attendance')
                    _ =self._unknown_recognized(
                        encoding, img_rgb, corner_1, corner_2, camera)

                else:
                    logging.info('{} recognized in frame, marking attendance'.format(name))
                    _ = self._known_recognized(
                        name, encoding, img_rgb, corner_1, corner_2, camera)
            else:
                logging.info('Move forward towards {}'.format(camera))
                # cv2.rectangle(img_rgb, corner_1, corner_2, (0, 0, 255), 2)
                # cv2.imwrite('frames/' + camera.get_location() + datetime.now().strftime(
                    # ' %-d %b %-y %-I:%-M %p (%f)') + '.jpg', img_rgb)
        return 

    def _unknown_recognized(self, encoding, img, corner_1, corner_2, camera):
        current_time = datetime.now()
        current_dir = self.detection_config['unknown_atttendance_path'] + str(current_time.strftime('%-d %b %-y')) + '/' + str(current_time.strftime('%-I %p')) + '/'
        os.makedirs(current_dir, exist_ok=True)

        unknown_name = None
        if list(self.unknown_encoding):
            distances = [cosine(enc, encoding)
                         for _, enc in self.unknown_encoding.items()]
            unknown_names = [key for key, value in self.unknown_encoding.items() if cosine(value, encoding) == min(distances)]
            if min(distances) < 0.6:
                unknown_name = unknown_names[0]
                if unknown_name in list(self.unkown_attendance_data.keys()):
                    if self.unkown_attendance_data[unknown_name] > 5 and self.unknown_attendance_time[unknown_name] == False:
                        # file_name = current_dir + unknown_name + '-' + amera.get_location() + '-' + current_time.strftime('%-I:%-M %p (%f)') '.jpg'
                        # cv2.imwrite(file_name, img)
                        self.unknown_attendance_time[unknown_name] = True
                    else:
                        self.unkown_attendance_data[unknown_name] += 1
                else:
                    self.unkown_attendance_data[unknown_name] = 1
                    self.unknown_attendance_time[unknown_name] = False
            else:
                self.unknown_counter = self.unknown_counter + 1
                unknown_name = 'Unknown_' + str(self.unknown_counter)
                self.unkown_attendance_data[unknown_name] = 1
                self._update_unknown_attendance(unknown_name, encoding)
                self.unknown_attendance_time[unknown_name] = False

        elif unknown_name == None:
            self.unknown_counter = self.unknown_counter + 1
            unknown_name = 'Unknown_' + str(self.unknown_counter)
            self.unkown_attendance_data[unknown_name] = 1
            self._update_unknown_attendance(unknown_name, encoding)
            self.unknown_attendance_time[unknown_name] = False

        cv2.rectangle(img, corner_1, corner_2, (255, 0, 0), 2)
        cv2.putText(img, unknown_name,
                    (corner_1[0], corner_2[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
        file_name = current_dir + unknown_name + '-' + camera.get_location() + '-' + current_time.strftime('%-I:%-M %p (%f)') + '.jpg'
        logging.info('{} recognized from {}'.format(unknown_name, camera))
        cv2.imwrite(file_name, img)
        return img

    def _known_recognized(self, name, encoding, img, corner_1, corner_2, camera):
        cv2.rectangle(img, corner_1, corner_2, (0, 255, 0), 2)
        cv2.putText(img, name, (corner_1[0], corner_2[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2)

        if name in self.known_attendance_data.keys():
            self.known_attendance_data[name] = self.known_attendance_data[name] + 1
        else:
            self.known_attendance_data[name] = 1
            self.known_attendance_time[name] = datetime.now(
            ) + timedelta(seconds=180)

        current_time = datetime.now()
        current_dir = self.detection_config['known_atttendance_path'] + current_time.strftime('%-d %b %-y') + '/' + current_time.strftime('%-I %p') + '/'
        os.makedirs(current_dir, exist_ok=True)

        if self.known_attendance_data[name] > 2 and (current_time - self.known_attendance_time[name]).seconds >= 180:
            file_name = current_dir + name + '-' + camera.get_location() + '-' + current_time.strftime('%-I:%-M %p (%f)') + '.jpg'
            cv2.imwrite(file_name, img)
            self._update_known_attendance(name, camera, file_name)
            logging.info(
                '{}\'s attendance recorded from {}'.format(name, camera))
            self.known_attendance_time[name] = datetime.now()
        return img

    def _get_encoding(self, encoder_model, face):
        face = utils.normalize(face)
        face = cv2.resize(face, tuple(
            self.detection_config['recognized_face_shape']))
        with self.sess.graph.as_default():
            encoding = encoder_model.predict(np.expand_dims(face, axis=0))[0]
        return encoding

    def run(self):
        logging.info('Loading models on GPU')
        config = tfv1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tfv1.Session(config=config)

        with self.sess.graph.as_default():
            K.set_session(tfv1.Session(config=config))
            detector_model = MTCNN()
            encoder_model = load_model(self.encoder_model_path, compile=False)
        logging.info('Running RecognitionModel process, waiting for producers to start')
        
        self._read_unknown_encodings()
        self._read_known_encodings()
        self.consumer_process_ready.set()

        while not self.producer_process_ready.is_set():
            pass
        
        logging.info('Resuming RecognitionModel process')
        
        BPS_COUNTER_INTERVAL = 1 # minutes
        current_bps_counter_time = datetime.now()
        next_bps_counter_time = current_bps_counter_time + timedelta(minutes=BPS_COUNTER_INTERVAL)
        consumed_batches = 0
        
        while not self.quit_event.is_set():
            if self.update_encodings.is_set():
                logging.info('Update encoding event is set, updating encodings')
                self._read_unknown_encodings()
                self._read_known_encodings()
                self.update_encodings.clear()
            try:
                batch_file_name = self.buffer.get(timeout=10)
            except queue.Empty:
                self.quit_event.set()
                break
            with open(batch_file_name, 'rb') as file:
                batch = pickle.load(file)
            os.remove(batch_file_name)
            
            print('Received batch {}'.format(batch_file_name))
            
            for frame in batch['frames']:
                img = Image.fromarray(frame.numpy())
                img = np.array(img)
                self._recognize(detector_model, encoder_model,
                                img, frame.get_camera())
            self.buffer.task_done()
            
            consumed_batches += 1
            if datetime.now() >= next_bps_counter_time:
                delta_seconds = (datetime.now() - current_bps_counter_time).seconds
                bps = consumed_batches / delta_seconds 
                print('RecognitionModel consuming {:4.1f} bps'.format(bps))
                print('RecognitionModel consuming {:4.1f} fps'.format(bps*self.BATCH_SIZE))
                current_bps_counter_time = datetime.now()
                next_bps_counter_time = current_bps_counter_time + timedelta(minutes=BPS_COUNTER_INTERVAL)
                consumed_batches = 0

        logging.info('Terminated RecognitionModel process')
