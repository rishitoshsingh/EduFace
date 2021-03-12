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
    """ A consumer to save frames from queue to disk at given directory. (Currently not used in final EduFace pipeline)
    """
    def __init__(self, buffer: multiprocessing.JoinableQueue, quit_event: multiprocessing.Event, directory='frames/'):
        """Iniitalize Consumer object

        Args:
            buffer (multiprocessing.JoinableQueue): Used to insert captured frame to get processed
            quit_event (multiprocessing.Event): used to trigger process to stop
            directory (str): dricetory where to save frames
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
    """This consumer read all frames from inBuffer, generate batches and pickel them into disks. \
        These pickeled batch file names are inserted in outBuffer (if not None) for processing by other consumer.
    """
    def __init__(self,
                 inBuffer: multiprocessing.JoinableQueue,
                 batch_size: int,
                 batch_timeout_minutes:int,
                 outBuffer: multiprocessing.JoinableQueue,
                 quit_event: multiprocessing.Event,
                 directory='batches/'):
        """Iniitalize Consumer object

        Args:
            inBuffer (multiprocessing.JoinableQueue): Used to insert captured frame to get processed
            batch_size (int): batch size 
            batch_timeout_minutes (int): time limit on batch (in minutes)
            outBuffer (multiprocessing.JoinableQueue): Used to insert file names of batches saved in disk
            quit_event (multiprocessing.Event): used to trigger process to stop
            diectory (str): directory where pickled objects are saved 
        """
        super(BatchGeneratorAndPiclker, self).__init__()
        logging.info('Creating BatchGeneratorAndPiclker process')
        self.inBuffer = inBuffer
        self.BATCH_SIZE = batch_size
        self.BATCH_TIMEOUT_MINUTES = batch_timeout_minutes
        self.quit_event = quit_event
        self.outBuffer = outBuffer
        self.directory = directory
        
    def _generate_batch(self,batch_id:int):
        """ Generate and initialize new batch to save incoming frames from inBuffer

        Args:
            batch_id (int): unique id for batch

        Returns:
            dict: new generated frames batch
        """
        batch = {}
        batch['id'] = batch_id
        batch['timestamp'] = datetime.now()
        batch['end_timestamp'] = datetime.now() + timedelta(minutes=self.BATCH_TIMEOUT_MINUTES)
        batch['batch_size'] = self.BATCH_SIZE
        batch['num_frames'] = 0
        batch['frames'] = []
        return batch

    def run(self):
        logging.info('Running BatchGeneratorAndPiclker process')
        batch_count = 1
        batch = self._generate_batch(batch_count)
        
        # to calculate consumed fps
        FPS_COUNTER_INTERVAL = 1 # minutes
        current_fps_counter_time = datetime.now()
        next_fps_counter_time = current_fps_counter_time + timedelta(minutes=FPS_COUNTER_INTERVAL)
        consumed_frames = 0
        
        # retreiving frames till quit_event is not set
        while not self.quit_event.is_set():
            frame = self.inBuffer.get(60)
            batch['frames'].append(frame)
            batch['num_frames'] += 1
            self.inBuffer.task_done()
            
            consumed_frames += 1
            # printing consuming fps every FPS_COUNTER_INTERVAL
            if datetime.now() >= next_fps_counter_time:
                delta_seconds = (datetime.now() - current_fps_counter_time).seconds 
                print('BatchGeneratorAndPiclker consuming {:4.1f} fps'.format(consumed_frames / delta_seconds))
                current_fps_counter_time = datetime.now()
                next_fps_counter_time = current_fps_counter_time + timedelta(minutes=FPS_COUNTER_INTERVAL)
                consumed_frames = 0
            
            # saving current batch till BATCH_SIZE frames have been inserted or BATCH_TIMEOUT_MINUTES have been triggered
            if batch['num_frames'] == self.BATCH_SIZE or batch['end_timestamp'] <= datetime.now():
                file_name = self.directory + 'batch-{}'.format(batch['id']) + ' ({})'.format(batch['timestamp'].strftime('%-d %b %-y, %-I:%-M %p (%f)')) + '.pickle'
                with open(file_name, 'wb') as file:
                    pickle.dump(batch, file)
                if self.outBuffer is not None:
                    self.outBuffer.put(file_name)
                batch_count += 1
                batch = self._generate_batch(batch_count)
                
        logging.info('Terminated BatchGeneratorAndPiclker process')


class ViewStream(multiprocessing.Process):
    """A consumer to view streams from all camera producers in seperate windows. Currently not used in EduFace pipeline.
    """
    def __init__(self,
                 buffer: multiprocessing.JoinableQueue,
                 quit_event: multiprocessing.Event):
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
            img = cv2.cvtColor(frame.numpy(), cv2.COLOR_BGR2RGB)
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
    """A consumer to detect and recognize faces in frames, if found it will be attendance will be inserted in database
    """

    def __init__(self,
                 buffer: multiprocessing.JoinableQueue,
                 consumer_process_ready: multiprocessing.Event,
                 producer_process_ready: multiprocessing.Event, 
                 update_encodings: multiprocessing.Event, 
                 encoder_model_path: str,
                 quit_event: multiprocessing.Event, 
                 **detection_config):
        """Create RecognitionModel consumer process.

        Args:
            buffer (multiprocessing.JoinableQueue): bffer from which file names of pickled objects are read.
            consumer_process_ready (multiprocessing.Event): an event to check wether (this) consumer is ready or not, if ready then only all producers will send frames.
            producer_process_ready (multiprocessing.Event): an event to check wether producers are ready or not, if producers are not ready, then (this) consumer will wait for them to be ready.
            update_encodings (multiprocessing.Event): an event to check when to update encodings from database
            encoder_model_path (str): encoder model path
            quit_event (multiprocessing.Event): an event to check when to terminate this consumer process
        """
        
        super(RecognitionModel, self).__init__()
        logging.info('Creating RecognitionModel process')
        self.buffer = buffer

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

    def _update_known_attendance(self, name: str, frame:Frame, file_name: str):
        mydb = self._get_mysql_connection()
        mycursor = mydb.cursor()
        sql = "SELECT id FROM labours id where name = %s"
        mycursor.execute(sql, (name, ))
        labour_id = int(mycursor.fetchall()[0][0])
        mydb.commit()

        camera = frame.get_camera()
        sql = """INSERT INTO attendances (labour_id, camera, location, timestamp, photo, created_at) VALUES (%s, %s, %s, %s, %s, %s)"""
        val = (labour_id, camera.get_id(), camera.get_location(), frame.datetime(), file_name, datetime.now())
        mycursor.execute(sql, val)
        mydb.commit()

    def _recognize(self, detector_model, encoder_model, frame: Frame):
        """private method to detect and recognize face in frame

        Args:
            detector_model (MTCNN): MTCNN model to detect faces in a frame
            encoder_model (Model): tensorflow model to encode face into vector
            frame (Frame): frame to be analyzed
        """
        img_rgb = cv2.cvtColor(frame.numpy(), cv2.COLOR_BGR2RGB)
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

                # finding cosnie distance
                for db_name, db_encode in self.known_encoding.items():
                    dist = cosine(db_encode, encoding)
                    if dist < self.detection_config['recognition_threshold'] and dist < distance:
                        name = db_name
                        distance = dist 
                print(name)
                if name == 'unknown':
                    logging.info('Unknown recognized in frame, marking attendance')
                    _ =self._unknown_recognized(
                        encoding, img_rgb, frame, corner_1, corner_2)

                else:
                    logging.info('{} recognized in frame, marking attendance'.format(name))
                    _ = self._known_recognized(
                        name, encoding, img_rgb, frame, corner_1, corner_2)
            else:
                logging.info('Move forward towards {}'.format(camera))
                # cv2.rectangle(img_rgb, corner_1, corner_2, (0, 0, 255), 2)
                # cv2.imwrite('frames/' + camera.get_location() + datetime.now().strftime(
                    # ' %-d %b %-y %-I:%-M %p (%f)') + '.jpg', img_rgb)
        return 

    def _unknown_recognized(self, encoding, img, frame:Frame, corner_1, corner_2):
        
        # creating directory gto store frame
        current_time = frame.datetime()
        current_dir = self.detection_config['unknown_atttendance_path'] + str(current_time.strftime('%-d %b %-y')) + '/' + str(current_time.strftime('%-I %p')) + '/'
        os.makedirs(current_dir, exist_ok=True)

        unknown_name = None
        # if some unknowns are already recognized
        if list(self.unknown_encoding):
            # finding cosine distance
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
        file_name = current_dir + unknown_name + '-' + frame.get_camera().get_location() + '-' + current_time.strftime('%-I:%-M %p (%f)') + '.jpg'
        logging.info('{} recognized from {}'.format(unknown_name, frame.get_camera()))
        cv2.imwrite(file_name, img)
        return img

    def _known_recognized(self, name, encoding, img, frame:Frame, corner_1, corner_2):
        cv2.rectangle(img, corner_1, corner_2, (0, 255, 0), 2)
        cv2.putText(img, name, (corner_1[0], corner_2[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2)

        if name in self.known_attendance_data.keys():
            self.known_attendance_data[name] = self.known_attendance_data[name] + 1
        else:
            self.known_attendance_data[name] = 1
            self.known_attendance_time[name] = datetime.now(
            ) + timedelta(seconds=180)

        # creating directory to save image
        current_time = frame.datetime()
        current_dir = self.detection_config['known_atttendance_path'] + current_time.strftime('%-d %b %-y') + '/' + current_time.strftime('%-I %p') + '/'
        os.makedirs(current_dir, exist_ok=True)

        if self.known_attendance_data[name] > 2 and (current_time - self.known_attendance_time[name]).seconds >= 180:
            file_name = current_dir + name + '-' + frame.get_camera().get_location() + '-' + current_time.strftime('%-I:%-M %p (%f)') + '.jpg'
            cv2.imwrite(file_name, img)
            self._update_known_attendance(name, frame, file_name)
            logging.info(
                '{}\'s attendance recorded from {}'.format(name, frame.get_camera()))
            self.known_attendance_time[name] = frame.datetime()
        return img

    def _get_encoding(self, encoder_model, face):
        """private funcion to get encoding using encoder model

        Args:
            encoder_model (Model): tensorflow encoder model 
            face (np.ndarray): face to be encoded

        Returns:
            np.ndarray: face encoding
        """
        face = utils.normalize(face)
        face = cv2.resize(face, tuple(
            self.detection_config['recognized_face_shape']))
        with self.sess.graph.as_default():
            encoding = encoder_model.predict(np.expand_dims(face, axis=0))[0]
        return encoding

    def run(self):
        """ main process function. First all models are loaded in GPU and (this) consumer will wait for producers till they are ready.
        """
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

        # checlking and waiting for producer(s) to be ready
        while not self.producer_process_ready.is_set():
            pass
        
        # now producer(s) are ready and sending frames to buffer, resuming (this) consumer
        logging.info('Resuming RecognitionModel process')
        
        # for calculating bps (batch per second) consumed
        BPS_COUNTER_INTERVAL = 1 # minute
        current_bps_counter_time = datetime.now()
        next_bps_counter_time = current_bps_counter_time + timedelta(minutes=BPS_COUNTER_INTERVAL)
        consumed_batches = 0
        consumed_frames = 0
        
        # loop to retreive file names of batch files, deserialize them, detect and recognize faces and update attendance database
        while not self.quit_event.is_set():
            
            # checking if encoding needs to be updated or not
            if self.update_encodings.is_set():
                logging.info('Update encoding event is set, updating encodings')
                self._read_unknown_encodings()
                self._read_known_encodings()
                self.update_encodings.clear()
            
            # try to retrieve data with timeout of 10 seconds. If buffer is empty, it will throw queue.Empty exception.
            try:
                batch_file_name = self.buffer.get(timeout=10)
            except queue.Empty:
                # if queue is empty, check wether quit_event is set or not
                if self.quit_event.is_set():
                    break
                else:
                    continue
            # deserializing and deleting batch file
            with open(batch_file_name, 'rb') as file:
                batch = pickle.load(file)
            os.remove(batch_file_name)
            
            print('Received batch {}'.format(batch_file_name))
            
            for frame in batch['frames']:
                self._recognize(detector_model, encoder_model, frame)
            self.buffer.task_done()
            
            # to print fps and bps every BPS_COUNTER_INTERVAL
            consumed_batches += 1
            consumed_frames += batch['num_frames']
            if datetime.now() >= next_bps_counter_time:
                delta_seconds = (datetime.now() - current_bps_counter_time).seconds
                bps = consumed_batches / delta_seconds 
                fps = consumed_frames / delta_seconds
                print('RecognitionModel consuming {:4.1f} bps'.format(bps))
                print('RecognitionModel consuming {:4.1f} fps'.format(fps))
                current_bps_counter_time = datetime.now()
                next_bps_counter_time = current_bps_counter_time + timedelta(minutes=BPS_COUNTER_INTERVAL)
                consumed_batches = 0

        logging.info('Terminated RecognitionModel process')
