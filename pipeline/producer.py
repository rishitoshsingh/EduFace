import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
import cv2

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
gi.require_version('GstVideo', '1.0')

from gi.repository import Gst, GLib, GstApp, GstVideo
Gst.init()

# import gstreamer.utils as utils
import pipeline.utils as utils

from models import Frame, Camera
# from pipeline.buffer import Buffer 

import logging

class StreamCamera(multiprocessing.Process):
    def __init__(self, 
                 camera: Camera, 
                 motion_config: dict, 
                 buffer: multiprocessing.JoinableQueue, 
                 quit_event: multiprocessing.Event, 
                 max_idle: int = 14400):
        """Iniitalize StreamCamera object

        Args:
            camera (models.Camera): Camera which need to be streamed
            motion_config (dict): dictionary containing coordinates of motion triggering area
            buffer (multiprocessing.JoinableQueue): used to insert frame for processing captured from camera
            quit_event (multiprocessing.Event): used to trigger process to stop
            max_idle (int, optional): to trigger process to stop if no frame is received for max_idle times (default: 14400 fames or 10 minutes ) 
        """
        super(StreamCamera,self).__init__(name=camera.get_location())
        logging.info('Creating StreamCamera process for {}'.format(camera))
        self.camera = camera
        self.motion_trigger_coordinates = [tuple(motion_config['corner_1']), tuple(motion_config['corner_2']) ]
        self.motion_trigger_area_fraction = 0.5
        width, height = abs(self.motion_trigger_coordinates[0][0] - self.motion_trigger_coordinates[1][0]), abs(self.motion_trigger_coordinates[0][1] - self.motion_trigger_coordinates[1][1]) 
        self.detection_area_threshold = abs(0.5*width*height)
        self.motion_detected = False 
        self.motion_timeout = None
        self.buffer = buffer
        self.quit_event = quit_event
        self.MAX_IDLE = max_idle
        self.current_idle = 0
        # self.main_loop = GLib.MainLoop()
        # self.Gthread = Thread(target=self.main_loop.run)
        # self.Gthread.start()
        # self.pipeline = Gst.parse_launch("rtspsrc location={} ! autovideosink".format(self.camera.get_stream()))
        pipeline_str = "uridecodebin uri={} uridecodebin0::source::latency=300 ! videoscale ! video/x-raw,width=1280,height=720 ! videoconvert !  video/x-raw, format=RGB ! appsink name={}".format(self.camera.get_stream(), self.camera.get_id())
        # pipeline_str = "rtspsrc location={} ! queue ! rtph265depay ! h265parse ! avdec_h265 ! videoconvert !  video/x-raw, format=RGB ! appsink name={}".format(self.camera.get_stream(), self.camera.get_id())
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name("{}".format(self.camera.get_id()))
        
    def _detect_motion(self, old_image, new_image):
        """function to detect motion

        Args:
            old_image (np.ndarray): old cropped image 
            new_image (np.ndarray): current cropped image

        Returns:
            bool: wether motion was detected or not
        """
        diff = cv2.absdiff(old_image, new_image)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            area = w*h
            if area < self.detection_area_threshold:
                continue
            self.motion_detected = True
            # logging.info('Motion detected in {}'.format(self.camera))
            self.motion_timeout = datetime.now() + timedelta(seconds=3)
            return True
        return False

    def _get_image_from_sample(self, sample:Gst.Sample):
        """Retreive image from Gst.Sample

        Args:
            sample (Gst.Sample): sample object received from gstream pipeline

        Returns:
            np.ndarray, tuple: image and it shape of image
        """
        # for extracting frame shape
        caps_format = sample.get_caps().get_structure(0)  # Gst.Structure
        frmt_str = caps_format.get_value('format') 
        video_format = GstVideo.VideoFormat.from_string(frmt_str)
        w, h = caps_format.get_value('width'), caps_format.get_value('height')
        c = utils.get_num_channels(video_format)
        shape = (h, w, c)

        # for extracting frame
        buff = sample.get_buffer()

        image = np.ndarray(shape=shape, buffer=buff.extract_dup(0, buff.get_size()), dtype=utils.get_np_dtype(video_format))
        image = np.squeeze(image)
        image = Image.fromarray(image)
        image = np.array(image)
        return image, shape
    
    def _crop_image(self, image: np.ndarray):
        """crop required area from image 

        Args:
            image (np.ndarray): full resolution image

        Returns:
            np.ndarray: cropped image 
        """
        return image[self.motion_trigger_coordinates[0][1] : self.motion_trigger_coordinates[1][1], self.motion_trigger_coordinates[0][0] : self.motion_trigger_coordinates[1][0] ]

    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        logging.info('Running StreamCamera proces for {}'.format(self.camera))
        
        FPS_COUNTER_INTERVAL = 1 # minutes
        current_fps_counter_time = datetime.now()
        next_fps_counter_time = current_fps_counter_time + timedelta(minutes=FPS_COUNTER_INTERVAL)
        received_frames = 0
        sent_frames = 0
        
        # for extracting first frame
        while True:
            old_sample = self.appsink.try_pull_sample(Gst.SECOND)
            if old_sample is None:
                self.current_idle += 1
                if self.current_idle > self.MAX_IDLE:
                    logging.info('No frames received for {} seconds, terminating StreamCamera for {}'.format(self.MAX_IDLE, self.camera))
                    self.quit_event.set()
                    break
                continue
            else:
                break
        self.current_idle = 0
        
        old_image, shape = self._get_image_from_sample(old_sample)
        cropped_old_image = self._crop_image(old_image)
        
        while not self.quit_event.is_set():
            sample = self.appsink.try_pull_sample(Gst.SECOND)
            if sample is None:
                self.current_idle += 1
                if self.current_idle > self.MAX_IDLE:
                    logging.info('No frames received for {} seconds, terminating StreamCamera for {}'.format(self.MAX_IDLE, self.camera))
                    break
                continue
            self.current_idle = 0
            
            image, shape = self._get_image_from_sample(sample)
            cropped_image = self._crop_image(image)
            
            if ( self.motion_detected or self._detect_motion(cropped_old_image, cropped_image) ) and (datetime.now() <= self.motion_timeout):
                frame = Frame(self.camera, datetime.now(), cropped_image, cropped_image.shape)
                self.buffer.put(frame)
                sent_frames += 1
            else:
                self.motion_detected = False
            
            cropped_old_image = cropped_image

            received_frames += 1
            if datetime.now() >= next_fps_counter_time:
                delta_seconds = (datetime.now() - current_fps_counter_time).seconds 
                print('{} producing {:4.1f} fps'.format(self.camera, received_frames / delta_seconds))
                print('{} sent {:4.1f} fps'.format(self.camera, sent_frames / delta_seconds))
                current_fps_counter_time = datetime.now()
                next_fps_counter_time = current_fps_counter_time + timedelta(minutes=FPS_COUNTER_INTERVAL)
                received_frames = 0
                sent_frames = 0                
        
        # changing state of elements if proces is terminated
        self.pipeline.set_state(Gst.State.NULL)
        # self.main_loop.quit()
        logging.info('Terminated StreamCamera process for {}'.format(self.camera))
