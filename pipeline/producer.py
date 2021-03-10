import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing
from datetime import datetime, timedelta
import numpy as np

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
    def __init__(self, camera: Camera, buffer: multiprocessing.JoinableQueue, quit_event: multiprocessing.Event, max_idle: int = 14400):
        """Iniitalize StreamCamera object

        Args:
            camera (models.Camera): Camera which need to be streamed
            buffer (multiprocessing.JoinableQueue): used to insert frame for processing captured from camera
            quit_event (multiprocessing.Event): used to trigger process to stop
            max_idle (int, optional): to trigger process to stop if no frame is received for max_idle times (default: 14400 fames or 10 minutes ) 
        """
        super(StreamCamera,self).__init__(name=camera.get_location())
        logging.info('Creating StreamCamera process for {}'.format(camera))
        self.camera = camera
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

    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        logging.info('Running StreamCamera proces for {}'.format(self.camera))
        
        FPS_COUNTER_INTERVAL = 1 # minutes
        current_fps_counter_time = datetime.now()
        next_fps_counter_time = current_fps_counter_time + timedelta(minutes=FPS_COUNTER_INTERVAL)
        produced_frames = 0
        
        while not self.quit_event.is_set():
            sample = self.appsink.try_pull_sample(Gst.SECOND)
            if sample is None:
                self.current_idle += 1
                if self.current_idle > self.MAX_IDLE:
                    logging.info('No frames received for {} seconds, terminating StreamCamera for {}'.format(self.MAX_IDLE, self.camera))
                    break
                continue
            self.current_idle = 0
            # for extracting frame shape
            caps_format = sample.get_caps().get_structure(0)  # Gst.Structure
            frmt_str = caps_format.get_value('format') 
            video_format = GstVideo.VideoFormat.from_string(frmt_str)
            w, h = caps_format.get_value('width'), caps_format.get_value('height')
            c = utils.get_num_channels(video_format)
            shape = (h, w, c)

            # for extracting frame
            buff = sample.get_buffer()

            frame_array = np.ndarray(shape=(h, w, c), buffer=buff.extract_dup(0, buff.get_size()), dtype=utils.get_np_dtype(video_format))
            frame_array = np.squeeze(frame_array)  

            frame = Frame(self.camera, datetime.now(), frame_array, shape)
            self.buffer.put(frame)

            produced_frames += 1
            if datetime.now() >= next_fps_counter_time:
                delta_seconds = (datetime.now() - current_fps_counter_time).seconds 
                print('{} producing {:4.1f} fps'.format(self.camera, produced_frames / delta_seconds))
                current_fps_counter_time = datetime.now()
                next_fps_counter_time = current_fps_counter_time + timedelta(minutes=FPS_COUNTER_INTERVAL)
                produced_frames = 0
                
        
        # shanging state of elements if proces is terminated
        self.pipeline.set_state(Gst.State.NULL)
        # self.main_loop.quit()
        logging.info('Terminated StreamCamera process for {}'.format(self.camera))
