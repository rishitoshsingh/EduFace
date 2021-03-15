import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datetime import datetime
import logging
from systemd.journal import JournalHandler
import signal

# for logging in journalctl
logger = logging.getLogger(__name__)
journald_handler = JournalHandler()
journald_handler.setFormatter(logging.Formatter(
    '[%(levelname)s] %(message)s'
))
logger.addHandler(journald_handler)

logging.basicConfig(filename='eduface.log', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.info('=====================================================================\n\n\nEduFace started at {}'.format(datetime.now()))

from pipeline.manager import RecognitionManager, BatchRecognitionManager, BatchPicklingManager
import json

# reading detection_config file
with open('detection_config.json','r') as file:
    detection_config = json.load(file)

# reading cameras details
with open('cameras.json','r') as file:
    cameras_dicts = json.load(file)
    
with open('motion_configs.json','r') as file:
    motion_configs = json.load(file)
    
encoder_model_path = 'data/model/facenet_keras.h5'
# manager = RecognitionManager(cameras_dicts, detection_config, encoder_model_path, MAX_BUFFER = 14400)
manager = BatchRecognitionManager(cameras_dicts, motion_configs, detection_config, encoder_model_path, MAX_BUFFER = 500)
# manager = BatchPicklingManager(cameras_dicts, motion_configs, MAX_BUFFER = 500)

manager.start()
print('EduFace Started')

# signal manager to terminate EduFace
def terminate_manager_pipeline(signalNum, frame):
    logging.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nEduFace received termination signal from system')
    manager.terminate()
signal.signal(signal.SIGTERM, terminate_manager_pipeline)

# pause main process and wait for signal from os
signal.pause()
