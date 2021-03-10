import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datetime import datetime
import logging

logging.basicConfig(filename='eduface.log', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logging.info('====================================================================\n\n\nEduFace started at {}'.format(datetime.now()))

from pipeline.manager import RecognitionManager, BatchRecognitionManager, BatchPicklingManager
import json
import time

# reading detection_config file
with open('detection_config.json','r') as file:
    detection_config = json.load(file)

# reading cameras details
with open('cameras.json','r') as file:
    cameras_dicts = json.load(file)
    
encoder_model_path = 'data/model/facenet_keras.h5'
# manager = RecognitionManager(cameras_dicts, detection_config, encoder_model_path, MAX_BUFFER = 14400)
# manager = BatchRecognitionManager(cameras_dicts, detection_config, encoder_model_path, MAX_BUFFER = 500)
manager = BatchPicklingManager(cameras_dicts, MAX_BUFFER = 500)
manager.start()

try:
    print('Running EduFace, Press ctrl+C to exit')
    while True:
        pass
except KeyboardInterrupt:
    print('Closing EduFace')

manager.terminate()
manager.kill()
