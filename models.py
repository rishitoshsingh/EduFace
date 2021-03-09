import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Camera:
    """Data class to store camera details
    """
    def __init__(self, cameraId, location, uri):
        """Initialize camera object 

        Args:
            cameraId (int): Id for camera
            location (str): Location where camera is installed
            uri (str): streaming url for camera
        """
        self.cameraId = cameraId
        self.location = location
        self.uri = uri
    
    def get_stream(self):
        return self.uri

    def get_id(self):
        return self.cameraId

    def get_location(self):
        return self.location
    
    def __format__(self,format):
        if format == 'loc':
            return "{}({!r})".format(self.__class__.__name__,self.location)
        else:
            return "{}({!r})".format(self.__class__.__name__,self.cameraId)

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self.cameraId)


class Frame():
    """Data class to store frame details
    """
    def __init__(self, camera, timestamp, array, shape):
        """Iniitalize Frame objecct

        Args:
            camera (Camera): Camera from which frame was catputed
            timestamp (datetime.datetime): timestamp when frame was captured
            array (np.ndarray): captured frame
            shape (tuple): shape of frame's array
        """
        self.camera = camera
        self.timestamp = timestamp
        self.array = array
        self.shape = shape

    def get_camera(self):
        return self.camera
    
    def numpy(self):
        return self.array
    
    def get_shape(self):
        return self.shape

    def get_timestamp_fmt(self):
        """returns timestamp string

        Returns:
            str: timestamp string
        """
        return self.timestamp.isoformat(sep=' ', timespec='milliseconds')

    def __repr__(self):
        return "{}( {}, {!r})".format(self.__class__.__name__, self.camera,  self.get_timestamp_fmt())