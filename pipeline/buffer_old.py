import multiprocessing

from models import Frame, Camera

class Buffer(multiprocessing.Queue):
    def __init__(self, max_frames:int=1440):
        """
        Args:
            max_frames (int, optional): maximum number of frames that can be inserted in queue. Defaults to 1440 (1 minute video frames).
        """
        self.MAX_FRAMES = max_frames
        super().__init__(maxsize=self.MAX_FRAMES)

    def get_frame(self) -> Frame:
        """Get first frame from bufer

        Returns:
            models.Frame: returns a Frame object
        """
        frame = self.get()
        return frame

    def insert_frame(self, frame: Frame, camera: Camera):
        """Insert new frame into buffer

        Args:
            frame (models.Frame): Frame object to be inserted in buffer
            camera (models.Camera): Camera from which frame is captured
        """
        self.put(frame)
