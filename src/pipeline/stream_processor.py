import cv2
import threading
import time
import numpy as np

class StreamProcessor:
    def __init__(self, camera_id=0, max_fps=30):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        
        # Optimize camera settings for low latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, max_fps)
        
        self.frame = None
        self.ret = False
        self.running = False
        self.lock = threading.Lock()
        
    def start(self):
        """Starts background thread to read frames to ensure buffer is always fresh."""
        if not self.cap.isOpened():
            print(f"Warning: Could not open camera {self.camera_id}. Using mock video stream.")
            self.running = True
            self.thread = threading.Thread(target=self._mock_update, args=())
            self.thread.daemon = True
            self.thread.start()
            while self.frame is None and self.running:
                time.sleep(0.01)
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()
        
        # Wait for first frame
        while self.frame is None and self.running:
            time.sleep(0.01)

    def _mock_update(self):
        """Generates a moving synthetic circle as a mock video stream."""
        x = 0
        while self.running:
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
            cv2.circle(frame, (x % 640, 240), 50, (100, 100, 100), -1)
            x += 10
            with self.lock:
                self.ret = True
                self.frame = frame
            time.sleep(1.0 / 30.0)
            
    def _update(self):
        """Reads frames as fast as possible."""
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
                
    def read(self):
        """Returns the most recent frame."""
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None
            
    def stop(self):
        """Stops the capture thread."""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        self.cap.release()
