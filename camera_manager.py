# camera_manager.py
import cv2
import time
import logging
import os

# Attempt to import config, with fallback for standalone testing
try:
    import config as config
    logger = logging.getLogger(__name__)
    CONFIG_MODULE_LOADED = True
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("CameraManager_standalone")
    class FallbackConfig:
        SHOW_VIDEO_WINDOW = True
    config = FallbackConfig()
    CONFIG_MODULE_LOADED = False
    logger.info("Running camera_manager.py in standalone mode with fallback configuration for testing.")

class Camera:
    def __init__(self, source, camera_id="cam0"):
        """
        Initializes the Camera object.
        Args:
            source (str or int): The video source (RTSP URL, file path, or camera index).
            camera_id (str): A unique identifier for this camera instance (for logging).
        """
        self.source = source
        self.camera_id = camera_id
        self.cap = None
        self.is_connected = False
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0
        self.connect_attempts = 0
        self.max_connect_attempts = 5
        self.last_attempt_time = 0
        self.reconnect_delay_base = 5
        self.connect()

    def connect(self):
        """
        Establishes or re-establishes connection to the video source.
        Sets an environment variable to tell OpenCV's FFmpeg backend to prefer TCP for RTSP.
        """
        self.last_attempt_time = time.time()
        self.connect_attempts += 1
        logger.info(f"[{self.camera_id}] Attempting to connect (attempt {self.connect_attempts}). Source: {self.source}")
        original_ffmpeg_options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        logger.debug(f"[{self.camera_id}] Set OPENCV_FFMPEG_CAPTURE_OPTIONS to 'rtsp_transport;tcp'")
        try:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                logger.error(f"[{self.camera_id}] Error: Could not open video stream from '{self.source}' even with TCP preference and FFMPEG backend.")
                self.cap = None
                self.is_connected = False
            else:
                self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.fps == 0:
                    logger.warning(f"[{self.camera_id}] Camera reported FPS as 0. Using a default of 15 for internal logic if needed.")
                logger.info(f"[{self.camera_id}] Successfully connected to camera. Resolution: {self.frame_width}x{self.frame_height}, Reported FPS: {self.fps:.2f}")
                self.is_connected = True
                self.connect_attempts = 0
        except Exception as e:
            logger.error(f"[{self.camera_id}] Exception during VideoCapture initialization for '{self.source}': {e}", exc_info=False)
            self.cap = None
            self.is_connected = False
        finally:
            if original_ffmpeg_options is None:
                if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
                    del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
                    logger.debug(f"[{self.camera_id}] Cleared OPENCV_FFMPEG_CAPTURE_OPTIONS.")
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = original_ffmpeg_options
                logger.debug(f"[{self.camera_id}] Restored OPENCV_FFMPEG_CAPTURE_OPTIONS to original value: '{original_ffmpeg_options}'.")

    def read_frame(self):
        """
        Reads a frame from the video stream. Handles reconnection attempts with exponential backoff.
        Returns (ret, frame): ret is True if frame is valid, frame is the image array or None.
        """
        if not self.is_connected or self.cap is None:
            delay_seconds = self.reconnect_delay_base * (2 ** min(self.connect_attempts, 4))
            delay_seconds = min(delay_seconds, 60)
            if time.time() - self.last_attempt_time > delay_seconds:
                if self.connect_attempts < self.max_connect_attempts:
                    logger.warning(f"[{self.camera_id}] Camera not connected. Attempting to reconnect (next attempt in {delay_seconds:.0f}s)...")
                    self.connect()
                else:
                    logger.error(f"[{self.camera_id}] Camera not connected after {self.max_connect_attempts} rapid attempts. Will retry after a longer pause ({delay_seconds:.0f}s).")
                    self.last_attempt_time = time.time()
            if not self.is_connected:
                return False, None
        ret, frame = self.cap.read()
        if not ret:
            logger.error(f"[{self.camera_id}] Error: cap.read() returned False. Can't receive frame. Stream might have ended or camera disconnected.")
            self.is_connected = False
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            return False, None
        return ret, frame

    def release(self):
        """
        Releases the video capture object.
        """
        if self.cap is not None:
            logger.info(f"[{self.camera_id}] Releasing camera capture object for source: {self.source}")
            self.cap.release()
            self.is_connected = False
            self.cap = None

    def get_properties(self):
        """
        Returns a dictionary of camera properties.
        """
        if self.is_connected and self.cap is not None:
            return {
                "source": self.source,
                "width": self.frame_width,
                "height": self.frame_height,
                "fps": self.fps,
                "connected": self.is_connected
            }
        return {"source": self.source, "status": "disconnected", "connected": self.is_connected}
