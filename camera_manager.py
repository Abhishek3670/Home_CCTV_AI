# camera_manager.py
import cv2
import time
import logging
import os

# Attempt to import config, with fallback for standalone testing
try:
    import config 
    logger = logging.getLogger(__name__)
    CONFIG_MODULE_LOADED = True
except ImportError:
    # Fallback logging and config if config.py is not found
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("CameraManager_standalone")
    class FallbackConfig:
        SHOW_VIDEO_WINDOW = True # Default to True for standalone test if X server is likely
        # Add any other config attributes this file might directly use if config.py is missing
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
        self.max_connect_attempts = 5 # Max attempts before pausing longer
        self.last_attempt_time = 0
        self.reconnect_delay_base = 5 # seconds

        self.connect()

    def connect(self):
        """
        Establishes or re-establishes connection to the video source.
        Sets an environment variable to tell OpenCV's FFmpeg backend to prefer TCP for RTSP.
        """
        self.last_attempt_time = time.time()
        self.connect_attempts += 1
        logger.info(f"[{self.camera_id}] Attempting to connect (attempt {self.connect_attempts}). Source: {self.source}")

        # Store original FFMPEG options if any, then set to prefer TCP for RTSP
        original_ffmpeg_options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        logger.debug(f"[{self.camera_id}] Set OPENCV_FFMPEG_CAPTURE_OPTIONS to 'rtsp_transport;tcp'")

        try:
            # Explicitly request the FFMPEG backend for RTSP streams
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

            if not self.cap.isOpened():
                logger.error(f"[{self.camera_id}] Error: Could not open video stream from '{self.source}' even with TCP preference and FFMPEG backend.")
                self.cap = None # Ensure cap is None if connection failed
                self.is_connected = False
            else:
                self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.fps == 0: # Some cameras might not report FPS correctly or stream is static
                    logger.warning(f"[{self.camera_id}] Camera reported FPS as 0. Using a default of 15 for internal logic if needed.")
                logger.info(f"[{self.camera_id}] Successfully connected to camera. Resolution: {self.frame_width}x{self.frame_height}, Reported FPS: {self.fps:.2f}")
                self.is_connected = True
                self.connect_attempts = 0 # Reset attempts on successful connection
        except Exception as e:
            logger.error(f"[{self.camera_id}] Exception during VideoCapture initialization for '{self.source}': {e}", exc_info=False)
            self.cap = None
            self.is_connected = False
        finally:
            # Restore original environment variable if it existed, or remove if we set it.
            if original_ffmpeg_options is None:
                if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ: # Check if it was indeed set by us
                    del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
                    logger.debug(f"[{self.camera_id}] Cleared OPENCV_FFMPEG_CAPTURE_OPTIONS.")
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = original_ffmpeg_options
                logger.debug(f"[{self.camera_id}] Restored OPENCV_FFMPEG_CAPTURE_OPTIONS to original value: '{original_ffmpeg_options}'.")


    def read_frame(self):
        """
        Reads a frame from the video stream. Handles reconnection attempts with exponential backoff.
        """
        if not self.is_connected or self.cap is None:
            # Calculate delay: base_delay * (2^attempts), capped at a max (e.g., 60s)
            delay_seconds = self.reconnect_delay_base * (2 ** min(self.connect_attempts, 4)) 
            delay_seconds = min(delay_seconds, 60) # Cap delay

            if time.time() - self.last_attempt_time > delay_seconds:
                if self.connect_attempts < self.max_connect_attempts: # Only retry if under max attempts for this cycle
                    logger.warning(f"[{self.camera_id}] Camera not connected. Attempting to reconnect (next attempt in {delay_seconds:.0f}s)...")
                    self.connect()
                else: # Reached max attempts for this short cycle, will wait longer before next auto-attempt by this logic
                    logger.error(f"[{self.camera_id}] Camera not connected after {self.max_connect_attempts} rapid attempts. Will retry after a longer pause ({delay_seconds:.0f}s).")
                    # Reset attempts to allow retrying after a longer pause by this logic
                    self.last_attempt_time = time.time() # Update last attempt time to enforce the longer pause
            
            if not self.is_connected: # If still not connected after attempt
                return False, None

        ret, frame = self.cap.read()

        if not ret:
            logger.error(f"[{self.camera_id}] Error: cap.read() returned False. Can't receive frame. Stream might have ended or camera disconnected.")
            self.is_connected = False 
            if self.cap is not None: # Ensure cap exists before trying to release
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

# Example Usage (for testing this module directly):
if __name__ == '__main__':
    show_video_flag = False
    if CONFIG_MODULE_LOADED:
        show_video_flag = getattr(config, 'SHOW_VIDEO_WINDOW', False)
        if not hasattr(config, 'SHOW_VIDEO_WINDOW') and not CONFIG_MODULE_LOADED: # Log only if actual config was expected
             logger.warning("Attribute 'SHOW_VIDEO_WINDOW' not found in loaded config.py. Defaulting to False for display in this test.")
    else: 
        show_video_flag = config.SHOW_VIDEO_WINDOW 

    CAMERA_RTSP_URL_TEST = "rtsp://<credentials>:554/cam/realmonitor?channel=1&subtype=0"
    # CAMERA_RTSP_URL_TEST = 0 # For webcam testing

    logger.info(f"--- Testing CameraManager with source: {CAMERA_RTSP_URL_TEST} ---")
    logger.info(f"Display window for testing (SHOW_VIDEO_WINDOW): {show_video_flag}")
    
    build_info = cv2.getBuildInformation()
    if "FFMPEG" not in build_info or "YES" not in build_info.split("FFMPEG")[1].split("\n")[0]:
        logger.warning("OpenCV build does not seem to have FFMPEG support enabled. RTSP streams might not work as expected.")
    else:
        logger.info("OpenCV appears to have FFMPEG support, which is good for RTSP.")

    camera_instance_test = Camera(source=CAMERA_RTSP_URL_TEST, camera_id="test_cam_standalone")

    if camera_instance_test.is_connected:
        logger.info(f"Camera properties: {camera_instance_test.get_properties()}")
        
        frames_read_count = 0
        # --- MODIFICATION: Run indefinitely until 'q' is pressed or error ---
        # max_frames_for_test = 150 # Old limit
        logger.info("Starting continuous frame reading test. Press 'q' in the OpenCV window to quit (if shown).")

        while True: # Loop indefinitely
            ret_val, current_frame = camera_instance_test.read_frame()
            
            if not ret_val or current_frame is None:
                logger.warning(f"Failed to read frame (after {frames_read_count} successful frames), or connection lost during test.")
                if not camera_instance_test.is_connected:
                    logger.info("Connection seems lost. Test will wait briefly before next read_frame attempt (which triggers reconnect logic).")
                    time.sleep(camera_instance_test.reconnect_delay_base) 
                # If continuously failing, this loop might spin. Consider adding a max consecutive failure counter.
                # For now, it relies on the reconnect logic within read_frame.
                if not camera_instance_test.is_connected and camera_instance_test.connect_attempts >= camera_instance_test.max_connect_attempts:
                    logger.error("Max connection attempts reached and still not connected. Ending test.")
                    break # Exit if persistently failing to connect
                continue 

            frames_read_count += 1
            
            if frames_read_count % 100 == 0: # Log every 100 frames
                logger.info(f"Successfully read {frames_read_count} frames. Last frame shape: {current_frame.shape}")


            if show_video_flag: 
                # Add timestamp to the frame for display
                timestamp_display = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(current_frame, timestamp_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow(f'CCTV Test Feed - {camera_instance_test.camera_id}', current_frame)
                
                key_press = cv2.waitKey(1) # Use waitKey(1) for continuous playback
                if key_press & 0xFF == ord('q'): 
                    logger.info("User pressed 'q' to quit test display.")
                    break
            else: 
                if frames_read_count % ( (camera_instance_test.fps if camera_instance_test.fps > 0 else 10) * 5) == 0: # Log approx every 5 seconds
                    logger.info(f"Successfully processed {frames_read_count} frames (display window is off)...")
        
        logger.info(f"Finished testing loop. Total frames read: {frames_read_count}.")
    else:
        logger.error("Failed to connect to the camera initially during standalone test. Please check logs and camera RTSP URL/settings.")

    camera_instance_test.release() 
    if show_video_flag:
        cv2.destroyAllWindows() 
    logger.info("--- CameraManager Standalone Test Complete ---")
