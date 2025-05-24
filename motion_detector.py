# motion_detector.py
import cv2
import numpy as np
import logging
import os # Added for os.path.exists in test section

# --- Global Configuration Handling ---
# Initialize to defaults. These will be updated if config.py is successfully imported.
CONFIG_MODULE_LOADED = False
config_object = None # This will hold either the imported 'config' module or a FallbackConfig instance

# Define a FallbackConfig class that will be used if config.py cannot be imported or fails.
class FallbackConfigClass:
    # Define all attributes that MotionDetector class or its test block might need from config
    MOTION_HISTORY = 500
    MOTION_VAR_THRESHOLD = 50
    MOTION_DETECT_SHADOWS = True
    MOTION_MIN_CONTOUR_AREA = 500
    SHOW_VIDEO_WINDOW = False # Default for standalone testing, especially if X server is an issue
    # Add other relevant fallbacks if MotionDetector directly uses more config items
    # For example, if the test block in this file uses FPS_LIMIT from config:
    FPS_LIMIT = 15 

try:
    import config as actual_config_module
    config_object = actual_config_module # Use the successfully imported module
    CONFIG_MODULE_LOADED = True
    # Assuming main_controller.py or config.py itself sets up the primary logging.
    # If running standalone and config.py doesn't set up logging, logger might not be fully configured yet.
    logger = logging.getLogger(__name__) # Get logger, assuming it's configured elsewhere or by basicConfig later
    if not logger.hasHandlers(): # If no handlers are set up by this point (e.g. by config.py)
        # This basicConfig is for when motion_detector.py is run directly AND config.py doesn't set up root logging.
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
        logger.info("Config module loaded, but no handlers found for logger. Applied basicConfig for standalone run.")

except ImportError:
    # This block runs if 'import config' fails specifically with ImportError
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    logger = logging.getLogger("MotionDetector_Fallback_ImportError")
    config_object = FallbackConfigClass() # Use fallback
    CONFIG_MODULE_LOADED = False
    logger.warning("Could not import 'config.py'. Running MotionDetector in standalone mode with fallback configurations.")
except Exception as e:
    # Catches any other exception during the import or initial setup in the try block
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    logger = logging.getLogger("MotionDetector_Fallback_OtherError")
    logger.error(f"An unexpected error occurred trying to import or process 'config.py': {e}", exc_info=True)
    config_object = FallbackConfigClass() # Use fallback
    CONFIG_MODULE_LOADED = False
    logger.info("Using fallback configurations due to an error during 'config.py' processing.")


class MotionDetector:
    def __init__(self, 
                 history=config_object.MOTION_HISTORY, 
                 var_threshold=config_object.MOTION_VAR_THRESHOLD, 
                 detect_shadows=config_object.MOTION_DETECT_SHADOWS, 
                 min_contour_area=config_object.MOTION_MIN_CONTOUR_AREA,
                 gaussian_blur_kernel_size=(21, 21), 
                 dilation_iterations=2, 
                 erosion_iterations=1):
        """
        Initializes the MotionDetector using Background Subtraction.
        All parameters default to values from the 'config_object' (either imported config or fallback).
        """
        self.min_contour_area = min_contour_area
        self.gaussian_blur_kernel_size = gaussian_blur_kernel_size
        self.dilation_iterations = dilation_iterations
        self.erosion_iterations = erosion_iterations
        
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=history, 
            varThreshold=var_threshold, 
            detectShadows=detect_shadows
        )
        logger.info(f"MotionDetector initialized: history={history}, varThreshold={var_threshold}, detectShadows={detect_shadows}, minArea={min_contour_area}")

    def detect_motion(self, frame_bgr):
        """
        Detects motion in a given BGR frame.
        """
        if frame_bgr is None:
            logger.warning("Received None frame in detect_motion.")
            return [], None

        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, self.gaussian_blur_kernel_size, 0)
        fgmask_raw = self.fgbg.apply(blurred_frame)
        kernel_erode = np.ones((3, 3), np.uint8)
        fgmask_eroded = cv2.erode(fgmask_raw, kernel_erode, iterations=self.erosion_iterations)
        kernel_dilate = np.ones((7, 7), np.uint8) # Typically larger kernel for dilation
        fgmask_processed = cv2.dilate(fgmask_eroded, kernel_dilate, iterations=self.dilation_iterations)
        contours, _ = cv2.findContours(fgmask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected_regions = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            motion_detected_regions.append((x, y, w, h))
        
        if motion_detected_regions:
            logger.debug(f"Motion detected in {len(motion_detected_regions)} region(s).")
        
        return motion_detected_regions, fgmask_processed


# Example Usage (for testing this module directly):
if __name__ == '__main__':
    # This logger will use the configuration from the try-except block at the top.
    logger.info("--- Testing MotionDetector Module (Standalone) ---")

    # Use the 'config_object' which is now guaranteed to be defined.
    show_video_flag_md_test = getattr(config_object, 'SHOW_VIDEO_WINDOW', False) 
    logger.info(f"Display windows for testing (SHOW_VIDEO_WINDOW from config_object): {show_video_flag_md_test}")

    # --- CHOOSE YOUR VIDEO SOURCE FOR TESTING ---
    test_video_source_md = "rtsp://<credentials>:554/cam/realmonitor?channel=1&subtype=0"
    # test_video_source_md = "my_test_video_with_motion.mp4" # Replace with your video file path
    # test_video_source_md = 0 # Webcam (check WSL2 USB access)

    logger.info(f"Using video source for motion detection test: {test_video_source_md}")

    if isinstance(test_video_source_md, str) and not test_video_source_md.startswith("rtsp://") and not os.path.exists(test_video_source_md):
        logger.error(f"Test video file not found: '{test_video_source_md}'. Please provide a valid path or RTSP URL.")
        exit()

    cap_test_md = None
    original_ffmpeg_options_md_test = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") # Save original
    if isinstance(test_video_source_md, str) and test_video_source_md.startswith("rtsp://"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        logger.debug("Set OPENCV_FFMPEG_CAPTURE_OPTIONS to 'rtsp_transport;tcp' for RTSP test.")
        cap_test_md = cv2.VideoCapture(test_video_source_md, cv2.CAP_FFMPEG)
    else:
        cap_test_md = cv2.VideoCapture(test_video_source_md)
    
    # Restore/clear env var after VideoCapture call
    if isinstance(test_video_source_md, str) and test_video_source_md.startswith("rtsp://"):
        if original_ffmpeg_options_md_test is None:
            if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
                del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = original_ffmpeg_options_md_test

    if not cap_test_md or not cap_test_md.isOpened():
        logger.error(f"Error opening video source: {test_video_source_md}")
        exit()

    motion_detector_instance_test = MotionDetector(
        min_contour_area=getattr(config_object, 'MOTION_MIN_CONTOUR_AREA', 700),
        var_threshold=getattr(config_object, 'MOTION_VAR_THRESHOLD', 50)
    )
    
    logger.info("Starting motion detection test. Press 'q' in an OpenCV window to quit (if shown).")

    frame_counter_md_test = 0
    max_test_frames = getattr(config_object, 'TEST_MAX_FRAMES', 0) # Allow limiting frames via config for headless tests
                                                                    # 0 means run indefinitely or until 'q'

    while True:
        ret_val_md, test_frame_md = cap_test_md.read()
        if not ret_val_md or test_frame_md is None:
            logger.info("End of video or cannot read frame. Exiting motion detection test.")
            break
        
        frame_counter_md_test += 1
        
        motion_regions_found_md, processed_mask_md = motion_detector_instance_test.detect_motion(test_frame_md)

        if show_video_flag_md_test:
            display_frame_md_test = test_frame_md.copy()
            for (x, y, w, h) in motion_regions_found_md:
                cv2.rectangle(display_frame_md_test, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Motion Detection Test - Frame with Motion", display_frame_md_test)
            if processed_mask_md is not None:
                cv2.imshow("Motion Detection Test - Processed Foreground Mask", processed_mask_md)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): # Use waitKey(1) for smoother playback
                logger.info("User pressed 'q' to quit display.")
                break
        else:
            if frame_counter_md_test % 30 == 0: 
                 logger.info(f"Processed {frame_counter_md_test} frames. Motion regions in last: {len(motion_regions_found_md)} (Display off)")
            if max_test_frames > 0 and frame_counter_md_test >= max_test_frames:
                logger.info(f"Reached max_test_frames ({max_test_frames}). Ending headless test.")
                break

    cap_test_md.release()
    if show_video_flag_md_test:
        cv2.destroyAllWindows()
    logger.info(f"--- MotionDetector Module Test Complete (Processed {frame_counter_md_test} frames) ---")
