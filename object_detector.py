# object_detector.py
from ultralytics import YOLO
import cv2
import torch # For checking GPU availability and setting device
import logging
import os
import numpy as np # Added for dummy image in _load_model
import time # Added for standalone test timing

# --- Global Configuration Handling ---
CONFIG_MODULE_LOADED = False
config_object = None 

class FallbackConfigClass:
    OBJECT_DETECTION_MODEL_PATH = 'yolov8s.pt'
    OBJECT_CONFIDENCE_THRESHOLD = 0.45
    SHOW_VIDEO_WINDOW = False
    TEST_MAX_FRAMES_OBJECT_DETECTION = 0 # For headless testing limit
    # Add any other config attributes ObjectDetector class or its test block might need

try:
    import config as actual_config_module
    config_object = actual_config_module
    CONFIG_MODULE_LOADED = True
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
        logger.info("Config module loaded, but no handlers found for logger. Applied basicConfig for standalone run.")
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    logger = logging.getLogger("ObjectDetector_Fallback_ImportError")
    config_object = FallbackConfigClass()
    CONFIG_MODULE_LOADED = False
    logger.warning("Could not import 'config.py'. Running ObjectDetector in standalone mode with fallback configurations.")
except Exception as e:
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    logger = logging.getLogger("ObjectDetector_Fallback_OtherError")
    logger.error(f"An unexpected error occurred trying to import or process 'config.py': {e}", exc_info=True)
    config_object = FallbackConfigClass()
    CONFIG_MODULE_LOADED = False
    logger.info("Using fallback configurations due to an error during 'config.py' processing.")


class ObjectDetector:
    def __init__(self, 
                 model_path=config_object.OBJECT_DETECTION_MODEL_PATH, 
                 confidence_threshold=config_object.OBJECT_CONFIDENCE_THRESHOLD):
        """
        Initializes the ObjectDetector using a YOLO model from Ultralytics.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = None

        self._load_model()

    def _load_model(self):
        """
        Loads the YOLO model and moves it to the appropriate device (GPU if available, else CPU).
        """
        try:
            if torch.cuda.is_available():
                self.device = 'cuda'
                logger.info("CUDA (GPU) is available. Object detection will run on GPU.")
            else:
                self.device = 'cpu'
                logger.info("CUDA (GPU) not available. Object detection will run on CPU.")

            if not os.path.isabs(self.model_path) and not os.path.exists(self.model_path) and self.model_path.endswith('.pt'):
                 logger.info(f"Model path '{self.model_path}' is not absolute and doesn't exist locally. Assuming Ultralytics will handle it (e.g., download if standard name).")
            elif not os.path.exists(self.model_path):
                 logger.warning(f"Model file not found at specified path: '{self.model_path}'. Ultralytics might try to download if it's a standard model name.")

            self.model = YOLO(self.model_path) 
            self.model.to(self.device) 
            
            dummy_image = np.zeros((240, 320, 3), dtype=np.uint8)
            self.model(dummy_image, verbose=False) 
            
            logger.info(f"YOLO model '{self.model_path}' loaded successfully on device '{self.device}'.")
            if hasattr(self.model, 'names'):
                 logger.info(f"Object classes (first 10): {list(self.model.names.values())[:10]}...")

        except Exception as e:
            logger.error(f"Error loading YOLO model from '{self.model_path}': {e}", exc_info=True)
            self.model = None

    def detect_objects(self, frame_bgr):
        """
        Detects objects in a given BGR frame.
        """
        if self.model is None:
            logger.error("Object detection model not loaded. Cannot detect objects.")
            return [] 

        if frame_bgr is None:
            logger.warning("Received None frame in detect_objects.")
            return []

        results = self.model(frame_bgr, stream=False, verbose=False, conf=self.confidence_threshold) 
        
        detected_objects_list = []

        for result in results: 
            boxes = result.boxes  
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0]) 
                class_id = int(box.cls[0])      
                class_name = self.model.names[class_id] 

                detected_objects_list.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2)
                })
        
        # --- CORRECTED LINE BELOW ---
        if detected_objects_list:
            # Create a more readable list of tuples or strings for logging
            log_objects_info = []
            for obj in detected_objects_list:
                log_objects_info.append((obj['class_name'], f"{obj['confidence']:.2f}"))
            logger.debug(f"Detected objects: {log_objects_info}")
            
        return detected_objects_list


# Example Usage (for testing this module directly):
if __name__ == '__main__':
    logger.info("--- Testing ObjectDetector Module (Standalone) ---")

    show_video_flag_od_test = getattr(config_object, 'SHOW_VIDEO_WINDOW', False)
    logger.info(f"Display windows for testing (SHOW_VIDEO_WINDOW from config_object): {show_video_flag_od_test}")

    test_video_source_od = "rtsp://<credentials>:554/cam/realmonitor?channel=1&subtype=0"
    # test_video_source_od = "my_test_video_with_objects.mp4" 
    # test_video_source_od = 0 
    
    logger.info(f"Using video source for object detection test: {test_video_source_od}")

    if isinstance(test_video_source_od, str) and not test_video_source_od.startswith("rtsp://") and not os.path.exists(test_video_source_od):
        logger.error(f"Test video file not found: '{test_video_source_od}'. Please provide a valid path or RTSP URL.")
        exit()

    model_file_path_od_test = getattr(config_object, 'OBJECT_DETECTION_MODEL_PATH', 'yolov8s.pt')
    logger.info(f"Using model for object detection test: {model_file_path_od_test}")
    if not os.path.exists(model_file_path_od_test) and model_file_path_od_test.endswith('.pt'):
        logger.warning(f"Model file '{model_file_path_od_test}' not found locally. YOLO will attempt to download it if it's a standard model name.")

    object_detector_instance_test = ObjectDetector(
        model_path=model_file_path_od_test,
        confidence_threshold=getattr(config_object, 'OBJECT_CONFIDENCE_THRESHOLD', 0.45)
    )

    if object_detector_instance_test.model is None:
        logger.error("Failed to initialize ObjectDetector. Model might not have loaded. Exiting test.")
        exit()

    cap_test_od = None
    original_ffmpeg_options_od_test = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS") 
    if isinstance(test_video_source_od, str) and test_video_source_od.startswith("rtsp://"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        logger.debug("Set OPENCV_FFMPEG_CAPTURE_OPTIONS to 'rtsp_transport;tcp' for RTSP test.")
        cap_test_od = cv2.VideoCapture(test_video_source_od, cv2.CAP_FFMPEG)
    else:
        cap_test_od = cv2.VideoCapture(test_video_source_od)
    
    if isinstance(test_video_source_od, str) and test_video_source_od.startswith("rtsp://"):
        if original_ffmpeg_options_od_test is None:
            if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
                del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = original_ffmpeg_options_od_test

    if not cap_test_od or not cap_test_od.isOpened():
        logger.error(f"Error opening video source for object detection test: {test_video_source_od}")
        exit()
    
    logger.info("Starting object detection test. Press 'q' in an OpenCV window to quit (if shown).")

    frame_counter_od_test = 0
    max_test_frames_od = getattr(config_object, 'TEST_MAX_FRAMES_OBJECT_DETECTION', 0) 

    while True:
        ret_val_od, test_frame_od = cap_test_od.read()
        if not ret_val_od or test_frame_od is None:
            logger.info("End of video or cannot read frame. Exiting object detection test.")
            break
        
        frame_counter_od_test += 1
        
        detection_start_time = time.time()
        detected_objects_list_od = object_detector_instance_test.detect_objects(test_frame_od)
        detection_duration = time.time() - detection_start_time
        
        if frame_counter_od_test % 10 == 0: 
            logger.info(f"Frame {frame_counter_od_test}: Object detection took {detection_duration:.3f}s. Found {len(detected_objects_list_od)} objects.")

        if show_video_flag_od_test:
            display_frame_od_test = test_frame_od.copy()
            for obj_info in detected_objects_list_od:
                x1, y1, x2, y2 = obj_info['bbox']
                label = f"{obj_info['class_name']}: {obj_info['confidence']:.2f}"
                color = (0, 165, 255) if obj_info["class_name"] == "person" else (255, 0, 0)
                cv2.rectangle(display_frame_od_test, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame_od_test, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Object Detection Test", display_frame_od_test)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User pressed 'q' to quit display.")
                break
        else:
            if frame_counter_od_test % 30 == 0: 
                 logger.info(f"Processed {frame_counter_od_test} frames for object detection. Objects in last: {len(detected_objects_list_od)} (Display off)")
            if max_test_frames_od > 0 and frame_counter_od_test >= max_test_frames_od:
                logger.info(f"Reached max_test_frames_od ({max_test_frames_od}). Ending headless test.")
                break
                
    cap_test_od.release()
    if show_video_flag_od_test:
        cv2.destroyAllWindows()
    logger.info(f"--- ObjectDetector Module Test Complete (Processed {frame_counter_od_test} frames) ---")
