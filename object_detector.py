# object_detector.py
from ultralytics import YOLO
import cv2
import torch
import logging
import os
import numpy as np
import time

# --- Global Configuration Handling ---
CONFIG_MODULE_LOADED = False
config_object = None

class FallbackConfigClass:
    OBJECT_DETECTION_MODEL_PATH = 'yolov8s.pt'
    OBJECT_CONFIDENCE_THRESHOLD = 0.45
    SHOW_VIDEO_WINDOW = False
    TEST_MAX_FRAMES_OBJECT_DETECTION = 0

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
        Returns a list of detected objects with their bounding boxes and confidence scores.
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
        if detected_objects_list:
            log_objects_info = []
            for obj in detected_objects_list:
                log_objects_info.append((obj['class_name'], f"{obj['confidence']:.2f}"))
            logger.debug(f"Detected objects: {log_objects_info}")
        return detected_objects_list
