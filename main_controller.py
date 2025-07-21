# main_controller.py
import cv2
import time
import logging
import os
from datetime import datetime

# --- Configuration Import ---
try:
    import config as config
except ImportError:
    print("FATAL ERROR: config.py not found. Please ensure it's in the same directory.")
    exit()

# --- Module Imports ---
try:
    from camera_manager import Camera
    from motion_detector import MotionDetector
    from remote_object_detector import RemoteObjectDetector  # Use remote GPU detector
    from video_recorder import VideoRecorder
except ImportError as e:
    print(f"FATAL ERROR: Failed to import modules: {e}")
    exit()

# --- Logging Setup ---
try:
    log_level_from_config = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    if not os.path.exists(config.LOG_FILE_DIR):
        os.makedirs(config.LOG_FILE_DIR, exist_ok=True)
    logging.basicConfig(level=log_level_from_config,
                        format='%(asctime)s - %(name)s [%(levelname)s] %(module)s:%(lineno)d - %(message)s',
                        handlers=[logging.FileHandler(config.LOG_FILE), logging.StreamHandler()])
except Exception as e:
    print(f"Warning: Logging setup failed: {e}. Using console logging.")
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s',
                        handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary data directories."""
    try:
        os.makedirs(config.RECORDINGS_DIR, exist_ok=True)
        os.makedirs(config.LOG_FILE_DIR, exist_ok=True)
        os.makedirs(config.KNOWN_FACES_DIR, exist_ok=True)
        logger.info(f"Data directories created/verified in: {config.BASE_STORAGE_DIR}")
    except OSError as e:
        logger.error(f"Could not create data directories: {e}")
        exit(1)

class CCTVSystem:
    def __init__(self):
        """Initialize the CCTV system with remote GPU processing."""
        setup_directories()
        logger.info("Initializing CCTV System with Remote GPU Processing...")
        
        # Initialize components
        self.cameras = []
        self.motion_detectors = []
        self.video_recorders = []
        
        # Initialize remote GPU object detector
        if config.REMOTE_GPU_ENABLED:
            logger.info(f"Initializing Remote GPU Object Detector: {config.REMOTE_GPU_SERVER_URL}")
            self.object_detector = RemoteObjectDetector(
                remote_gpu_url=config.REMOTE_GPU_SERVER_URL,
                confidence_threshold=config.OBJECT_CONFIDENCE_THRESHOLD,
                fallback_to_cpu=config.FALLBACK_TO_CPU
            )
        else:
            logger.info("Remote GPU disabled, using local CPU detector")
            from object_detector import ObjectDetector
            self.object_detector = ObjectDetector()
        
        # Initialize cameras and related components
        self._initialize_cameras()
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        logger.info("CCTV System initialization complete")
    
    def _initialize_cameras(self):
        """Initialize cameras and related components."""
        for i, camera_source in enumerate(config.CAMERA_SOURCES):
            try:
                # Initialize camera
                camera = Camera(camera_source, camera_id=i)
                self.cameras.append(camera)
                
                # Initialize motion detector for this camera
                motion_detector = MotionDetector()
                self.motion_detectors.append(motion_detector)
                
                # Initialize video recorder for this camera
                video_recorder = VideoRecorder(
                    base_recordings_dir=config.RECORDINGS_DIR,
                    camera_id=i,
                    pre_event_seconds=config.RECORD_SECONDS_BEFORE_EVENT,
                    post_event_seconds=config.RECORD_SECONDS_AFTER_EVENT,
                    max_clip_duration_minutes=config.MAX_RECORDING_MINUTES
                )
                self.video_recorders.append(video_recorder)
                
                logger.info(f"Camera {i} initialized: {camera_source}")
                
            except Exception as e:
                logger.error(f"Failed to initialize camera {i} ({camera_source}): {e}")
    
    def process_frame(self, camera_id, frame):
        """Process a single frame from a camera."""
        detections = []
        motion_detected = False
        faces_detected = []
        
        try:
            # Motion detection (always runs locally - lightweight)
            motion_detected = self.motion_detectors[camera_id].detect_motion(frame)
            
            # AI processing (every Nth frame for performance)
            if self.frame_count % config.AI_PROCESSING_FRAME_INTERVAL == 0:
                # Object detection (remote GPU or local CPU)
                start_time = time.time()
                detections = self.object_detector.detect_objects(frame)
                detection_time = time.time() - start_time
                
                if detections:
                    logger.info(f"Camera {camera_id}: Detected {len(detections)} objects in {detection_time:.3f}s")
                # Face recognition logic is currently disabled
            
            # Event triggering
            event_triggered = (motion_detected or 
                             len(detections) > 0)
            
            if event_triggered:
                # Start/continue recording
                self.video_recorders[camera_id].start_or_update_recording("event")
                
                # Log event
                event_details = {
                    'motion': motion_detected,
                    'objects': len(detections),
                }
                logger.info(f"Camera {camera_id}: Event triggered - {event_details}")
            else:
                # Add frame to buffer (for pre-event recording)
                self.video_recorders[camera_id].add_frame_to_buffer(frame)
            
            # Display frame (if enabled)
            if config.SHOW_VIDEO_WINDOW:
                self._draw_detections(frame, detections, faces_detected, motion_detected)
                cv2.imshow(f"Camera {camera_id}", frame)
        
        except Exception as e:
            logger.error(f"Error processing frame from camera {camera_id}: {e}")
    
    def _draw_detections(self, frame, detections, faces, motion_detected):
        """Draw detection results on frame for display."""
        # Draw object detections
        for detection in detections:
            bbox = detection.get('bbox', [0, 0, 0, 0])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{detection['class_name']}: {detection['confidence']:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # Draw face detections (currently not used)
        # Draw motion indicator
        if motion_detected:
            cv2.putText(frame, "MOTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    def run(self):
        """Main processing loop."""
        logger.info("Starting CCTV System main loop...")
        
        try:
            while True:
                for camera_id, camera in enumerate(self.cameras):
                    try:
                        ret, frame = camera.read_frame()
                        if ret and frame is not None:
                            self.process_frame(camera_id, frame)
                        else:
                            logger.warning(f"No frame received from camera {camera_id}")
                    except Exception as e:
                        logger.error(f"Error with camera {camera_id}: {e}")
                # FPS limiting
                if config.FPS_LIMIT:
                    time.sleep(1.0 / config.FPS_LIMIT)
                # Performance monitoring
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 30:  # Log FPS every 30 seconds
                    fps = self.frame_count / (current_time - self.last_fps_time)
                    logger.info(f"Processing FPS: {fps:.2f}")
                    self.frame_count = 0
                    self.last_fps_time = current_time
                # Check for quit command
                if config.SHOW_VIDEO_WINDOW and cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up CCTV System...")
        # Stop video recorders
        for recorder in self.video_recorders:
            try:
                recorder.stop_recording_if_idle()
            except:
                pass
        # Release cameras
        for camera in self.cameras:
            try:
                camera.release()
            except:
                pass
        # Close windows
        if config.SHOW_VIDEO_WINDOW:
            cv2.destroyAllWindows()
        logger.info("CCTV System cleanup complete")

if __name__ == "__main__":
    system = CCTVSystem()
    system.run()
