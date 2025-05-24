# main_controller.py
import cv2
import time
import logging
from datetime import datetime # Not directly used in this version, but good for future
import os # For path checks

# --- Configuration Import ---
# This will load all settings from config.py
try:
    import config
except ImportError:
    print("FATAL ERROR: config.py not found. Please ensure it's in the same directory or PYTHONPATH.")
    exit()
except AttributeError as e:
    print(f"FATAL ERROR: Attribute missing in config.py: {e}. Please check your config.py file.")
    exit()


# --- Module Imports ---
# These will use the config loaded above when they are imported and initialized
try:
    from camera_manager import Camera
    from motion_detector import MotionDetector
    from object_detector import ObjectDetector
    from face_recognizer import FaceRecognizer
    from video_recorder import VideoRecorder
except ImportError as e:
    print(f"FATAL ERROR: Failed to import one or more AI modules (Camera, MotionDetector, etc.): {e}")
    print("Ensure all .py files (camera_manager.py, motion_detector.py, etc.) are in the same directory or PYTHONPATH.")
    exit()

# --- Logging Setup ---
# Configure logging based on settings in config.py
# Ensure LOG_FILE_DIR from config.py is created or writable
try:
    log_level_from_config = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    # Create log directory if it doesn't exist
    if hasattr(config, 'LOG_FILE_DIR') and not os.path.exists(config.LOG_FILE_DIR):
        try:
            os.makedirs(config.LOG_FILE_DIR, exist_ok=True)
        except OSError as e:
            print(f"Warning: Could not create log directory {config.LOG_FILE_DIR}: {e}. Logging to console only.")
            logging.basicConfig(level=log_level_from_config,
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                handlers=[logging.StreamHandler()]) # Fallback to console
    else:
        logging.basicConfig(level=log_level_from_config,
                            format='%(asctime)s - %(name)s [%(levelname)s] %(module)s:%(lineno)d - %(message)s',
                            handlers=[logging.FileHandler(config.LOG_FILE), logging.StreamHandler()])
except Exception as e:
    print(f"Error setting up logging from config: {e}. Using basic console logging.")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__) # Get logger for this module

# --- Global Dictionaries for Managing Modules per Camera ---
cameras_dict = {}
motion_detectors_dict = {}
# Object detector and Face recognizer can be shared if models/settings are same for all cameras
# Or they can be per-camera if different models are needed.
# For simplicity and typical resource constraints (one GPU), we'll use shared instances here.
shared_object_detector = None
shared_face_recognizer = None
video_recorders_dict = {}

# --- Per-Camera State Tracking ---
last_motion_trigger_time = {} # cam_id: timestamp of last significant motion that triggered AI
last_significant_event_log_time = {} # cam_id: timestamp to control logging frequency for similar events
# last_frame_processed_ai_time = {} # cam_id: timestamp for AI_PROCESSING_FRAME_INTERVAL logic

def initialize_modules():
    """
    Initializes all AI modules and camera connections based on config.py.
    Returns True if successful, False otherwise.
    """
    global shared_object_detector, shared_face_recognizer
    logger.info("Initializing AI modules and cameras...")

    # --- Initialize Shared AI Modules ---
    try:
        logger.info(f"Loading Object Detector model: {config.OBJECT_DETECTION_MODEL_PATH}")
        shared_object_detector = ObjectDetector(
            model_path=config.OBJECT_DETECTION_MODEL_PATH,
            confidence_threshold=config.OBJECT_CONFIDENCE_THRESHOLD
        )
        if not shared_object_detector.model: # Check if model loaded successfully
            logger.error("FATAL: Shared Object Detector model failed to load. Exiting.")
            return False
    except Exception as e:
        logger.error(f"FATAL: Exception initializing Shared Object Detector: {e}", exc_info=True)
        return False

    try:
        logger.info(f"Loading Face Recognizer. Known faces from: {config.KNOWN_FACES_DIR}")
        shared_face_recognizer = FaceRecognizer(
            known_faces_dir=config.KNOWN_FACES_DIR,
            encodings_path=config.KNOWN_ENCODINGS_PATH,
            detection_model=config.FACE_DETECTION_MODEL,
            tolerance=config.FACE_RECOGNITION_TOLERANCE
        )
        # FaceRecognizer logs internally if no faces are enrolled.
    except Exception as e:
        logger.error(f"FATAL: Exception initializing Shared Face Recognizer: {e}", exc_info=True)
        return False

    # --- Initialize Cameras and Per-Camera Modules ---
    if not config.CAMERA_SOURCES:
        logger.error("FATAL: No camera sources defined in config.CAMERA_SOURCES. Exiting.")
        return False

    for i, cam_source_uri in enumerate(config.CAMERA_SOURCES):
        cam_id = f"cam_{i}_{str(cam_source_uri).split('@')[-1].replace('/','_').replace(':','-')[:30]}" # Create a somewhat unique ID
        logger.info(f"Initializing modules for camera ID: {cam_id} (Source: {cam_source_uri})")

        camera_instance = Camera(source=cam_source_uri, camera_id=cam_id)
        if not camera_instance.is_connected:
            logger.warning(f"Failed to connect to camera {cam_id} during initialization. Will keep trying.")
            # Still add it, read_frame will attempt reconnections.
        
        cameras_dict[cam_id] = camera_instance
        motion_detectors_dict[cam_id] = MotionDetector(
            history=config.MOTION_HISTORY,
            var_threshold=config.MOTION_VAR_THRESHOLD,
            detect_shadows=config.MOTION_DETECT_SHADOWS,
            min_contour_area=config.MOTION_MIN_CONTOUR_AREA
        )
        video_recorders_dict[cam_id] = VideoRecorder(
            camera_id=cam_id, # Sanitized ID is handled within VideoRecorder
            base_recordings_dir=config.RECORDINGS_DIR,
            pre_event_seconds=config.RECORD_SECONDS_BEFORE_EVENT,
            post_event_seconds=config.RECORD_SECONDS_AFTER_EVENT,
            max_clip_duration_minutes=config.MAX_RECORDING_MINUTES,
            target_fps= (config.FPS_LIMIT if config.FPS_LIMIT is not None and config.FPS_LIMIT > 0 else 15) # Pass a valid FPS
        )
        
        # Initialize state tracking for this camera
        last_motion_trigger_time[cam_id] = 0 # Allow immediate trigger if motion on first frame
        last_significant_event_log_time[cam_id] = 0
        # last_frame_processed_ai_time[cam_id] = 0

    logger.info(f"Initialization complete. {len(cameras_dict)} camera(s) configured.")
    return True


def process_camera_frame(cam_id, camera_instance, frame_bgr, frame_process_counter):
    """
    Processes a single frame from a camera: motion, object, face detection, and recording.
    """
    current_time = time.time()
    event_description_for_this_frame = None # What kind of event happened in this frame for recording
    run_heavy_ai_this_frame = False # Flag to run object/face detection

    # --- 1. Add frame to recorder's buffer (always) ---
    if cam_id in video_recorders_dict:
        video_recorders_dict[cam_id].add_frame(frame_bgr)

    # --- 2. Motion Detection (always, as it's lightweight) ---
    motion_regions, motion_mask_for_display = [], None # Initialize
    if cam_id in motion_detectors_dict:
        motion_regions, motion_mask_for_display = motion_detectors_dict[cam_id].detect_motion(frame_bgr)

    if motion_regions:
        # If significant motion, and cooldown has passed, flag for AI processing
        if (current_time - last_motion_trigger_time.get(cam_id, 0)) > config.MOTION_COOLDOWN_SECONDS:
            logger.info(f"[{cam_id}] Significant motion detected ({len(motion_regions)} regions). Triggering AI processing.")
            last_motion_trigger_time[cam_id] = current_time
            run_heavy_ai_this_frame = True
            event_description_for_this_frame = "motion_detected" # Basic event
    
    # --- 3. Scheduled AI Processing (if no motion trigger, run by interval) ---
    if not run_heavy_ai_this_frame and (frame_process_counter % config.AI_PROCESSING_FRAME_INTERVAL == 0):
        logger.debug(f"[{cam_id}] Scheduled AI processing interval reached.")
        run_heavy_ai_this_frame = True
        # No specific event_description here unless objects/faces are found

    # --- 4. Heavy AI: Object Detection & Face Recognition (if flagged) ---
    detected_objects_info = [] # Store info about objects detected by AI
    recognized_faces_info = [] # Store info about faces recognized by AI

    if run_heavy_ai_this_frame and shared_object_detector:
        frame_for_ai = frame_bgr.copy() # Process a copy to avoid drawing on original before all ops
        
        # --- Object Detection ---
        detected_objects_list = shared_object_detector.detect_objects(frame_for_ai)
        
        persons_detected_this_frame = []
        other_focused_objects_this_frame = []

        for obj in detected_objects_list:
            detected_objects_info.append(obj) # For drawing later
            if obj["class_name"] == "person" and obj["confidence"] >= config.OBJECT_CONFIDENCE_THRESHOLD:
                persons_detected_this_frame.append(obj)
            elif config.FOCUSED_OBJECT_CLASSES and \
                 obj["class_name"] in config.FOCUSED_OBJECT_CLASSES and \
                 obj["confidence"] >= config.OBJECT_CONFIDENCE_THRESHOLD:
                other_focused_objects_this_frame.append(obj)

        if persons_detected_this_frame:
            person_names_str = ", ".join(sorted(list(set([p['class_name'] for p in persons_detected_this_frame]))))
            log_msg = f"[{cam_id}] PERSON(S) DETECTED: {len(persons_detected_this_frame)} of type(s) '{person_names_str}'."
            if (current_time - last_significant_event_log_time.get(cam_id, 0)) > config.EVENT_COOLDOWN_SECONDS:
                logger.info(log_msg)
                last_significant_event_log_time[cam_id] = current_time
            event_description_for_this_frame = f"person_{len(persons_detected_this_frame)}" # Overwrite basic motion

            # --- Face Recognition (only if persons are detected and it's enabled for 'person') ---
            if "person" in config.PROCESS_FACES_FOR_CLASSES and shared_face_recognizer:
                # You could pass cropped person regions to face_recognizer for efficiency.
                # For now, passing the whole frame_for_ai if a person is present.
                recognized_faces_list = shared_face_recognizer.recognize_faces(frame_for_ai)
                
                known_persons_recognized = []
                for face_res in recognized_faces_list:
                    recognized_faces_info.append(face_res) # For drawing
                    if face_res["name"] != "Unknown":
                        known_persons_recognized.append(face_res["name"])
                
                if known_persons_recognized:
                    known_names_str = ", ".join(sorted(list(set(known_persons_recognized))))
                    face_log_msg = f"[{cam_id}] KNOWN FACE(S) RECOGNIZED: {known_names_str}."
                    if (current_time - last_significant_event_log_time.get(cam_id, 0)) > config.EVENT_COOLDOWN_SECONDS:
                        logger.info(face_log_msg) # Log separately or append
                        last_significant_event_log_time[cam_id] = current_time # Reset cooldown
                    event_description_for_this_frame = f"known_person_{known_names_str.replace(' ','_')[:20]}" # Prioritize known person
                elif recognized_faces_list: # Some faces detected, but all unknown
                    face_log_msg = f"[{cam_id}] UNKNOWN FACE(S) DETECTED: {len(recognized_faces_list)}."
                    if (current_time - last_significant_event_log_time.get(cam_id, 0)) > config.EVENT_COOLDOWN_SECONDS:
                        logger.info(face_log_msg)
                        last_significant_event_log_time[cam_id] = current_time
                    if not event_description_for_this_frame or "person" in event_description_for_this_frame : # Don't overwrite known_person
                         event_description_for_this_frame = f"unknown_person_{len(recognized_faces_list)}"


        elif other_focused_objects_this_frame: # No persons, but other focused objects detected
            obj_names_str = ", ".join(sorted(list(set([o['class_name'] for o in other_focused_objects_this_frame]))))
            log_msg = f"[{cam_id}] FOCUSED OBJECT(S) DETECTED: {obj_names_str}."
            if (current_time - last_significant_event_log_time.get(cam_id, 0)) > config.EVENT_COOLDOWN_SECONDS:
                logger.info(log_msg)
                last_significant_event_log_time[cam_id] = current_time
            # Only set event_description if not already set by person or more critical motion
            if event_description_for_this_frame is None or event_description_for_this_frame == "motion_detected":
                event_description_for_this_frame = f"object_{obj_names_str.replace(' ','_')[:20]}"

    # --- 5. Trigger Recording (if a significant event was identified) ---
    if event_description_for_this_frame and cam_id in video_recorders_dict:
        logger.debug(f"[{cam_id}] Event detected: '{event_description_for_this_frame}'. Triggering/updating recording.")
        video_recorders_dict[cam_id].start_or_update_recording(event_description_for_this_frame)

    # --- 6. Display Frame (Optional, with overlays) ---
    if config.SHOW_VIDEO_WINDOW:
        display_frame = frame_bgr.copy() # Work on a copy for display
        
        # Draw motion regions (light green)
        for (x, y, w, h) in motion_regions:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (152, 251, 152), 1)

        # Draw object detection boxes (if AI ran and objects were found)
        for obj in detected_objects_info: # This list is populated if run_heavy_ai_this_frame was True
            x1, y1, x2, y2 = obj["bbox"]
            label = f'{obj["class_name"]}: {obj["confidence"]:.2f}'
            color = (0, 165, 255) if obj["class_name"] == "person" else (255, 0, 0) # Orange for person, Red for others
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw face recognition boxes (if AI ran and faces were found)
        for face in recognized_faces_info: # This list is populated if run_heavy_ai_this_frame was True and persons found
            l, t, r, b = face["bbox"]
            name = face["name"]
            conf = face["confidence"]
            face_color = (0, 255, 255) if name != "Unknown" else (0, 0, 255) # Yellow for known, Red for unknown
            cv2.rectangle(display_frame, (l, t), (r, b), face_color, 2)
            label = f"{name}" + (f" ({conf*100:.0f}%)" if conf is not None and name != "Unknown" else "")
            cv2.putText(display_frame, label, (l + 6, b - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Add timestamp and cam_id to display
        timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        cv2.putText(display_frame, f"{cam_id} - {timestamp_text}", (10, display_frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(f"CCTV AI - {cam_id}", display_frame)
        # Optionally display motion mask for debugging:
        # if motion_mask_for_display is not None:
        #    cv2.imshow(f"Motion Mask - {cam_id}", motion_mask_for_display)


def main_loop():
    """
    Main processing loop for all cameras.
    """
    if not initialize_modules():
        logger.error("Application initialization failed. Please check logs. Exiting.")
        return

    # Counter for AI_PROCESSING_FRAME_INTERVAL, per camera or global
    # Using a global counter here for simplicity, assuming FPS_LIMIT is applied per frame cycle.
    # If cameras have very different native FPS, a per-camera counter might be better.
    global_frame_processing_counter = 0 

    while True:
        start_of_frame_cycle_time = time.time()
        active_cameras_this_cycle = 0

        for cam_id, camera_instance in cameras_dict.items():
            if not camera_instance.is_connected: # Try to connect if not already
                camera_instance.read_frame() # This will trigger connect() if needed
                if not camera_instance.is_connected:
                    logger.debug(f"[{cam_id}] Still not connected, skipping this cycle.")
                    continue # Skip this camera for this cycle
            
            ret, frame_bgr = camera_instance.read_frame()
            if not ret or frame_bgr is None:
                logger.warning(f"[{cam_id}] Failed to get frame. Will retry.")
                continue # Skip this frame, will try again

            active_cameras_this_cycle += 1
            
            # Resize frame if dimensions are specified in config
            if config.FRAME_WIDTH and config.FRAME_HEIGHT:
                frame_bgr = cv2.resize(frame_bgr, (config.FRAME_WIDTH, config.FRAME_HEIGHT), interpolation=cv2.INTER_AREA)

            process_camera_frame(cam_id, camera_instance, frame_bgr, global_frame_processing_counter)

        global_frame_processing_counter += 1

        # --- FPS Control & GUI Exit ---
        if config.SHOW_VIDEO_WINDOW:
            # Allow GUI events to be processed. waitKey(1) is typical for video loops.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Quit signal ('q') received from OpenCV window. Shutting down...")
                break
        
        # Frame rate limiting for the entire cycle
        # This ensures the loop doesn't run faster than FPS_LIMIT if set
        if config.FPS_LIMIT and config.FPS_LIMIT > 0:
            elapsed_time_for_cycle = time.time() - start_of_frame_cycle_time
            sleep_time = (1.0 / config.FPS_LIMIT) - elapsed_time_for_cycle
            if sleep_time > 0:
                time.sleep(sleep_time)
        elif active_cameras_this_cycle == 0 : # No active cameras, sleep a bit to avoid busy loop
             time.sleep(0.5)


    # --- Cleanup on Exit ---
    logger.info("Initiating shutdown sequence...")
    for cam_id in cameras_dict:
        if cameras_dict[cam_id]:
            cameras_dict[cam_id].release()
        if cam_id in video_recorders_dict and video_recorders_dict[cam_id]:
            video_recorders_dict[cam_id].force_stop_and_join() # Ensure recordings are finalized
    
    if config.SHOW_VIDEO_WINDOW:
        cv2.destroyAllWindows()
    logger.info("Application shut down successfully.")


if __name__ == "__main__":
    logger.info("Starting Home CCTV AI Application...")
    try:
        main_loop()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"An unhandled exception occurred in main_loop: {e}", exc_info=True)
    finally:
        # Ensure cleanup runs even if main_loop crashes before normal exit
        logger.info("Performing final cleanup...")
        for cam_id_cleanup in cameras_dict: # Use a different var name to avoid scope issues if main_loop didn't run
            if cam_id_cleanup in cameras_dict and cameras_dict[cam_id_cleanup]:
                cameras_dict[cam_id_cleanup].release()
            if cam_id_cleanup in video_recorders_dict and video_recorders_dict[cam_id_cleanup]:
                 if video_recorders_dict[cam_id_cleanup].is_recording: # Check if it might be recording
                    video_recorders_dict[cam_id_cleanup].force_stop_and_join()
        
        if hasattr(config, 'SHOW_VIDEO_WINDOW') and config.SHOW_VIDEO_WINDOW: # Check attribute exists before using
            cv2.destroyAllWindows()
        logger.info("Application final cleanup complete.")

