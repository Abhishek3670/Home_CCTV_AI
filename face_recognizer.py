# face_recognizer.py
import face_recognition
import cv2
import os
import numpy as np
import pickle # For saving and loading encodings
import logging
import time # For performance timing if needed

# --- Global Configuration Handling ---
CONFIG_MODULE_LOADED = False
config_object = None 

class FallbackConfigClass:
    KNOWN_FACES_DIR = "known_faces_test_standalone" 
    KNOWN_ENCODINGS_PATH = "known_face_encodings_standalone_test.pkl"
    FACE_DETECTION_MODEL = "hog" 
    FACE_RECOGNITION_TOLERANCE = 0.55
    SHOW_VIDEO_WINDOW = False 
    TEST_MAX_FRAMES_FACE_RECOGNITION = 0 # For headless testing limit
    # Add any other config attributes FaceRecognizer class or its test block might need

try:
    import config as actual_config_module
    config_object = actual_config_module
    CONFIG_MODULE_LOADED = True
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        # This basicConfig is for when face_recognizer.py is run directly
        # AND config.py doesn't set up root logging, or logging setup in config.py failed.
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s [%(levelname)s] %(module)s - %(message)s')
        logger.info("Config module loaded, but no handlers found for logger. Applied basicConfig for standalone run.")

except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s [%(levelname)s] %(module)s - %(message)s')
    logger = logging.getLogger("FaceRecognizer_Fallback_ImportError")
    config_object = FallbackConfigClass()
    CONFIG_MODULE_LOADED = False
    logger.warning("Could not import 'config.py'. Running FaceRecognizer in standalone mode with fallback configurations.")
except Exception as e:
    # Catches any other exception during the import or initial setup in the try block
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s [%(levelname)s] %(module)s - %(message)s')
    logger = logging.getLogger("FaceRecognizer_Fallback_OtherError")
    config_object = FallbackConfigClass() # Ensure config_object is set in this path too
    CONFIG_MODULE_LOADED = False
    logger.error(f"An unexpected error occurred trying to import or process 'config.py': {e}", exc_info=True)
    logger.info("Using fallback configurations due to an error during 'config.py' processing.")


class FaceRecognizer:
    def __init__(self,
                 known_faces_dir=config_object.KNOWN_FACES_DIR,
                 encodings_path=config_object.KNOWN_ENCODINGS_PATH,
                 detection_model=config_object.FACE_DETECTION_MODEL,
                 tolerance=config_object.FACE_RECOGNITION_TOLERANCE):
        """
        Initializes the FaceRecognizer.
        All parameters default to values from the 'config_object'.
        """
        self.known_faces_dir = known_faces_dir
        self.encodings_path = encodings_path
        self.detection_model = detection_model
        self.tolerance = tolerance
        self.known_face_encodings = []
        self.known_face_names = []

        logger.info(f"FaceRecognizer initialized. Known faces dir: '{self.known_faces_dir}', Encodings path: '{self.encodings_path}'")
        logger.info(f"Using face detection model: '{self.detection_model}', Tolerance: {self.tolerance}")

        if not os.path.exists(self.known_faces_dir):
            try:
                os.makedirs(self.known_faces_dir, exist_ok=True)
                logger.info(f"Created known_faces directory at: {self.known_faces_dir}")
            except OSError as e:
                logger.error(f"Error creating known_faces directory {self.known_faces_dir}: {e}. Please check permissions and path.")

        self.load_or_generate_known_encodings()

    def load_or_generate_known_encodings(self):
        """
        Loads known face encodings from a pickle file if it exists.
        Otherwise, it generates them from images in the known_faces_dir and saves them.
        """
        if os.path.exists(self.encodings_path):
            logger.info(f"Attempting to load known face encodings from {self.encodings_path}")
            try:
                with open(self.encodings_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                logger.info(f"Successfully loaded {len(self.known_face_names)} known person(s) with {len(self.known_face_encodings)} encodings.")
                if len(self.known_face_names) == 0:
                    logger.warning("Encodings file loaded, but it contained no known faces. Consider re-enrolling or checking the file.")
                return 
            except Exception as e:
                logger.error(f"Error loading encodings file '{self.encodings_path}': {e}. Will attempt to regenerate.")

        logger.info(f"Generating new face encodings from images in '{self.known_faces_dir}'...")
        processed_persons_count = 0
        total_images_processed = 0

        if not os.path.isdir(self.known_faces_dir):
            logger.error(f"Known faces directory '{self.known_faces_dir}' does not exist or is not a directory. Cannot enroll faces.")
            return

        for person_name in os.listdir(self.known_faces_dir):
            person_dir_path = os.path.join(self.known_faces_dir, person_name)
            if not os.path.isdir(person_dir_path):
                logger.debug(f"Skipping '{person_dir_path}' as it's not a directory.")
                continue 

            images_for_this_person = 0
            for image_name in os.listdir(person_dir_path):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_path = os.path.join(person_dir_path, image_name)
                    try:
                        logger.debug(f"Processing image for enrollment: {image_path}")
                        image = face_recognition.load_image_file(image_path)
                        face_locations = face_recognition.face_locations(image, model=self.detection_model)

                        if face_locations:
                            face_encs = face_recognition.face_encodings(image, known_face_locations=face_locations)
                            if face_encs: 
                                self.known_face_encodings.append(face_encs[0]) 
                                self.known_face_names.append(person_name)
                                images_for_this_person += 1
                                total_images_processed += 1
                            else:
                                logger.warning(f"Could not generate encoding for '{image_path}' even though face locations were found.")
                        else:
                            logger.warning(f"No faces found in enrollment image: '{image_path}' for person '{person_name}'. Skipping this image.")
                    except Exception as e:
                        logger.error(f"Error processing enrollment image '{image_path}': {e}", exc_info=False) 
            
            if images_for_this_person > 0:
                processed_persons_count +=1
                logger.info(f"Enrolled person '{person_name}' with {images_for_this_person} image(s).")

        if self.known_face_encodings:
            logger.info(f"Enrollment complete. Total {processed_persons_count} persons enrolled from {total_images_processed} images, resulting in {len(self.known_face_encodings)} encodings.")
            try:
                encodings_dir = os.path.dirname(self.encodings_path)
                if encodings_dir and not os.path.exists(encodings_dir): 
                    os.makedirs(encodings_dir, exist_ok=True)
                    logger.info(f"Created directory for encodings file: {encodings_dir}")

                with open(self.encodings_path, 'wb') as f:
                    pickle.dump({'encodings': self.known_face_encodings, 'names': self.known_face_names}, f)
                logger.info(f"Saved new known face encodings to '{self.encodings_path}'")
            except Exception as e:
                logger.error(f"Error saving encodings file to '{self.encodings_path}': {e}")
        else:
            logger.warning("No known faces were encoded after processing. Face recognition will not be able to identify anyone. "
                           f"Please ensure images are correctly placed in subdirectories of '{self.known_faces_dir}'.")

    def recognize_faces(self, frame_bgr):
        """
        Detects and recognizes faces in a given BGR frame.
        """
        if not self.known_face_encodings:
            return [] 

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        current_face_locations = face_recognition.face_locations(rgb_frame, model=self.detection_model)
        if not current_face_locations:
            return [] 

        current_face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=current_face_locations)
        recognized_face_results = []

        for i, face_encoding in enumerate(current_face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.tolerance)
            name = "Unknown"
            confidence_score = None 
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            if len(face_distances) > 0: 
                best_match_index = np.argmin(face_distances) 
                if matches[best_match_index]: 
                    name = self.known_face_names[best_match_index]
                    distance_at_match = face_distances[best_match_index]
                    confidence_score = max(0.0, 1.0 - (distance_at_match / self.tolerance))
            
            top, right, bottom, left = current_face_locations[i]
            recognized_face_results.append({
                "name": name,
                "confidence": confidence_score,
                "bbox": (left, top, right, bottom) 
            })

        if recognized_face_results:
             logger.debug(f"Face recognition results for current frame: {recognized_face_results}")
        return recognized_face_results

# Example Usage (for testing this module directly):
if __name__ == '__main__':
    from test_utils import get_test_video_capture
    # This logger will use the configuration from the try-except block at the top.
    logger.info("--- Testing FaceRecognizer Module (Standalone) ---")

    # Use the 'config_object' which is now guaranteed to be defined.
    show_video_flag_fr_test = getattr(config_object, 'SHOW_VIDEO_WINDOW', False) 
    logger.info(f"Display windows for testing (SHOW_VIDEO_WINDOW from config_object): {show_video_flag_fr_test}")

    known_faces_test_dir = getattr(config_object, 'KNOWN_FACES_DIR', 'known_faces_test_standalone')
    if not os.path.exists(known_faces_test_dir):
        try:
            os.makedirs(known_faces_test_dir, exist_ok=True)
            logger.info(f"Test: Created KNOWN_FACES_DIR for testing at {known_faces_test_dir}")
            logger.info(f"Test: IMPORTANT - Please populate subdirectories in '{known_faces_test_dir}' with face images for enrollment testing.")
        except Exception as e:
            logger.error(f"Test: Could not create KNOWN_FACES_DIR '{known_faces_test_dir}' for testing: {e}. Enrollment might fail.")
    else:
        logger.info(f"Test: Using existing KNOWN_FACES_DIR for enrollment: {known_faces_test_dir}")

    face_recognizer_instance_test = FaceRecognizer(
        known_faces_dir=known_faces_test_dir,
        encodings_path=getattr(config_object, 'KNOWN_ENCODINGS_PATH', 'known_face_encodings_standalone_test.pkl'),
        detection_model=getattr(config_object, 'FACE_DETECTION_MODEL', 'hog'),
        tolerance=getattr(config_object, 'FACE_RECOGNITION_TOLERANCE', 0.55)
    )

    if not face_recognizer_instance_test.known_face_encodings:
        logger.warning("Test: No known faces were loaded or enrolled by FaceRecognizer. Recognition will likely only find 'Unknown'.")
    else:
        logger.info(f"Test: FaceRecognizer loaded/enrolled {len(face_recognizer_instance_test.known_face_names)} unique known names.")

    test_video_source_fr = "rtsp://<credentials>:554/cam/realmonitor?channel=3&subtype=0"
    # test_video_source_fr = "my_test_video_with_faces.mp4" 
    # test_video_source_fr = 0 
    
    logger.info(f"Using video source for face recognition test: {test_video_source_fr}")

    cap_test_fr = get_test_video_capture(test_video_source_fr)
    if not cap_test_fr:
        logger.error("Could not initialize video capture. Exiting test.")
        exit()
    
    logger.info("Starting face recognition test. Press 'q' in an OpenCV window to quit (if shown).")

    frame_counter_fr_test = 0
    max_test_frames_fr = getattr(config_object, 'TEST_MAX_FRAMES_FACE_RECOGNITION', 0) 

    while True:
        ret_val_fr, test_frame_fr_bgr = cap_test_fr.read()
        if not ret_val_fr or test_frame_fr_bgr is None:
            logger.info("End of video or cannot read frame. Exiting face recognition test.")
            break
        
        frame_counter_fr_test += 1
        
        recognition_start_time = time.time()
        list_of_recognized_faces = face_recognizer_instance_test.recognize_faces(test_frame_fr_bgr)
        recognition_duration = time.time() - recognition_start_time
        
        if frame_counter_fr_test % 10 == 0: 
            logger.info(f"Frame {frame_counter_fr_test}: Face recognition took {recognition_duration:.3f}s. Found {len(list_of_recognized_faces)} faces.")

        if show_video_flag_fr_test:
            display_frame_fr_test = test_frame_fr_bgr.copy() 
            for face_info in list_of_recognized_faces:
                l, t, r, b = face_info["bbox"]
                name = face_info["name"]
                conf = face_info["confidence"] 
                color = (0, 255, 255) if name != "Unknown" else (0, 0, 255) 
                cv2.rectangle(display_frame_fr_test, (l, t), (r, b), color, 2)
                label = f"{name}"
                if conf is not None and name != "Unknown": 
                    label += f" ({conf*100:.0f}%)" 
                cv2.putText(display_frame_fr_test, label, (l + 6, b - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Face Recognition Test", display_frame_fr_test)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User pressed 'q' to quit display.")
                break
        else:
            if frame_counter_fr_test % 30 == 0: 
                 logger.info(f"Processed {frame_counter_fr_test} frames for face recognition. Faces in last: {len(list_of_recognized_faces)} (Display off)")
            if max_test_frames_fr > 0 and frame_counter_fr_test >= max_test_frames_fr:
                logger.info(f"Reached max_test_frames_fr ({max_test_frames_fr}). Ending headless test.")
                break
                
    cap_test_fr.release()
    if show_video_flag_fr_test:
        cv2.destroyAllWindows()
    logger.info(f"--- FaceRecognizer Module Test Complete (Processed {frame_counter_fr_test} frames) ---")

