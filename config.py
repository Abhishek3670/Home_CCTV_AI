# config.py
import os
import logging 

# --- WSL2 Specific Path Configuration ---
WINDOWS_HDD_MOUNT_POINT = "/mnt/g"

PROJECT_DATA_ON_HDD = os.path.join(WINDOWS_HDD_MOUNT_POINT, "CCTV_AI_Project_Data")

# --- Camera Settings ---
CAMERA_SOURCES = [
    "rtsp://<credentials>:554/cam/realmonitor?channel=1&subtype=0",
    # Add more camera sources here if needed
]
FRAME_WIDTH = 1280  # Desired frame width for processing (can be None to use camera default)
FRAME_HEIGHT = 720 # Desired frame height for processing (can be None)
FPS_LIMIT = 10     # Limit processing FPS to save resources, None for no limit. Affects main loop.

# --- Storage Settings ---
# Recordings will be saved to your Windows G: drive via the WSL2 mount point
RECORDINGS_DIR = os.path.join(PROJECT_DATA_ON_HDD, "recordings")

# --- Face Recognition Settings ---
KNOWN_FACES_DIR = os.path.join(PROJECT_DATA_ON_HDD, "ai_datasets", "known_faces")
KNOWN_ENCODINGS_PATH = os.path.join(PROJECT_DATA_ON_HDD, "ai_datasets", "known_faces_encodings.pkl")
FACE_DETECTION_MODEL = "hog" # 'hog' (faster) or 'cnn' (more accurate, needs GPU for good speed)
FACE_RECOGNITION_TOLERANCE = 0.55 # Lower is stricter (0.0 to 1.0 for distance)

# --- Motion Detection Settings ---
MOTION_HISTORY = 500
MOTION_VAR_THRESHOLD = 60 # Higher values mean less sensitivity to motion
MOTION_DETECT_SHADOWS = True # If True, tries to identify and ignore shadows
MOTION_MIN_CONTOUR_AREA = 700 # Minimum size of a moving object to be considered significant
MOTION_COOLDOWN_SECONDS = 5   # Cooldown before re-triggering motion for the same source

# --- Object Detection Settings ---
# Store models within the WSL2 filesystem (e.g., in your project directory) for potentially faster loading.
# Or place on /mnt/g/ if preferred and update path.
# Assumes model (e.g., yolov8s.pt) is in the same directory as main_controller.py or current working directory.
OBJECT_DETECTION_MODEL_PATH = 'mnt/f/AI_Models/yolov8s.pt' 
OBJECT_CONFIDENCE_THRESHOLD = 0.45 # Minimum confidence to consider an object detected
# List of specific object classes to focus on for triggering events (e.g., recording, further analysis)
# Refer to COCO dataset names if using a pre-trained YOLO model (e.g., 'person', 'car', 'dog')
# An empty list or None means consider all detected objects above threshold.
FOCUSED_OBJECT_CLASSES = ['person', 'car', 'bicycle', 'motorcycle', 'dog', 'cat', 'bird']
# Only run face recognition if one of these object classes is detected by the object detector
PROCESS_FACES_FOR_CLASSES = ['person']

# --- Event & Recording Settings ---
RECORD_SECONDS_BEFORE_EVENT = 5  # How many seconds of footage to include before an event trigger
RECORD_SECONDS_AFTER_EVENT = 10  # How many seconds to record after the last detected activity for an event
MAX_RECORDING_MINUTES = 5      # Max duration for a single event recording clip to prevent excessively long files
EVENT_COOLDOWN_SECONDS = 30    # Cooldown for logging similar significant events to avoid spamming logs

# --- Display & Logging Settings ---
# To show video window from WSL2, you'll need an X Server running on Windows (e.g., VcXsrv, X410)
# and DISPLAY environment variable set in WSL2 (e.g., export DISPLAY=$(ip route|awk '/default/ {print $3}'):0.0 or export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0)
# For a headless server setup, set SHOW_VIDEO_WINDOW = False
SHOW_VIDEO_WINDOW = True # Set to True if you have an X Server configured and want to see output windows
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL (standard logging levels)
LOG_FILE_DIR = os.path.join(PROJECT_DATA_ON_HDD, "logs")
LOG_FILE = os.path.join(LOG_FILE_DIR, "cctv_ai_app.log") # Log to HDD

# --- Performance ---
# Process every Nth frame for heavy AI tasks (object/face detection) to save resources,
# especially if FPS_LIMIT is high or multiple cameras are used.
# e.g., if FPS_LIMIT is 30 and AI_PROCESSING_FRAME_INTERVAL is 3, AI effectively runs at 10 FPS on those frames.
AI_PROCESSING_FRAME_INTERVAL = 2

# --- Ensure base directories exist on the G: drive (via /mnt/g) ---
# It's best to create the root PROJECT_DATA_ON_HDD (e.g., G:\CCTV_AI_Project_Data) manually in Windows first.
# Python can then create the subdirectories.
if not os.path.exists(PROJECT_DATA_ON_HDD):
    print(f"WARNING: Base data directory '{PROJECT_DATA_ON_HDD}' (mapped from G: drive) not found.")
    print(f"Please create G:\\CCTV_AI_Project_Data manually on your Windows host system.")
    # Consider raising an error or exiting if this critical path doesn't exist.

# Create subdirectories if they don't exist.
try:
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True) # Also creates PROJECT_DATA_ON_HDD/ai_datasets if needed
    os.makedirs(LOG_FILE_DIR, exist_ok=True)
    print(f"Data directories checked/created under: {PROJECT_DATA_ON_HDD}")
except OSError as e:
    print(f"ERROR: Could not create required data directories under '{PROJECT_DATA_ON_HDD}': {e}")
    print("Please check permissions and ensure the base path on G: drive is accessible from WSL2.")

# Basic logging configuration check (actual setup is in main_controller.py)
# This is just to inform the user if config is loaded.
print(f"Config loaded. Log file target: {LOG_FILE}")
print(f"Display window for main application: {SHOW_VIDEO_WINDOW}")

