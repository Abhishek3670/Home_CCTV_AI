# config.py - Final configuration with LUKS storage and remote AI models
import os
import logging

# =============================================================================
# STORAGE SETTINGS
# =============================================================================
# All data will be stored in a 'data' directory inside the project folder.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_STORAGE_DIR = os.path.join(PROJECT_ROOT, "data")
RECORDINGS_DIR = os.path.join(BASE_STORAGE_DIR, "recordings")
LOG_FILE_DIR = os.path.join(BASE_STORAGE_DIR, "logs")
LOG_FILE = os.path.join(LOG_FILE_DIR, "cctv_ai_app.log")
KNOWN_FACES_DIR = os.path.join(BASE_STORAGE_DIR, "known_faces")
KNOWN_ENCODINGS_PATH = os.path.join(BASE_STORAGE_DIR, "known_faces_encodings.pkl")

# =============================================================================
# REMOTE GPU SETTINGS (CRITICAL)
# =============================================================================
REMOTE_GPU_ENABLED = True
REMOTE_GPU_SERVER_URL = "http://192.168.29.78:5000"  # Your local machine IP
FALLBACK_TO_CPU = False  # Force GPU usage for better performance
GPU_CONNECTION_TIMEOUT = 30  # Timeout for GPU requests
GPU_RETRY_ATTEMPTS = 3  # Number of retry attempts

# =============================================================================
# AI MODELS SETTINGS (REMOTE STORAGE)
# =============================================================================
# Models are stored on your local machine F:\AI_Models
# The GPU server will access them directly from F:\AI_Models\
REMOTE_AI_MODELS_PATH = "F:/AI_Models"  # Path on your local Windows machine
OBJECT_DETECTION_MODEL_NAME = "yolov8s.pt"  # Model filename
FACE_RECOGNITION_MODEL_NAME = "face_recognition_models"  # Face models directory

# Local model cache (optional - for fallback)
LOCAL_MODEL_CACHE = "/tmp/ai_models_cache"

# =============================================================================
# SECURE STORAGE SETTINGS (LUKS ENCRYPTED)
# =============================================================================
# Use LUKS encrypted storage for all recordings and sensitive data
# SECURE_STORAGE_MOUNT = "/mnt/secure_data"
# BASE_STORAGE_DIR = SECURE_STORAGE_MOUNT
# RECORDINGS_DIR = os.path.join(BASE_STORAGE_DIR, "cctv_recordings")
# LOG_FILE_DIR = os.path.join(BASE_STORAGE_DIR, "logs")
# LOG_FILE = os.path.join(LOG_FILE_DIR, "cctv_ai_app.log")

# Backup storage (in case LUKS is unmounted)
# BACKUP_STORAGE_DIR = "/tmp/cctv_backup"
# BACKUP_RECORDINGS_DIR = os.path.join(BACKUP_STORAGE_DIR, "recordings")
# BACKUP_LOG_DIR = os.path.join(BACKUP_STORAGE_DIR, "logs")

# =============================================================================
# CAMERA SETTINGS
# =============================================================================
CAMERA_SOURCES = [
    # Primary RTSP Camera
    "rtsp://admin:admin%40123@192.168.29.56:554/cam/realmonitor?channel=1&subtype=0",
    
    # Backup options (uncomment if needed)
    # 0,  # USB camera index 0
    # "rtsp://username:password@another_camera:554/stream",
    # "/path/to/test/video.mp4",
    # "http://192.168.1.100:8080/video",
]

# Camera resolution - Optimized for remote GPU processing
FRAME_WIDTH = 1280   # Reduced for better network performance
FRAME_HEIGHT = 720  # Reduced for better network performance
FPS_LIMIT = 5       # Optimized for remote GPU processing

# =============================================================================
# PERFORMANCE SETTINGS (OPTIMIZED FOR REMOTE GPU)
# =============================================================================
AI_PROCESSING_FRAME_INTERVAL = 3  # Process every 3rd frame

# Image compression for network transfer
IMAGE_QUALITY = 70  # JPEG quality (0-100, lower = faster transfer)
RESIZE_BEFORE_GPU = True  # Resize frames before sending to GPU
GPU_PROCESSING_SIZE = (416, 416)  # YOLOv8 optimal input size

# =============================================================================
# MOTION DETECTION SETTINGS
# =============================================================================
MOTION_HISTORY = 300
MOTION_VAR_THRESHOLD = 40  # More sensitive detection
MOTION_DETECT_SHADOWS = False  # Disabled for performance
MOTION_MIN_CONTOUR_AREA = 500
MOTION_COOLDOWN_SECONDS = 3

# =============================================================================
# OBJECT DETECTION SETTINGS
# =============================================================================
OBJECT_CONFIDENCE_THRESHOLD = 0.5
FOCUSED_OBJECT_CLASSES = [
    'person',      # People detection
    'car',         # Vehicles
    'bicycle',     # Bikes
    'motorcycle',  # Motorbikes
    'dog',         # Pets
    'cat',         # Pets
    'bird'         # Animals
]

# Classes that trigger face recognition (if enabled)
PROCESS_FACES_FOR_CLASSES = ['person']

# =============================================================================
# RECORDING SETTINGS
# =============================================================================
RECORD_SECONDS_BEFORE_EVENT = 3   # Pre-event recording
RECORD_SECONDS_AFTER_EVENT = 7    # Post-event recording
MAX_RECORDING_MINUTES = 3         # Maximum recording length
EVENT_COOLDOWN_SECONDS = 15       # Cooldown between events

# Recording format and quality
VIDEO_CODEC = 'mp4v'  # Video codec
VIDEO_FPS = 15        # Recording FPS
VIDEO_QUALITY = 80    # Video quality (0-100)

# =============================================================================
# SECURITY SETTINGS
# =============================================================================
# Encryption settings for recordings (if needed)
ENCRYPT_RECORDINGS = False  # Enable if you want additional encryption
ENCRYPTION_KEY_FILE = os.path.join(BASE_STORAGE_DIR, ".encryption_key")

# Access control
REQUIRE_AUTHENTICATION = False  # Enable for remote access control
API_KEY_FILE = os.path.join(BASE_STORAGE_DIR, ".api_keys")

# =============================================================================
# DISPLAY & LOGGING SETTINGS
# =============================================================================
SHOW_VIDEO_WINDOW = False  # Keep False for headless server operation
LOG_LEVEL = "INFO"         # Options: DEBUG, INFO, WARNING, ERROR

# Advanced logging
ENABLE_PERFORMANCE_LOGGING = True  # Log performance metrics
LOG_GPU_STATS = True              # Log GPU processing times
LOG_FRAME_SKIP_INFO = True        # Log when frames are skipped
LOG_STORAGE_STATS = True          # Log storage usage

# =============================================================================
# FACE RECOGNITION SETTINGS (OPTIONAL)
# =============================================================================
ENABLE_FACE_RECOGNITION = False   # Disabled due to installation complexity
# KNOWN_FACES_DIR = os.path.join(BASE_STORAGE_DIR, "known_faces")
# KNOWN_ENCODINGS_PATH = os.path.join(BASE_STORAGE_DIR, "known_faces_encodings.pkl")
FACE_DETECTION_MODEL = "hog"      # Use 'hog' for CPU, 'cnn' for GPU
FACE_RECOGNITION_TOLERANCE = 0.6

# =============================================================================
# NETWORK & RELIABILITY SETTINGS
# =============================================================================
NETWORK_TIMEOUT = 10  # Timeout for network requests
MAX_RETRY_DELAY = 5   # Max delay between retries

# Connection health monitoring
GPU_HEALTH_CHECK_INTERVAL = 60  # Check GPU server health every 60 seconds
AUTO_RECONNECT = True           # Auto-reconnect to GPU server if connection lost

# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================
print("=" * 70)
print("SECURE CCTV AI SYSTEM CONFIGURATION")
print("=" * 70)
print(f"Remote GPU Server: {REMOTE_GPU_SERVER_URL}")
print(f"AI Models Location: {REMOTE_AI_MODELS_PATH} (on local machine)")
print(f"Storage Location: {BASE_STORAGE_DIR}")
print(f"Recordings Directory: {RECORDINGS_DIR}")
print(f"Camera Sources: {len(CAMERA_SOURCES)} configured")
print(f"Frame Size: {FRAME_WIDTH}x{FRAME_HEIGHT}")
print(f"FPS Limit: {FPS_LIMIT}")
print(f"AI Processing: Every {AI_PROCESSING_FRAME_INTERVAL} frames")
# print(f"LUKS Storage: {'✓ Mounted' if os.path.ismount(SECURE_STORAGE_MOUNT) else '✗ Not mounted'}")
print(f"Face Recognition: {'Enabled' if ENABLE_FACE_RECOGNITION else 'Disabled'}")
print(f"Video Display: {'Enabled' if SHOW_VIDEO_WINDOW else 'Disabled'}")
print("=" * 70)
