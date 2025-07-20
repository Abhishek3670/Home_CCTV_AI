# Home CCTV AI

An intelligent CCTV surveillance system powered by AI that provides real-time object detection, motion detection, face recognition, and automatic video recording capabilities.

## 🚀 Features

- **🎯 Real-Time Object Detection**: Uses YOLO (You Only Look Once) models to detect and classify objects like persons, vehicles, animals, and more
- **👁️ Motion Detection**: Advanced background subtraction algorithms to detect motion with configurable sensitivity
- **👤 Face Recognition**: Identify known individuals and distinguish them from strangers using face encodings
- **📹 Automatic Video Recording**: Records video clips triggered by events with configurable pre-event and post-event footage
- **📊 Multi-Camera Support**: Handle multiple camera sources simultaneously (RTSP, USB, video files)
- **🔧 Configurable Settings**: Highly customizable through configuration files
- **📝 Comprehensive Logging**: Detailed logging for monitoring and troubleshooting
- **⚡ Performance Optimized**: Frame processing intervals and FPS limiting for resource management

## 🏗️ Architecture

The system consists of modular components:

- **`main_controller.py`**: Main orchestrator that coordinates all modules
- **`camera_manager.py`**: Handles camera connections and frame retrieval with reconnection logic
- **`motion_detector.py`**: Motion detection using MOG2 background subtraction
- **`object_detector.py`**: YOLO-based object detection with GPU/CPU support
- **`face_recognizer.py`**: Face detection and recognition using face_recognition library
- **`video_recorder.py`**: Threaded video recording with circular buffer for pre-event footage
- **`config.py`**: Configuration management for all system parameters

## 📋 Requirements

### System Requirements
- Python 3.7+
- OpenCV with FFMPEG support (for RTSP streams)
- CUDA-compatible GPU (optional, for faster object detection)

### Python Dependencies
```
opencv-python>=4.5.0
ultralytics>=8.0.0
face-recognition>=1.3.0
numpy>=1.21.0
torch>=1.9.0
torchvision>=0.10.0
```

## 🛠️ Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Abhishek3670/Home_CCTV_AI.git
   cd Home_CCTV_AI
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install opencv-python ultralytics face-recognition numpy torch torchvision
   ```

3. **Download YOLO Model** (if not using default)
   - The system will automatically download `yolov8s.pt` on first run
   - For custom models, place them in the specified path in `config.py`

4. **Setup Face Recognition** (optional)
   - Create a `known_faces` directory structure:
     ```
     known_faces/
     ├── person1/
     │   ├── photo1.jpg
     │   └── photo2.jpg
     └── person2/
         ├── photo1.jpg
         └── photo2.jpg
     ```

## ⚙️ Configuration

Edit `config.py` to customize the system:

### Camera Settings
```python
CAMERA_SOURCES = [
    "rtsp://username:password@camera_ip:554/stream_path",
    0,  # USB camera
    "path/to/video/file.mp4"
]
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_LIMIT = 10
```

### AI Model Settings
```python
OBJECT_DETECTION_MODEL_PATH = 'yolov8s.pt'
OBJECT_CONFIDENCE_THRESHOLD = 0.45
FOCUSED_OBJECT_CLASSES = ['person', 'car', 'bicycle', 'motorcycle']

FACE_DETECTION_MODEL = "hog"  # or "cnn" for better accuracy
FACE_RECOGNITION_TOLERANCE = 0.55
```

### Recording Settings
```python
RECORDINGS_DIR = "/path/to/recordings"
RECORD_SECONDS_BEFORE_EVENT = 5
RECORD_SECONDS_AFTER_EVENT = 10
MAX_RECORDING_MINUTES = 5
```

### Motion Detection
```python
MOTION_VAR_THRESHOLD = 60  # Higher = less sensitive
MOTION_MIN_CONTOUR_AREA = 700
MOTION_COOLDOWN_SECONDS = 5
```

## 🚀 Usage

### Basic Usage
```bash
python main_controller.py
```

### Testing Individual Modules
Each module can be tested independently:

```bash
# Test camera connection
python camera_manager.py

# Test motion detection
python motion_detector.py

# Test object detection
python object_detector.py

# Test face recognition
python face_recognizer.py

# Test video recording
python video_recorder.py
```

### Display Options
- Set `SHOW_VIDEO_WINDOW = True` in config to see live video feeds
- Press 'q' in any video window to quit
- For headless operation, set `SHOW_VIDEO_WINDOW = False`

## 📁 Directory Structure

```
Home_CCTV_AI/
├── main_controller.py      # Main application entry point
├── camera_manager.py       # Camera handling and connections
├── motion_detector.py      # Motion detection algorithms
├── object_detector.py      # YOLO object detection
├── face_recognizer.py      # Face recognition system
├── video_recorder.py       # Video recording with threading
├── config.py              # Configuration settings
├── README.md              # This file
├── .gitignore
└── mnt/f/AI_Models/       # AI model storage
    └── yolov8s.pt
```

## 🔧 Advanced Configuration

### GPU Acceleration
The system automatically detects and uses CUDA if available. To force CPU usage:
```python
# In object_detector.py, modify the _load_model method
self.device = 'cpu'
```

### Custom Object Classes
Modify `FOCUSED_OBJECT_CLASSES` in config.py to focus on specific objects:
```python
FOCUSED_OBJECT_CLASSES = ['person', 'car', 'truck', 'bus', 'bicycle', 'dog', 'cat']
```

### Recording Optimization
- Adjust `AI_PROCESSING_FRAME_INTERVAL` to process every Nth frame
- Set appropriate `FPS_LIMIT` based on your hardware capabilities
- Configure `MAX_RECORDING_MINUTES` to prevent excessive file sizes

## 🐛 Troubleshooting

### Common Issues

1. **Camera Connection Issues**
   - Verify RTSP URL format and credentials
   - Ensure camera supports the specified stream format
   - Check network connectivity and firewall settings

2. **CUDA/GPU Issues**
   - Install CUDA toolkit and compatible PyTorch version
   - Verify GPU drivers are up to date
   - System will fall back to CPU if CUDA is unavailable

3. **Face Recognition Performance**
   - Use "hog" model for faster processing on CPU
   - Use "cnn" model for better accuracy with GPU
   - Ensure good quality, well-lit face images for enrollment

4. **Video Recording Issues**
   - Check disk space and write permissions
   - Verify output directory exists and is writable
   - Monitor system resources during recording

### Logging
Check the log files for detailed error information:
- Log level can be adjusted in `config.py` (DEBUG, INFO, WARNING, ERROR)
- Logs include timestamps, module names, and detailed error messages

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow Python PEP 8 style guidelines
- Add appropriate error handling and logging
- Update documentation for new features
- Test modules independently before integration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for the YOLO implementation
- [face_recognition](https://github.com/ageitgey/face_recognition) library by Adam Geitgey
- OpenCV community for computer vision tools
- PyTorch team for the deep learning framework

## 📞 Support

For support, questions, or feature requests:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the configuration documentation

## 🔮 Future Enhancements

- [ ] Web-based dashboard for remote monitoring
- [ ] Mobile app integration
- [ ] Cloud storage integration
- [ ] Advanced analytics and reporting
- [ ] Multiple AI model support
- [ ] Real-time alerts and notifications
- [ ] Integration with home automation systems

---

**Note**: This system is designed for legitimate surveillance purposes only. Please ensure compliance with local privacy laws and regulations when deploying CCTV systems.
