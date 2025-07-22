# Home CCTV AI - Phase 1 Complete ✅

An intelligent CCTV surveillance system powered by AI that provides real-time object detection, motion detection, face recognition, and automatic video recording capabilities. **Phase 1 of the integration with Luks_storage is now complete.**

## 🎯 Project Status - Phase 1 Complete

### ✅ Phase 1: API Server Implementation (COMPLETED)
- **✅ FastAPI Framework**: Modern, fast Python web framework integrated
- **✅ API Server**: `api_server.py` fully implemented and functional
- **✅ Core Endpoints**: Status, camera management, and live streaming endpoints
- **✅ Background Processing**: CCTV processing runs in dedicated background threads
- **✅ Production Ready**: Code refactored and optimized for distributed systems

### 🔄 Phase 2: Rust Backend Integration (NEXT)
- **⏳ Python Process Management**: Rust subprocess handling
- **⏳ HTTP Client Integration**: reqwest crate implementation  
- **⏳ Rust API Client**: Communication layer development
- **⏳ Backend Endpoints**: Secure proxy endpoints creation

### 📋 Phase 3: Frontend UI Development (PLANNED)
- **⏳ CCTV Management Page**: Web interface creation
- **⏳ Navigation Integration**: Seamless UI integration
- **⏳ Live Streaming**: Real-time video display
- **⏳ Recordings Browser**: Video playback interface

## 🚀 API Features (Phase 1 Complete)

### 🎯 Real-Time Object Detection
- Uses YOLO (You Only Look Once) models to detect and classify objects
- Supports persons, vehicles, animals, and more
- GPU/CPU acceleration support

### 👁️ Motion Detection  
- Advanced background subtraction algorithms
- Configurable sensitivity settings
- Motion event triggering

### 👤 Face Recognition
- Identify known individuals and distinguish from strangers
- Face encodings with configurable tolerance
- Support for multiple face databases

### 📹 Automatic Video Recording
- Event-triggered recording with pre/post-event footage
- Configurable recording duration and quality
- Circular buffer for continuous recording

### 📊 Multi-Camera Support
- Handle multiple camera sources simultaneously
- RTSP, USB, and video file support
- Independent camera configuration

### 🔧 Configurable Settings
- Comprehensive configuration through `config.py`
- JSON-based camera configuration
- Runtime parameter adjustment

### ⚡ Performance Optimized
- Frame processing intervals and FPS limiting
- Resource management and monitoring
- Distributed processing architecture

## 📋 API Endpoints (Phase 1)

### Camera Management
- `GET /cameras` - List all configured cameras
- `GET /cameras/{camera_id}` - Get specific camera details
- `POST /cameras` - Add new camera configuration
- `PUT /cameras/{camera_id}` - Update camera settings
- `DELETE /cameras/{camera_id}` - Remove camera

### System Status
- `GET /status` - Get system health and status
- `GET /health` - Health check endpoint

### Live Streaming  
- `GET /stream/{camera_id}` - Live MJPEG stream
- `GET /snapshot/{camera_id}` - Single frame capture

### Recordings
- `GET /recordings` - List available recordings
- `GET /recordings/{filename}` - Download specific recording

## 🛠️ Installation & Setup

### System Requirements
- Python 3.7+
- OpenCV with FFMPEG support (for RTSP streams)
- CUDA-compatible GPU (optional, for faster processing)

### Python Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies:
- `fastapi>=0.68.0` - Modern web framework
- `uvicorn>=0.15.0` - ASGI server
- `opencv-python>=4.5.0` - Computer vision
- `ultralytics>=8.0.0` - YOLO models
- `face-recognition>=1.3.0` - Face recognition
- `numpy>=1.21.0` - Numerical computing
- `torch>=1.9.0` - Deep learning framework

## ⚙️ Configuration

### Camera Configuration (`cameras.json`)
```json
[
  {
    "id": "cam1",
    "name": "Front Door",
    "url": "rtsp://admin:password@192.168.1.100:554/stream"
  },
  {
    "id": "cam2", 
    "name": "Back Yard",
    "url": "rtsp://admin:password@192.168.1.101:554/stream"
  }
]
```

### System Configuration (`config.py`)
- **Storage Settings**: Recordings and logs directory
- **Remote GPU Settings**: Distributed processing configuration
- **AI Models**: Object detection and face recognition models
- **Recording Settings**: Duration, quality, and triggers
- **Motion Detection**: Sensitivity and parameters

## 🚀 Running the API Server (Phase 1)

### Development Mode
```bash
python api_server.py
```

### Production Mode
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8082 --reload
```

### Background Service Mode
```bash
# The main controller can run independently
python main_controller.py
```

## 🔧 Advanced Configuration

### GPU Acceleration
The system automatically detects and uses CUDA if available:
```python
# In config.py
REMOTE_GPU_ENABLED = True
REMOTE_GPU_SERVER_URL = "http://192.168.29.78:5000"
FALLBACK_TO_CPU = False
```

### Custom Object Classes
Focus detection on specific objects:
```python
FOCUSED_OBJECT_CLASSES = ['person', 'car', 'truck', 'bicycle', 'dog', 'cat']
```

### Recording Optimization
- `AI_PROCESSING_FRAME_INTERVAL`: Process every Nth frame
- `FPS_LIMIT`: Hardware-appropriate frame rate
- `MAX_RECORDING_MINUTES`: Prevent excessive file sizes

## 🐛 Troubleshooting

### Common Issues

1. **Camera Connection Issues**
   - Verify RTSP URL format and credentials
   - Check network connectivity and firewall settings

2. **CUDA/GPU Issues** 
   - Install CUDA toolkit and compatible PyTorch version
   - System falls back to CPU if CUDA unavailable

3. **API Server Issues**
   - Check port 8082 availability
   - Verify FastAPI and uvicorn installation
   - Review logs for startup errors

### Logging
- Log level configurable in `config.py`
- Detailed timestamps and error messages
- Separate logs for API server and core processing

## 📞 Integration Status

This API server is ready for **Phase 2 integration** with the Luks_storage Rust backend. The API provides all necessary endpoints for:

- Camera management and configuration
- Live video streaming via MJPEG
- Recording access and playback
- System status and health monitoring
- Real-time AI processing results

## 🔮 Next Steps (Phase 2)

1. **Rust Backend Integration**
   - Add process management for Python API server
   - Implement HTTP client for API communication
   - Create secure proxy endpoints
   - Add authentication and authorization

2. **Advanced Features**
   - WebSocket support for real-time updates
   - Advanced analytics and reporting
   - Mobile app integration
   - Cloud storage integration

---

**Phase 1 Status**: ✅ **COMPLETE** - Ready for Rust backend integration
**Current Branch**: `phase-1-api-complete`
**Next Phase**: Integration with Luks_storage Rust backend

