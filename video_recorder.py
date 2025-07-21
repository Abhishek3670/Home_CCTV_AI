# video_recorder.py
import cv2
import os
import time
import queue
import threading
from datetime import datetime
import logging

try:
    import config as config
    logger = logging.getLogger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("VideoRecorder_standalone")
    class FallbackConfig:
        RECORDINGS_DIR = "recordings_test"
        RECORD_SECONDS_BEFORE_EVENT = 2
        RECORD_SECONDS_AFTER_EVENT = 5
        MAX_RECORDING_MINUTES = 1
        FPS_LIMIT = 15
        FRAME_WIDTH = 640
        FRAME_HEIGHT = 480
    config = FallbackConfig()
    logger.info("Running video_recorder.py in standalone mode with fallback configuration.")

class VideoRecorder:
    def __init__(self, camera_id, 
                 base_recordings_dir=config.RECORDINGS_DIR,
                 pre_event_seconds=config.RECORD_SECONDS_BEFORE_EVENT,
                 post_event_seconds=config.RECORD_SECONDS_AFTER_EVENT,
                 max_clip_duration_minutes=config.MAX_RECORDING_MINUTES,
                 target_fps=None):
        """
        Initializes the VideoRecorder.
        Args:
            camera_id (str): Identifier for the camera (used in filenames/paths).
            base_recordings_dir (str): Base directory to save recordings.
            pre_event_seconds (int): Seconds of footage to buffer before an event.
            post_event_seconds (int): Seconds to record after an event trigger ends.
            max_clip_duration_minutes (int): Maximum duration for a single recording clip.
            target_fps (int, optional): Desired FPS for the recording. If None, uses config.FPS_LIMIT or a default.
        """
        self.camera_id_sanitized = str(camera_id).replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
        self.base_recordings_dir = base_recordings_dir
        self.pre_event_seconds = pre_event_seconds
        self.post_event_seconds = post_event_seconds
        self.max_clip_duration_seconds = max_clip_duration_minutes * 60
        self.target_fps = target_fps if target_fps is not None else getattr(config, 'FPS_LIMIT', 15)
        if self.target_fps is None or self.target_fps <= 0:
            self.target_fps = 15 
            logger.warning(f"[{self.camera_id_sanitized}] Invalid target_fps, defaulting to 15 FPS for recording.")
        buffer_duration_seconds = self.pre_event_seconds + 10 
        self.buffer_max_size = int(buffer_duration_seconds * self.target_fps)
        self.buffer = queue.Queue(maxsize=self.buffer_max_size)
        self.is_recording = False
        self.recording_thread = None
        self.stop_recording_event = threading.Event()
        self.current_clip_writer = None
        self.current_clip_path = None
        self.current_clip_start_time_actual = None
        self.last_event_trigger_time = 0
        self.output_dir_for_camera = os.path.join(self.base_recordings_dir, self.camera_id_sanitized)
        try:
            os.makedirs(self.output_dir_for_camera, exist_ok=True)
        except OSError as e:
            logger.error(f"[{self.camera_id_sanitized}] Failed to create output directory {self.output_dir_for_camera}: {e}. Recordings may fail.")
        logger.info(f"VideoRecorder for '{self.camera_id_sanitized}' initialized. Output to: '{self.output_dir_for_camera}'. Buffer size: {self.buffer_max_size} frames.")

    def add_frame(self, frame_bgr):
        """
        Adds a BGR frame to the internal buffer.
        """
        if frame_bgr is None:
            return
        timestamped_frame = (time.time(), frame_bgr.copy())
        if self.buffer.full():
            try:
                self.buffer.get_nowait()
            except queue.Empty:
                pass
        self.buffer.put_nowait(timestamped_frame)

    def _start_new_clip_writer(self, event_description="event"):
        """
        Initializes a new cv2.VideoWriter for a new recording clip.
        """
        if self.current_clip_writer is not None:
            logger.warning(f"[{self.camera_id_sanitized}] Attempted to start a new clip while one is already active. Finalizing previous.")
            self._finalize_clip()
        now = datetime.now()
        safe_event_desc = "".join(c if c.isalnum() or c in ('_','-') else '_' for c in event_description)[:30]
        filename = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{safe_event_desc}.mp4"
        self.current_clip_path = os.path.join(self.output_dir_for_camera, filename)
        frame_width = getattr(config, 'FRAME_WIDTH', 640)
        frame_height = getattr(config, 'FRAME_HEIGHT', 480)
        if not self.buffer.empty():
            try:
                _, sample_frame = self.buffer.queue[0]
                h, w, _ = sample_frame.shape
                frame_width, frame_height = w, h
            except Exception as e:
                logger.warning(f"[{self.camera_id_sanitized}] Could not get frame dimensions from buffer for VideoWriter: {e}. Using defaults.")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        try:
            self.current_clip_writer = cv2.VideoWriter(self.current_clip_path, fourcc, self.target_fps, (frame_width, frame_height))
            if not self.current_clip_writer.isOpened():
                logger.error(f"[{self.camera_id_sanitized}] Failed to open VideoWriter for {self.current_clip_path}. Check codec, path, permissions.")
                self.current_clip_writer = None
                return False
            logger.info(f"[{self.camera_id_sanitized}] Started new recording clip: {self.current_clip_path} at {self.target_fps} FPS, {frame_width}x{frame_height}")
            self.current_clip_start_time_actual = time.time()
            return True
        except Exception as e:
            logger.error(f"[{self.camera_id_sanitized}] Exception initializing VideoWriter for {self.current_clip_path}: {e}")
            self.current_clip_writer = None
            return False

    def _finalize_clip(self):
        """
        Releases the current cv2.VideoWriter and resets clip-specific variables.
        """
        if self.current_clip_writer is not None:
            logger.info(f"[{self.camera_id_sanitized}] Finalizing recording clip: {self.current_clip_path}")
            self.current_clip_writer.release()
            self.current_clip_writer = None
            self.current_clip_path = None
            self.current_clip_start_time_actual = None
        else:
            logger.debug(f"[{self.camera_id_sanitized}] Finalize clip called but no active writer.")

    def _recording_worker_thread_func(self, initial_event_description):
        """
        The actual recording logic that runs in a separate thread.
        """
        if not self._start_new_clip_writer(initial_event_description):
            logger.error(f"[{self.camera_id_sanitized}] Worker thread: Failed to start video writer. Aborting recording.")
            self.is_recording = False
            self.stop_recording_event.clear()
            return
        pre_event_frames_to_write = []
        while not self.buffer.empty():
            try:
                pre_event_frames_to_write.append(self.buffer.get_nowait())
            except queue.Empty:
                break
        num_pre_event_frames_target = int(self.pre_event_seconds * self.target_fps)
        actual_pre_event_frames = pre_event_frames_to_write[-num_pre_event_frames_target:]
        logger.info(f"[{self.camera_id_sanitized}] Writing {len(actual_pre_event_frames)} pre-event frames to {self.current_clip_path}...")
        for _, frame_data in actual_pre_event_frames:
            if self.current_clip_writer:
                 self.current_clip_writer.write(frame_data)
        frames_written_this_session = len(actual_pre_event_frames)
        while not self.stop_recording_event.is_set():
            current_time = time.time()
            if self.current_clip_start_time_actual and (current_time - self.current_clip_start_time_actual) >= self.max_clip_duration_seconds:
                logger.info(f"[{self.camera_id_sanitized}] Max clip duration ({self.max_clip_duration_seconds}s) reached for {self.current_clip_path}.")
                break 
            if current_time > (self.last_event_trigger_time + self.post_event_seconds):
                logger.info(f"[{self.camera_id_sanitized}] Post-event duration ({self.post_event_seconds}s) passed for {self.current_clip_path}.")
                break
            try:
                _timestamp, frame_to_write = self.buffer.get(timeout=0.1)
                if self.current_clip_writer:
                    self.current_clip_writer.write(frame_to_write)
                    frames_written_this_session += 1
            except queue.Empty:
                time.sleep(0.01)
                continue
            except Exception as e:
                logger.error(f"[{self.camera_id_sanitized}] Error writing frame during recording: {e}")
                break
        logger.info(f"[{self.camera_id_sanitized}] Recording worker finished. Total frames written to {self.current_clip_path}: {frames_written_this_session}")
        self._finalize_clip()
        self.is_recording = False
        self.stop_recording_event.clear()

    def start_or_update_recording(self, event_description="event"):
        """
        Starts a new recording if not already recording, or updates the
        last_event_trigger_time if already recording to extend it.
        """
        current_time = time.time()
        self.last_event_trigger_time = current_time
        if not self.is_recording:
            self.is_recording = True
            self.stop_recording_event.clear()
            logger.info(f"[{self.camera_id_sanitized}] Starting new recording due to: {event_description}")
            self.recording_thread = threading.Thread(target=self._recording_worker_thread_func, args=(event_description,))
            self.recording_thread.daemon = True
            self.recording_thread.start()
        else:
            logger.info(f"[{self.camera_id_sanitized}] Already recording. Event '{event_description}' updated last trigger time. Recording will extend.")

    def stop_recording_if_idle(self):
        """
        Checks if the recording should be stopped based on post_event_seconds from last trigger.
        This method is intended to be called periodically by the main loop if needed,
        though the worker thread handles its own timeout.
        This can act as a failsafe or an alternative way to manage stopping.
        """
        if self.is_recording and time.time() > (self.last_event_trigger_time + self.post_event_seconds):
            logger.info(f"[{self.camera_id_sanitized}] Idle period detected. Signaling recording thread to stop.")
            self.signal_stop()

    def signal_stop(self):
        """
        Signals the recording thread to stop gracefully.
        The thread will finish writing its current buffer/post-event frames.
        """
        if self.is_recording:
            logger.info(f"[{self.camera_id_sanitized}] Received signal to stop recording.")
            self.stop_recording_event.set()

    def force_stop_and_join(self, timeout_seconds=10):
        """
        For application shutdown: signals stop and waits for thread to finish.
        """
        if self.is_recording and self.recording_thread and self.recording_thread.is_alive():
            logger.info(f"[{self.camera_id_sanitized}] Force stopping recording and joining thread...")
            self.stop_recording_event.set()
            self.recording_thread.join(timeout=timeout_seconds)
            if self.recording_thread.is_alive():
                logger.warning(f"[{self.camera_id_sanitized}] Recording thread did not terminate cleanly after {timeout_seconds}s on force_stop.")
            else:
                logger.info(f"[{self.camera_id_sanitized}] Recording thread joined successfully on force_stop.")
        self.is_recording = False
        self._finalize_clip()
