#!/usr/bin/python3.10
"""SmartPepper Camera Continuous Ingestion Service — optimized for LXC system Python."""

import sys, os, io, time, json, base64, signal, glob, traceback
from datetime import datetime

# Validate critical libraries before proceeding
for lib_name in ['cv2', 'numpy', 'requests', 'paho.mqtt']:
    try:
        if lib_name == 'cv2':
            import cv2
        elif lib_name == 'numpy':
            import numpy as np
        elif lib_name == 'requests':
            import requests
        elif lib_name == 'paho.mqtt':
            import paho.mqtt.client as mqtt_client
    except ImportError as e:
        print(f"[FATAL] {lib_name} not available: {e}")
        print("[INFO] Ensure all deps installed: apt install python3-opencv python3-numpy python3-pip")
        print("       pip3 install requests paho-mqtt")
        sys.exit(1)

# === CONFIGURATION ===
SERVER_AI_URL = "http://127.0.0.1:5055/predict/image"
SERVER_DB_URL = "http://127.0.0.1:8081/api/detect/image"
MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC_STATUS = "smartpepper/camera_status"

CAMERA_SOURCE_TYPE = "local"          # local | url | snapshot | rtsp
CAMERA_URL = ""                       # e.g. http://192.168.x.x:8080/stream
CAMERA_SNAPSHOT = ""                  # e.g. http://192.168.x.x:8080/snap.jpg
RTSP_URL = ""                         # e.g. rtsp://192.168.x.x:8554/livestream
LOCAL_CAM_PATH = "/dev/video0"        # USB webcam device

INTERVAL_SECONDS = 15                 # Frame capture interval
MIN_CONFIDENCE = 0.45
OUTPUT_DIR = "/opt/smartpepper-laravel/public/storage/detections/live/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

running = True

def handle_signal(signum, frame):
    global running
    print("\n[SHUTDOWN] Signal received, stopping gracefully...")
    running = False

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


class StreamCapture:
    """Handles different camera input sources with robust reconnection."""

    def __init__(self):
        self.source_type = CAMERA_SOURCE_TYPE
        self.frame_cache = None

    def get_frame(self):
        """Returns BGR numpy array or None on failure."""
        try:
            if self.source_type == "local":
                return self._capture_local()
            elif self.source_type == "url":
                return self._capture_http_url()
            elif self.source_type == "snapshot":
                return self._capture_snapshot()
            elif self.source_type == "rtsp":
                return self._capture_rtsp()
            else:
                raise ValueError(f"Unknown source: {self.source_type}")
        except Exception as e:
            print(f"[ERR Frame] {e}")
            traceback.print_exc()
            return None

    def _capture_local(self):
        cap = cv2.VideoCapture(LOCAL_CAM_PATH)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def _capture_http_url(self):
        resp = requests.get(CAMERA_URL, timeout=8, stream=True)
        chunks = []
        found_start = False
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk:
                if b'\xff\xd8' in chunk:
                    found_start = True
                if found_start:
                    chunks.append(chunk)
                if b'\xff\xd9' in chunk and found_start:
                    break
        if not chunks:
            return None
        img_data = b''.join(chunks)
        arr = np.frombuffer(img_data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _capture_snapshot(self):
        resp = requests.get(CAMERA_SNAPSHOT, timeout=8)
        if resp.status_code != 200:
            return None
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _capture_rtsp(self):
        cap = cv2.VideoCapture(RTSP_URL)
        retries = 3
        for r in range(retries):
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                return frame
            time.sleep(1)
        cap.release()
        return None

    def validate_and_warmup(self):
        """Test connection before starting continuous loop."""
        print(f"\n[TEST] Validasi sumber kamera ({self.source_type})...")
        frame = self.get_frame()
        if frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            print(f"[OK] Kamera aktif: {w}x{h}")
            self.frame_cache = frame
            return True
        else:
            print("[WARN] Frame kosong atau gagal ambil. Coba lagi...")
            time.sleep(2)
            frame = self.get_frame()
            if frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                print(f"[OK] Kamera aktif setelah retry: {w}x{h}")
                self.frame_cache = frame
                return True
        return False


class AIInferenceEngine:
    """Interfaces with SmartPepper AI microservice at port 5055."""

    def __init__(self):
        self.ai_url = SERVER_AI_URL
        self.model_ready = False
        self._warmup()

    def _warmup(self):
        """Check AI service health and warm up model cache."""
        try:
            resp = requests.get("http://127.0.0.1:5055/health", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "UNKNOWN")
                models = data.get("models_loaded", {})
                print(f"[AI OK] Status: {status} | Models: fruit={models.get('fruit')}, leaf={models.get('leaf_base')}, custom={models.get('custom')}")
                # Warmup: send a dummy request
                if self.frame_cache is not None:
                    tmp_path = f"/tmp/_ai_warmup_{int(time.time())}.jpg"
                    cv2.imwrite(tmp_path, self.frame_cache)
                    with open(tmp_path, 'rb') as f:
                        files = {'image': ('warmup.jpg', f, 'image/jpeg')}
                        requests.post(self.ai_url, files=files, data={'conf_thresh': 0.45, 'target': 'all'}, timeout=30)
                    os.remove(tmp_path)
                    print("[AI] Model warmed up.")
                    self.model_ready = True
            else:
                print(f"[AI WARN] Health check returned HTTP {resp.status_code}")
        except Exception as e:
            print(f"[AI ERR] Could not reach AI engine: {e}")

    def detect(self, frame_bgr):
        """Send frame to AI engine, return parsed JSON response."""
        tmp_path = f"/tmp/_cam_detect_{int(time.time())}_{os.getpid()}.jpg"
        try:
            cv2.imwrite(tmp_path, frame_bgr)
            with open(tmp_path, 'rb') as f:
                files = {'image': ('detect.jpg', f, 'image/jpeg')}
                data = {'conf_thresh': MIN_CONFIDENCE, 'target': 'all'}
                resp = requests.post(self.ai_url, files=files, data=data, timeout=30)
            
            if resp.status_code != 200:
                print(f"[AI ERR] HTTP {resp.status_code}: {resp.text[:200]}")
                return None
            
            result = resp.json()
            return result
        except Exception as e:
            print(f"[AI DETECT ERR] {e}")
            return None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class DataStorage:
    """Saves frames, annotations, and records to Laravel backend + SQLite."""

    def __init__(self):
        self.output_dir = OUTPUT_DIR
        self.db_endpoint = SERVER_DB_URL

    def save_original(self, frame_bgr, prefix="live"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid_suffix = str(os.getpid())[-4:]
        filename = f"{prefix}_{ts}_{pid_suffix}_orig.jpg"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return filepath

    def save_annotated(self, base64_str, prefix="annotated"):
        """Decode base64 annotation image from AI engine response."""
        clean = base64_str.replace('data:image/jpeg;base64,', '')
        if not clean:
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{ts}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(base64.b64decode(clean))
        return filepath

    def record_to_laravel(self, filepath, detection_result=None):
        """Submit to Laravel API endpoint to persist into SQLite database."""
        try:
            with open(filepath, 'rb') as f:
                files = {'image': (os.path.basename(filepath), f, 'image/jpeg')}
                data = {}
                if detection_result:
                    overall = detection_result.get('overall_condition', '')
                    count = detection_result.get('count', 0)
                    data['notes'] = json.dumps({
                        'overall': overall,
                        'detection_count': count,
                        'timestamp': datetime.now().isoformat()
                    })
                resp = requests.post(self.db_endpoint, files=files, data=data, timeout=30)
                if resp.status_code in (200, 201, 302):
                    print(f"[DB OK] Recorded to Laravel DB (HTTP {resp.status_code})")
                    return True
                else:
                    print(f"[DB ERR] Laravel returned HTTP {resp.status_code}: {resp.text[:100]}")
                    return False
        except Exception as e:
            print(f"[DB RECORD ERR] {e}")
            traceback.print_exc()
            return False


class MQTTReporter:
    """Reports detection results via MQTT to central broker."""

    def __init__(self):
        self.client = mqtt_client.Client(client_id=f"camera_ingestion_{os.getpid()}")
        self.connected = False
        
        def on_connect(cl, obj, flags, rc):
            self.connected = True
            print("[MQTT OK] Connected to broker")
        
        def on_disconnect(cl, obj, rc):
            self.connected = False
            print(f"[MQTT DISconnected rc={rc}]")
        
        self.client.on_connect = on_connect
        self.client.on_disconnect = on_disconnect
        
        try:
            self.client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            self.client.loop_start()
            time.sleep(1)
            self.client.publish("smartpepper/status", json.dumps({"service": "camera_ingestion", "status": "online"}), qos=1)
            print(f"[MQTT OK] Publisher active -> {MQTT_BROKER}:{MQTT_PORT}")
        except Exception as e:
            print(f"[MQTT WARN] Could not connect: {e}")
            traceback.print_exc()
            self.client = None

    def report_detection(self, result):
        """Publish detection summary to MQTT topic."""
        if not self.client:
            return
        payload = {
            "service": "camera_ingestion",
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
        try:
            self.client.publish(MQTT_TOPIC_STATUS, json.dumps(payload), qos=1)
        except Exception as e:
            print(f"[MQTT PUB ERR] {e}")


# === MAIN ENGINE ===

def run_pipeline():
    print("=" * 60)
    print("  SMARTPEPPER CAMERA INGESTION SERVICE")
    print(f"  Source: {CAMERA_SOURCE_TYPE} | Interval: {INTERVAL_SECONDS}s")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  AI Engine: {SERVER_AI_URL}")
    print("=" * 60)

    cam = StreamCapture()
    
    # Validate camera
    if not cam.validate_and_warmup():
        print("[FATAL] Kamera tidak terdeteksi atau tidak bisa diakses.")
        print("[INFO] Script akan menjalankan auto-reconnect setiap 10 detik...")
        while running:
            time.sleep(10)
            if cam.validate_and_warmup():
                print("[AUTO-RECONNECT] Kamera sekarang tersedia!")
                break
    
    ai = AIInferenceEngine()
    storage = DataStorage()
    mqtt = MQTTReporter()

    cycle_count = 0

    while running:
        cycle_count += 1
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] ── Cycle #{cycle_count} ──")

        # 1. Capture frame
        frame = cam.get_frame()
        if frame is None or frame.size == 0:
            print(f"[{ts}] ❌ Frame kosong/gagal diambil")
            time.sleep(INTERVAL_SECONDS)
            continue

        h, w = frame.shape[:2]
        print(f"[{ts}] 📸 Frame captured: {w}x{h}")

        # 2. Save original
        orig_path = storage.save_original(frame)
        print(f"[{ts}] 💾 Original saved: {orig_path}")

        # 3. AI Detection
        print(f"[{ts}] 🔬 Mengirim ke AI Engine...")
        det_result = ai.detect(frame)

        annotated_path = None
        if det_result and det_result.get('status') == 'SUCCESS':
            conditions = [d.get('condition', '?') for d in det_result.get('detections', [])]
            overall = det_result.get('overall_condition', '?')
            print(f"[{ts}] ✅ Result: {overall} ({len(conditions)} objects)")
            for c in conditions:
                print(f"   → {c}")

            # Save annotated image
            ann_b64 = det_result.get('annotated_image', '')
            if ann_b64:
                annotated_path = storage.save_annotated(ann_b64)
                if annotated_path:
                    print(f"[{ts}] 🎨 Annotated saved: {annotated_path}")

        elif det_result and det_result.get('status') == 'REJECTED_NON_PLANT':
            ratio = det_result.get('plant_ratio', 0)
            print(f"[{ts}] ⚠️ Bukan tanaman! Plant ratio: {ratio}%")
        else:
            print(f"[{ts}] ❌ Tidak ada hasil valid dari AI Engine")

        # 4. Record to Laravel Database
        print(f"[{ts}] 💽 Menyimpan ke database...")
        storage.record_to_laravel(orig_path, det_result)

        # 5. Report via MQTT
        if det_result and det_result.get('status') == 'SUCCESS':
            mqtt.report_detection(det_result)

        # Summary
        print(f"[{ts}] ✓ Cycle complete. Waiting {INTERVAL_SECONDS}s...")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    run_pipeline()
