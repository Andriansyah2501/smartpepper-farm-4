#!/bin/bash
# SmartPepper Camera Service Manager
# Auto-detect camera sources dan start appropriate ingestion mode

LOG="/opt/smartpepper-yolo/camera_service.log"
PIDFILE="/opt/smartpepper-yolo/camera_ingestion.pid"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"; }

check_camera_sources() {
    # Cek webcam lokal
    if [ -c /dev/video0 ]; then
        log "[OK] Webcam lokal terdeteksi: /dev/video0"
        return 0
    fi
    
    # Cek Raspberry Pi streaming
    RPI_IP="192.168.0.30"
    if curl -s --connect-timeout 2 "http://$RPI_IP:8080/stream" > /dev/null 2>&1; then
        log "[OK] Streaming server RPi aktif: http://$RPI_IP:8080/stream"
        sed -i 's/^CAMERA_SOURCE_TYPE=.*/CAMERA_SOURCE_TYPE="url"/' /opt/smartpepper-yolo/camera_ingestion.py
        sed -i 's|^CAMERA_URL=.*|CAMERA_URL="http://$RPI_IP:8080/stream"|' /opt/smartpepper-yolo/camera_ingestion.py
        return 0
    fi
    
    log "[WARN] Tidak ada sumber kamera aktif."
    log "[INFO] Menggunakan mode fallback: uji dengan dataset sample."
    
    # Buat mode demo dengan dataset
    cat >> /opt/smartpepper-yolo/camera_ingestion.py << 'DEMO_MODE'

def demo_with_dataset():
    """Mode demo: cycle through dataset images for testing."""
    import glob
    dataset_dir = "/opt/smartpepper-yolo/dataset/val/images"
    if not os.path.exists(dataset_dir):
        return False
    
    files = sorted(glob.glob(f"{dataset_dir}/*.jpg"))
    if not files:
        files = sorted(glob.glob(f"{dataset_dir}/*.png"))
    
    idx = 0
    while True:
        if idx >= len(files):
            idx = 0
        
        frame = cv2.imread(files[idx])
        if frame is not None:
            break
        idx = (idx + 1) % len(files)
        time.sleep(1)
    
    idx = (idx + 1) % len(files)
    return frame, files[idx], True

demo_mode = True
DEMO_MODE
    
    return 1
}

start_service() {
    log "Memulai SmartPepper Camera Ingestion Service..."
    
    cd /opt/smartpepper-yolo || exit 1
    
    python3 camera_ingestion.py >> "$LOG" 2>&1 &
    PID=$!
    echo $PID > "$PIDFILE"
    
    log "Service started with PID: $PID"
    
    # Monitor service
    while kill -0 $PID 2>/dev/null; do
        sleep 5
        ACTIVE=$(ps aux | grep -c "[p]ython3 camera_ingestion")
        if [ "$ACTIVE" -eq 0 ]; then
            log "[ERROR] Service mati. Restarting..."
            python3 camera_ingestion.py >> "$LOG" 2>&1 &
            PID=$!
            echo $PID > "$PIDFILE"
        fi
    done
}

stop_service() {
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        if kill -0 $PID 2>/dev/null; then
            kill $PID
            log "Service stopped (PID: $PID)"
        fi
        rm -f "$PIDFILE"
    fi
}

case "${1:-start}" in
    start)
        check_camera_sources
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        stop_service
        sleep 2
        start_service
        ;;
    status)
        if [ -f "$PIDFILE" ]; then
            PID=$(cat "$PIDFILE")
            if kill -0 $PID 2>/dev/null; then
                echo "Status: RUNNING (PID: $PID)"
            else
                echo "Status: STOPPED"
            fi
        else
            echo "Status: NOT INSTALLED"
        fi
        tail -n 5 "$LOG" 2>/dev/null || true
        ;;
    log)
        tail -f "$LOG"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|log}"
        ;;
esac
