import os
import io
import time
import base64
import threading
import glob
import shutil
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, jsonify, Response
from ultralytics import YOLO
from image_preprocessor import enhance_pepper_image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

CUSTOM_MODEL_PATH = '/opt/smartpepper-yolo/models/best.pt'
BASE_MODEL_PATH = '/opt/smartpepper-yolo/models/plant_disease_base.pt'
FRUIT_MODEL_PATH = '/opt/smartpepper-yolo/pepper_fruit_model.pt'
COCO_MODEL_PATH = '/opt/smartpepper-yolo/yolov8n.pt'

custom_model = YOLO(CUSTOM_MODEL_PATH) if os.path.exists(CUSTOM_MODEL_PATH) else None
base_model = YOLO(BASE_MODEL_PATH) if os.path.exists(BASE_MODEL_PATH) else None
fruit_model = YOLO(FRUIT_MODEL_PATH) if os.path.exists(FRUIT_MODEL_PATH) else None
coco_model = YOLO(COCO_MODEL_PATH) if os.path.exists(COCO_MODEL_PATH) else None

NON_PLANT_DICT = {
    'person': 'Manusia / Wajah / Tangan',
    'cell phone': 'Handphone / Gadget',
    'laptop': 'Layar Laptop / Monitor',
    'tv': 'Televisi / Monitor',
    'keyboard': 'Keyboard Komputer',
    'mouse': 'Mouse Komputer',
    'bottle': 'Botol Minum / Wadah',
    'cup': 'Cangkir / Gelas',
    'chair': 'Kursi / Mebel',
    'couch': 'Sofa / Tempat Duduk',
    'bed': 'Tempat Tidur',
    'dining table': 'Meja Makan / Perabot',
    'book': 'Buku / Kertas',
    'clock': 'Jam Dinding / Tangan',
    'scissors': 'Gunting / Logam',
    'backpack': 'Tas Ransel',
    'umbrella': 'Payung',
    'handbag': 'Tas Tangan',
    'tie': 'Dasi / Pakaian',
    'suitcase': 'Koper',
    'car': 'Mobil / Kendaraan',
    'motorcycle': 'Sepeda Motor',
    'bicycle': 'Sepeda',
    'cat': 'Kucing / Hewan',
    'dog': 'Anjing / Hewan'
}

DISEASE_KB = {
    "Pepper__bell___Bacterial_spot": {
        "name": "Bercak Bakteri Daun Lada",
        "condition": "TIDAK SEHAT",
        "color": "#f43f5e",
        "pathogen": "Xanthomonas campestris pv. vesicatoria",
        "severity": "Sedang - Tinggi",
        "description": "Bercak melingkar kecokelatan kehitaman pada permukaan daun yang memicu nekrosis dan gugur dini.",
        "pengendalian_hayati": "Aplikasi bio-fungisida Trichoderma sp. atau Pseudomonas fluorescens pada sore hari.",
        "pengendalian_kimiawi": "Fungisida tembaga hidroksida terukur dengan perekat perata.",
        "teknik_budidaya": "Sanitasi daun terinfeksi dan kurangi kelembapan kanopi mikro.",
        "actuator": {"fan": "HIGH (Airflow 100%)", "pump": "REDUCE", "growlight": "ON"}
    },
    "Pepper__bell___healthy": {
        "name": "Daun Lada Sehat",
        "condition": "SEHAT",
        "color": "#10b981",
        "pathogen": "Aman",
        "severity": "Optimal",
        "description": "Pigmentasi klorofil merata, lamina daun kokoh tanpa nekrosis.",
        "pengendalian_hayati": "Pertahankan pupuk hayati mikoriza dan bokashi.",
        "pengendalian_kimiawi": "Pemupukan makro-mikro berimbang sesuai fase vegetatif.",
        "teknik_budidaya": "Drainase gembur dan monitoring EC teratur.",
        "actuator": {"fan": "AUTO", "pump": "AUTO", "growlight": "AUTO"}
    },
    "lada_sehat": {
        "name": "Daun Lada Sehat",
        "condition": "SEHAT",
        "color": "#10b981",
        "pathogen": "Aman",
        "severity": "Optimal",
        "description": "Lamina daun hijau segar, tanpa bercak atau lesi.",
        "pengendalian_hayati": "Pupuk organik cair rutin.",
        "pengendalian_kimiawi": "Nutrisi kalsium-magnesium seimbang.",
        "teknik_budidaya": "Ventilasi sirkulasi udara stabil.",
        "actuator": {"fan": "AUTO", "pump": "AUTO", "growlight": "AUTO"}
    },
    "lada_tidak_sehat": {
        "name": "Daun Lada Bergejala Penyakit",
        "condition": "TIDAK SEHAT",
        "color": "#f43f5e",
        "pathogen": "Patogen Daun / Jamur / Bakteri",
        "severity": "Perlu Penanganan",
        "description": "Terlihat lesi klorosis atau nekrosis aktif pada jaringan daun.",
        "pengendalian_hayati": "Semprot larutan Trichoderma.",
        "pengendalian_kimiawi": "Fungisida sistemik spektrum luas sesuai dosis.",
        "teknik_budidaya": "Pangkas helai sakit untuk cegah spora menyebar.",
        "actuator": {"fan": "HIGH (100%)", "pump": "REDUCE", "growlight": "ON"}
    },
    "lada_buah_sehat": {
        "name": "Buah Lada Sehat",
        "condition": "SEHAT",
        "color": "#10b981",
        "pathogen": "Aman (Siap Panen)",
        "severity": "Prima",
        "description": "Butiran buah bulat padat, kulit mulus mengkilap, perikarp bersih.",
        "pengendalian_hayati": "Semprot pupuk hayati perangsang buah.",
        "pengendalian_kimiawi": "Kalsium nitrat penguat perikarp buah.",
        "teknik_budidaya": "Intensitas cahaya fotosintesis cukup.",
        "actuator": {"fan": "AUTO", "pump": "AUTO", "growlight": "AUTO"}
    },
    "lada_buah_tidak_sehat": {
        "name": "Buah Lada Busuk / Antraknosa",
        "condition": "TIDAK SEHAT",
        "color": "#f43f5e",
        "pathogen": "Colletotrichum gloeosporioides",
        "severity": "Gawat",
        "description": "Buah lada berbercak hitam melekuk, keriput, tangkai busuk rontok.",
        "pengendalian_hayati": "Aplikasi bio-fungisida pada dompolan buah.",
        "pengendalian_kimiawi": "Semprot bahan aktif difenokonazol atau azoksistrobin.",
        "teknik_budidaya": "Petik dan musnahkan buah terinfeksi.",
        "actuator": {"fan": "HIGH (Airflow 100%)", "pump": "REDUCE", "growlight": "ON"}
    }
}

TRAIN_STATE = {
    "status": "IDLE",
    "epoch": 0,
    "total_epochs": 15,
    "progress": 0,
    "message": "Siap melatih dataset."
}

def check_is_non_plant(img_np, conf_thresh=0.30):
    h, w, _ = img_np.shape
    detected = []
    if coco_model is not None:
        res = coco_model.predict(img_np, conf=conf_thresh, verbose=False)[0]
        for box in res.boxes:
            c_name = coco_model.names.get(int(box.cls[0]), '')
            if c_name in NON_PLANT_DICT:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                detected.append({
                    'class_raw': c_name,
                    'name_id': NON_PLANT_DICT[c_name],
                    'confidence': round(conf * 100, 1),
                    'bbox': [round(x, 1) for x in xyxy],
                    'norm_bbox': [round(xyxy[0]/w, 3), round(xyxy[1]/h, 3), round(xyxy[2]/w, 3), round(xyxy[3]/h, 3)]
                })

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    mask_g = cv2.inRange(hsv, (25, 25, 25), (90, 255, 255))
    mask_yb = cv2.inRange(hsv, (10, 35, 25), (25, 255, 255))
    mask_r1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    mask_r2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    plant_mask = mask_g | mask_yb | mask_r1 | mask_r2
    ratio = round((cv2.countNonZero(plant_mask) / (h * w)) * 100, 1)
    return detected, ratio

def perform_ensemble_detection(img_np, conf_thresh=0.35, target='all'):
    h, w, _ = img_np.shape
    detections = []

    if target in ('all', 'fruit') and fruit_model is not None:
        res = fruit_model.predict(img_np, conf=conf_thresh, verbose=False)[0]
        for box in res.boxes:
            cls_id = int(box.cls[0])
            cls_name = fruit_model.names.get(cls_id, 'lada_buah_sehat')
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            detections.append({
                'source': 'yolo_fruit_model',
                'organ': 'BUAH',
                'class_raw': cls_name,
                'confidence': round(conf * 100, 1),
                'bbox': [round(x, 1) for x in xyxy],
                'norm_bbox': [round(xyxy[0]/w, 3), round(xyxy[1]/h, 3), round(xyxy[2]/w, 3), round(xyxy[3]/h, 3)],
                'condition': 'SEHAT' if 'sehat' in cls_name and 'tidak' not in cls_name else 'TIDAK SEHAT'
            })

    if target in ('all', 'leaf') and base_model is not None:
        res = base_model.predict(img_np, conf=conf_thresh, verbose=False)[0]
        for box in res.boxes:
            cls_id = int(box.cls[0])
            cls_name = base_model.names.get(cls_id, '')
            if 'pepper' in cls_name.lower() or 'healthy' in cls_name.lower():
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                cond = 'SEHAT' if 'healthy' in cls_name.lower() else 'TIDAK SEHAT'
                detections.append({
                    'source': 'plant_pathology_base',
                    'organ': 'DAUN',
                    'class_raw': 'lada_sehat' if cond == 'SEHAT' else 'lada_tidak_sehat',
                    'confidence': round(conf * 100, 1),
                    'bbox': [round(x, 1) for x in xyxy],
                    'norm_bbox': [round(xyxy[0]/w, 3), round(xyxy[1]/h, 3), round(xyxy[2]/w, 3), round(xyxy[3]/h, 3)],
                    'condition': cond
                })

    return detections

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'OK',
        'service': 'SmartPepper YOLOv8 AI Inference Engine',
        'device': 'CPU',
        'models_loaded': {
            'fruit': fruit_model is not None,
            'leaf_base': base_model is not None,
            'custom': custom_model is not None
        }
    })

@app.route('/predict/image', methods=['POST'])
def predict_image():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'Image file required'}), 400

    conf_thresh = float(request.form.get('conf_thresh', 0.35))
    target = request.form.get('target', 'all')

    raw = file.read()
    img_pil = Image.open(io.BytesIO(raw)).convert('RGB')
    img_np = np.array(img_pil)
    h, w, _ = img_np.shape

    # Jalankan Pipeline Pengolahan Citra Digital (Preprocessing & Contrast Enhancement)
    try:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        enhanced_bgr = enhance_pepper_image(img_bgr)
        processed_np = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        processed_np = img_np

    non_plants, plant_ratio = check_is_non_plant(img_np, conf_thresh=0.30)
    if non_plants and plant_ratio < 12.0:
        annotated = img_np.copy()
        for o in non_plants:
            b = [int(v) for v in o['bbox']]
            cv2.rectangle(annotated, (b[0], b[1]), (b[2], b[3]), (234, 179, 8), 2)
            lbl = f"{o['name_id']} {o['confidence']}%"
            cv2.putText(annotated, lbl, (b[0], max(20, b[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (234, 179, 8), 2)

        pil_ann = Image.fromarray(annotated)
        buf = io.BytesIO()
        pil_ann.save(buf, format='JPEG', quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return jsonify({
            'status': 'REJECTED_NON_PLANT',
            'overall_condition': 'BUKAN TANAMAN',
            'plant_ratio': plant_ratio,
            'non_plant_objects': non_plants,
            'detections': [{
                'organ': 'BUKAN DAUN / BUAH',
                'class_raw': 'bukan_tanaman',
                'condition': 'BUKAN DAUN',
                'confidence': non_plants[0]['confidence'],
                'bbox': non_plants[0]['bbox'],
                'norm_bbox': non_plants[0]['norm_bbox'],
                'info': {
                    'name': 'Objek Terdeteksi Bukan Daun Lada',
                    'condition': 'BUKAN DAUN',
                    'color': '#eab308',
                    'pathogen': 'Bukan Tanaman',
                    'severity': 'Salah Sampel',
                    'description': f"Terdeteksi {non_plants[0]['name_id']} (Kandungan pigmen tanaman hanya {plant_ratio}%).",
                    'pengendalian_hayati': 'Harap arahkan kamera langsung ke helai daun atau butiran buah lada.',
                    'pengendalian_kimiawi': 'Tidak ada tindakan kimiawi yang diperlukan.',
                    'teknik_budidaya': 'Pastikan daun atau buah lada berada di jarak 15-30 cm agar kamera fokus.'
                }
            }],
            'annotated_image': f'data:image/jpeg;base64,{b64}'
        })

    detections = perform_ensemble_detection(processed_np, conf_thresh, target)
    annotated = img_np.copy()
    for d in detections:
        box = [int(v) for v in d['bbox']]
        is_healthy = d['condition'] == 'SEHAT'
        color = (16, 185, 129) if is_healthy else (244, 63, 94)
        cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, max(2, int(w/250)))
        organ_tag = f"{d.get('organ', '')} " if d.get('organ') else ""
        label = f"{organ_tag}{d['condition']} {d['confidence']}%"
        cv2.putText(annotated, label, (box[0], max(20, box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, label, (box[0], max(20, box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        fallback_key = ('lada_buah_sehat' if is_healthy else 'lada_buah_tidak_sehat') if d.get('organ') == 'BUAH' else ('lada_sehat' if is_healthy else 'lada_tidak_sehat')
        d['info'] = DISEASE_KB.get(d['class_raw'], DISEASE_KB[fallback_key])

    pil_ann = Image.fromarray(annotated)
    buf = io.BytesIO()
    pil_ann.save(buf, format='JPEG', quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    if len(detections) == 0:
        overall = "TIDAK_ADA_SAMPEL"
    elif all(d['condition'] == 'SEHAT' for d in detections):
        overall = "SEHAT"
    else:
        overall = "TIDAK SEHAT"

    return jsonify({
        'status': 'SUCCESS',
        'overall_condition': overall,
        'count': len(detections),
        'detections': detections,
        'annotated_image': f'data:image/jpeg;base64,{b64}'
    })

def generate_stream():
    while True:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (20, 24, 28)
        cv2.rectangle(frame, (100, 220), (280, 420), (35, 45, 55), -1)
        cv2.rectangle(frame, (360, 220), (540, 420), (35, 45, 55), -1)
        cv2.rectangle(frame, (100, 220), (280, 420), (0, 200, 120), 2)
        cv2.rectangle(frame, (360, 220), (540, 420), (0, 200, 120), 2)
        cv2.circle(frame, (190, 300), 65, (40, 140, 50), -1)
        cv2.circle(frame, (450, 300), 65, (40, 140, 50), -1)
        cv2.putText(frame, "PEPPER POT #1", (125, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "PEPPER POT #2", (385, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "SMARTPEPPER 4.0 - TOP CANOPY CV CAMERA", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
        cv2.putText(frame, f"FPS: 25.0 | {time.strftime('%Y-%m-%d %H:%M:%S')}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        ret, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.04)

@app.route('/stream/mjpg')
def stream_mjpg():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/train/status', methods=['GET'])
def train_status():
    return jsonify(TRAIN_STATE)

@app.route('/train/start', methods=['POST'])
def train_start():
    data = request.json or {}
    epochs = int(data.get('epochs', 15))
    TRAIN_STATE["status"] = "IDLE"
    return jsonify({'status': 'SUCCESS', 'message': f'Worker siap melatih {epochs} epochs.'})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5055)
