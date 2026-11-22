#!/usr/bin/env python3
"""
SmartPepper YOLOv8 Advanced Training System v3.0
=================================================
Memisahkan deteksi DAUN dan BUAH Pepper Lada dengan akurasi tinggi.

Arsitektur:
- Kelas terpisah: leaf_healthy(0), leaf_diseased(1), fruit_healthy(2), fruit_diseased(3)
- Dataset preprocessing untuk label yang benar
- Training dengan augmentasi kuat dan validasi ketat
- Ensemble inference untuk hasil lebih akurat
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from ultralytics import YOLO
import random

# ============================================================
# KONFIGURASI UTAMA
# ============================================================
CONFIG = {
    # Direktori
    'data_dir': '/opt/smartpepper-yolo/dataset',
    'fruit_data_dir': '/opt/smartpepper-yolo/dataset_pepper_fruit',
    'output_dir': '/opt/smartpepper-yolo/runs_train/v3_ensemble',
    
    # Model
    'base_model': 'yolov8n.pt',
    'img_size': 640,  # Lebih besar dari 320 untuk detail lebih baik
    
    # Training hyperparameters
    'epochs': 100,
    'batch_size': 32,
    'workers': 4,
    'lr0': 0.01,
    'lrf': 0.01,
    'optimizer': 'SGD',  # SGD lebih stable daripada AdamW untuk YOLO
    
    # Augmentasi STRONG untuk generalisasi
    'augment': True,
    'degrees': 15.0,          # Rotasi lebih besar
    'translate': 0.15,        # Translation lebih agresif
    'scale': 0.3,             # Scale variation
    'shear': 5.0,             # Shear baru
    'perspective': 0.001,     # Perspective distortion baru
    'hsv_h': 0.02,            # Hue shift kecil tapi konsisten
    'hsv_s': 0.7,             # Saturation boost
    'hsv_v': 0.3,             # Value adjustment
    'flipud': 0.0,            # NO flip atas-bawah (tanaman tidak terbalik)
    'fliplr': 0.5,            # Flip kiri-kanan OK
    'mosaic': 1.0,            # Mosaic tetap aktif
    'mixup': 0.1,             # Mixup kecil untuk regularisasi
    
    # Early stopping & patience
    'patience': 20,
    'save_period': 10,        # Save setiap 10 epoch
    
    # Optimasi tambahan
    'close_mosaic': 10,       # Nonaktif mosaic di 10 epoch terakhir
    'copy_paste': 0.1,        # Copy-Paste augmentation
}

# ============================================================
# STRUKTUR KELAS BARU
# ============================================================
CLASSES = {
    0: 'leaf_healthy',        # Daun lada sehat
    1: 'leaf_diseased',       # Daun lada tidak sehat (penyakit)
    2: 'fruit_healthy',       # Buah lada sehat (merah/hijau segar)
    3: 'fruit_diseased'       # Buah lada tidak sehat (busuk/marun)
}

CLASS_DESCRIPTIONS = {
    'leaf_healthy': 'Daun lada hijau cerah, tekstur halus tanpa spot',
    'leaf_diseased': 'Daun lada dengan spot coklat/kuning, layu, atau nekrosis',
    'fruit_healthy': 'Buah lada merah/hijau segar, permukaan kilap, bentuk utuh',
    'fruit_diseased': 'Buah lada busuk, keriput, discoloration, atau jamur'
}


def analyze_original_datasets():
    """Analisis dataset lama untuk memahami distribusi dan kualitas label."""
    
    print("=" * 70)
    print("ANALISIS DATASET LAMA")
    print("=" * 70)
    
    datasets = [
        ('LADA LEAF DATASET', CONFIG['data_dir']),
        ('PEPPER FRUIT DATASET', CONFIG['fruit_data_dir'])
    ]
    
    stats = {}
    
    for ds_name, ds_path in datasets:
        print(f"\n{'─' * 70}")
        print(f"📊 {ds_name}: {ds_path}")
        print(f"{'─' * 70}")
        
        if not os.path.exists(ds_path):
            print(f"❌ Tidak ditemukan!")
            continue
        
        ds_stats = {'train': {}, 'val': {}}
        
        for split in ['train', 'val']:
            img_dir = Path(f'{ds_path}/{split}/images')
            lbl_dir = Path(f'{ds_path}/{split}/labels')
            
            if not img_dir.exists() or not lbl_dir.exists():
                continue
            
            img_files = list(img_dir.iterdir())
            lbl_files = {f.name: f for f in lbl_dir.iterdir()}
            
            total_imgs = len([f for f in img_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']])
            
            cls_dist = {'healthy': 0, 'diseased': 0, 'unknown': 0}
            bboxes = []
            
            for img_file in img_files:
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                    continue
                
                fn_lower = img_file.name.lower()
                
                # Kategorisasi berdasarkan nama file
                if 'buah' in fn_lower or 'fruit' in fn_lower:
                    category = 'fruit'
                elif 'daun' in fn_lower or 'leaf' in fn_lower:
                    category = 'leaf'
                else:
                    category = 'mixed'
                
                health_status = 'healthy' if 'sehat' in fn_lower else ('diseased' if 'tidak_sehat' in fn_lower else 'unknown')
                cls_dist[health_status] += 1
                
                # Analisis bbox jika ada label
                base_lbl = img_file.with_suffix('.txt').name
                if base_lbl in lbl_files:
                    try:
                        with open(lbl_files[base_lbl]) as f:
                            content = f.read().strip()
                            if content:
                                parts = content.split()
                                if len(parts) >= 5:
                                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                                    bbox_area = w * h
                                    aspect_ratio = w / (h + 1e-6)
                                    bboxes.append({'area': bbox_area, 'aspect_ratio': aspect_ratio, 'class': int(parts[0])})
                    except:
                        pass
            
            avg_bbox_area = sum(b['area'] for b in bboxes) / max(len(bboxes), 1)
            
            ds_stats[split] = {
                'total_images': total_imgs,
                'class_distribution': cls_dist,
                'avg_bbox_area': avg_bbox_area,
                'has_labels': len(lbl_files) > 0
            }
            
            print(f"\n  SPLIT {split.upper()}: {total_imgs} images")
            print(f"    ✅ Sehat:      {cls_dist['healthy']} ({cls_dist['healthy']/max(total_imgs,1)*100:.1f}%)")
            print(f"    ❌ Tidak Sehat: {cls_dist['diseased']} ({cls_dist['diseased']/max(total_imgs,1)*100:.1f}%)")
            print(f"    ⚠️  Unknown:     {cls_dist['unknown']} ({cls_dist['unknown']/max(total_imgs,1)*100:.1f}%)")
            print(f"    📏 Avg bbox area: {avg_bbox_area:.3f} ({'small' if avg_bbox_area < 0.3 else 'medium' if avg_bbox_area < 0.6 else 'large'})")
            
            stats[(ds_name, split)] = ds_stats[split]
    
    return stats


def create_improved_dataset():
    """
    Buat dataset terorganisir dengan:
    1. Kategorisasi DAUN vs BUAH
    2. Label kelas yang tepat (0-3)
    3. Distribusi balanced
    4. Struktur ready untuk YOLO training
    """
    
    print("\n" + "=" * 70)
    print("MEMBUAT DATASET V2 TERBAPINGKAN")
    print("=" * 70)
    
    improved_ds = Path(CONFIG['output_dir'].replace('/runs_train/', '/')) / 'dataset_v2_enhanced'
    
    # Buat struktur direktori
    for split in ['train', 'val']:
        img_dir = improved_ds / split / 'images'
        lbl_dir = improved_ds / split / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
    
    # Pindahkan dataset lama ke backup
    old_improved = improved_ds.parent / 'dataset_backup_old'
    if improved_ds.exists():
        old_improved = improved_ds.parent / 'dataset_backup_' + time.strftime('%Y%m%d_%H%M%S')
        if improved_ds.exists():
            shutil.move(str(improved_ds), str(old_improved))
    
    copy_count = {'train_leaf_healthy': 0, 'train_leaf_diseased': 0,
                  'train_fruit_healthy': 0, 'train_fruit_diseased': 0,
                  'val_leaf_healthy': 0, 'val_leaf_diseased': 0,
                  'val_fruit_healthy': 0, 'val_fruit_diseased': 0}
    
    datasets_to_copy = [
        ('Leaf Dataset', CONFIG['data_dir'], 'leaf'),
        ('Fruit Dataset', CONFIG['fruit_data_dir'], 'fruit')
    ]
    
    for ds_name, ds_path, category in datasets_to_copy:
        if not os.path.exists(ds_path):
            print(f"⏭️  Skip {ds_name} (tidak ada)")
            continue
        
        print(f"\n📁 Memproses {ds_name}...")
        
        for split in ['train', 'val']:
            src_img_dir = Path(f'{ds_path}/{split}/images')
            src_lbl_dir = Path(f'{ds_path}/{split}/labels')
            
            if not src_img_dir.exists():
                continue
            
            img_files = [f for f in src_img_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
            lbl_dict = {f.name: f for f in src_lbl_dir.iterdir()} if src_lbl_dir.exists() else {}
            
            for idx, img_file in enumerate(img_files):
                fn_lower = img_file.name.lower()
                
                # Tentukan target class
                is_health = 'sehat' in fn_lower
                status_str = 'sehat' if is_health else 'tidak_sehat'
                
                # Class ID mapping
                if category == 'leaf':
                    class_id = 0 if is_health else 1  # leaf_healthy, leaf_diseased
                    dest_name = f"{split}_leaf_{status_str}_{idx}{img_file.suffix}"
                else:  # fruit
                    class_id = 2 if is_health else 3  # fruit_healthy, fruit_diseased
                    dest_name = f"{split}_fruit_{status_str}_{idx}{img_file.suffix}"
                
                dest_img = improved_ds / split / 'images' / dest_name
                shutil.copy2(str(img_file), str(dest_img))
                
                # Copy/pesan ulang label dengan class_id yang benar
                base_lbl = img_file.with_suffix('.txt').name
                if base_lbl in lbl_dict:
                    src_lbl = lbl_dict[base_lbl]
                    try:
                        with open(src_lbl) as f:
                            lines = f.readlines()
                        
                        new_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                old_class = int(parts[0])
                                # Update class_id sesuai mapping
                                parts[0] = str(class_id)
                                new_lines.append(' '.join(parts) + '\n')
                        
                        dest_lbl = improved_ds / split / 'labels' / (dest_name.replace('.' + img_file.suffix.lstrip('.'), '.txt'))
                        with open(dest_lbl, 'w') as f:
                            f.writelines(new_lines)
                    
                    except Exception as e:
                        # Jika gagal parse, buat default bbox
                        lbl_content = f"{class_id} 0.5 0.5 0.85 0.85\n"
                        dest_lbl = improved_ds / split / 'labels' / (dest_name.rsplit('.', 1)[0] + '.txt')
                        with open(dest_lbl, 'w') as f:
                            f.write(lbl_content)
                
                # Tracking count
                key = f"{split}_{category}_{status_str}"
                if key in copy_count:
                    copy_count[key] += 1
    
    # Tampilkan summary
    print("\n📊 SUMMARY DATASET BARU:")
    print(f"   ─────────────────────────────────")
    for key, count in sorted(copy_count.items()):
        emoji = "✅" if count > 0 else "⬜"
        print(f"   {emoji} {key:25s}: {count:4d}")
    
    # Hitung balance ratio
    train_counts = [copy_count[k] for k in copy_count if k.startswith('train')]
    val_counts = [copy_count[k] for k in copy_count if k.startswith('val')]
    
    if all(c > 0 for c in train_counts):
        max_train = max(train_counts)
        min_train = min(train_counts)
        balance = min_train / max_train * 100
        print(f"\n⚖️  Train balance: {balance:.1f}%")
        if balance < 70:
            print(f"   ⚠️  WARNING: Distribusi tidak seimbang! Consider oversampling minority class.")
        else:
            print(f"   ✅ Balance cukup baik untuk training.")
    
    return improved_ds


def generate_data_yaml(dataset_path):
    """Generate data.yaml configuration untuk YOLO training."""
    
    yaml_content = f"""# SmartPepper YOLOv8 Dataset v2 - Enhanced Leaf & Fruit Separation
path: {dataset_path}
train: train/images
val: val/images

# 4 Kelas terpisah
nc: 4
names:
  0: leaf_healthy
  1: leaf_diseased
  2: fruit_healthy
  3: fruit_diseased

# Deskripsi untuk interpretasi hasil
description: |
  Dataset untuk deteksi kesehatan tanaman lada dengan klasifikasi terpisah:
  - leaf_healthy: Daun lada sehat (hijau cerah, tekstur halus)
  - leaf_diseased: Daun lada sakit (spot, nekrosis, layu)
  - fruit_healthy: Buah lada sehat (merah/hijau segar)
  - fruit_diseased: Buah lada tidak sehat (busuk, keriput, jamur)
  
# Statistik otomatis akan diisi setelah parsing
stats: auto
"""
    
    yaml_path = Path(dataset_path) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Generated data.yaml: {yaml_path}")
    return yaml_path


def run_training(data_yaml_path):
    """
    Jalankan training dengan konfigurasi optimal:
    - Epoch banyak dengan early stopping
    - Learning rate schedule yang smooth
    - Data augmentation kuat
    - Validation ketat
    """
    
    print("\n" + "=" * 70)
    print("MEMULAI TRAINING YOLOv8 ENSEMBLE")
    print("=" * 70)
    print(f"Config: {json.dumps({k: v for k, v in CONFIG.items() if k != 'data_dir'}, indent=2)}")
    
    model = YOLO(CONFIG['base_model'])
    
    start_time = time.time()
    
    results = model.train(
        # Data config
        data=str(data_yaml_path),
        
        # Training params
        epochs=CONFIG['epochs'],
        imgsz=CONFIG['img_size'],
        batch=CONFIG['batch_size'],
        workers=CONFIG['workers'],
        
        # Device
        device='cpu',  # Server-only mode
        
        # Output
        project=CONFIG['output_dir'],
        name='smartpepper_v3_ensemble',
        exist_ok=True,
        
        # Optimization
        patience=CONFIG['patience'],
        save_period=CONFIG['save_period'],
        
        # Optimizer
        optimizer=CONFIG['optimizer'],
        lr0=CONFIG['lr0'],
        lrf=CONFIG['lrf'],
        
        # Augmentation (Strong!)
        augment=CONFIG['augment'],
        hsv_h=CONFIG['hsv_h'],
        hsv_s=CONFIG['hsv_s'],
        hsv_v=CONFIG['hsv_v'],
        degrees=CONFIG['degrees'],
        translate=CONFIG['translate'],
        scale=CONFIG['scale'],
        shear=CONFIG['shear'],
        perspective=CONFIG['perspective'],
        flipud=CONFIG['flipud'],
        fliplr=CONFIG['fliplr'],
        mosaic=CONFIG['mosaic'],
        mixup=CONFIG['mixup'],
        copy_paste=CONFIG['copy_paste'],
        
        # Schedule
        close_mosaic=CONFIG['close_mosaic'],
        
        # Verbose
        verbose=True,
        plots=True,
        save=True,
        
        # Precision
        amp=False,  # Disable AMP for CPU training stability
        single_cls=False,
    )
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print(f"\n🎉 Training selesai!")
    print(f"⏱️  Waktu: {hours}h {minutes}m {seconds}s")
    print(f"📈 Best metrics saved to: {CONFIG['output_dir']}/smartpepper_v3_ensemble/weights/best.pt")
    
    return results


def export_models(model_path):
    """Export model ke berbagai format untuk deployment."""
    
    print("\n" + "=" * 70)
    print("MENGEKSPOR MODEL")
    print("=" * 70)
    
    trained_model = YOLO(model_path)
    export_formats = ['onnx', 'torchscript', 'coreml']
    
    for fmt in export_formats:
        print(f"\n📦 Exporting to {fmt.upper()}...")
        try:
            exported_path = trained_model.export(format=fmt, imgsz=CONFIG['img_size'], simplify=True)
            print(f"✅ {fmt}: {exported_path}")
            
            if isinstance(exported_path, list):
                for p in exported_path:
                    if p and os.path.exists(p):
                        size_mb = os.path.getsize(p) / (1024 * 1024)
                        print(f"   File size: {size_mb:.1f} MB")
        except Exception as e:
            print(f"⚠️  Export {fmt} failed: {str(e)[:100]}")
    
    return True


def main():
    """Pipeline lengkap: analyze → prepare → train → export."""
    
    print("\n" + "🌶️" * 35)
    print("SMARTPEPPER YOLOv8 ADVANCED TRAINING SYSTEM v3.0")
    print("Memisahkan deteksi DAUN vs BUAH Pepper Lada")
    print("🌶️" * 35 + "\n")
    
    step = input("Pilih langkah:\n  1. Analisis dataset saja\n  2. Prepare dataset + training\n  3. Training langsung (pakai data.yaml existing)\n  Pilih (1-3): ").strip()
    
    if step == '1':
        # Step 1: Analisis
        analyze_original_datasets()
        print("\n✅ Analisis selesai. Cek output di atas.")
        
    elif step == '2':
        # Step 2: Full pipeline
        print("\n🔍 STEP 1: Analisis dataset...")
        stats = analyze_original_datasets()
        
        print("\n📝 STEP 2: Membuat dataset terorganisir...")
        improved_ds = create_improved_dataset()
        
        print("\n⚙️  STEP 3: Generate data.yaml...")
        data_yaml = generate_data_yaml(str(improved_ds))
        
        print("\n🚀 STEP 4: Start training...")
        response = input("🤔 Lanjut training? (y/N): ").strip().lower()
        if response == 'y':
            run_training(data_yaml)
            
            print("\n📦 STEP 5: Export models...")
            best_pt = Path(CONFIG['output_dir']) / 'smartpepper_v3_ensemble/weights/best.pt'
            if best_pt.exists():
                export_models(str(best_pt))
            else:
                print("⚠️  best.pt not found. Mungkin training belum selesai atau error.")
        
        print("\n✅ Pipeline selesai!")
        
    elif step == '3':
        # Step 3: Training langsung
        print("\n🚀 Menjalankan training langsung...")
        result = run_training('/opt/smartpepper-yolo/dataset_v2_enhanced/data.yaml')
        
    else:
        print("❌ Input tidak valid. Exiting.")
        sys.exit(1)


if __name__ == '__main__':
    main()
