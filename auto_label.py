#!/usr/bin/env python3
"""
SmartPepper Auto-Label Annotator v1.0
=====================================
Generate bounding boxes otomatis dari gambar dengan:
- Color-based segmentation untuk daun vs buah
- Morphological operations untuk isolasi objek
- COCO format labeling yang presisi
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

class SmartPepperAnnotator:
    """Auto-label images for pepper leaf and fruit detection."""
    
    def __init__(self):
        self.class_names = {
            0: 'leaf_healthy',
            1: 'leaf_diseased', 
            2: 'fruit_healthy',
            3: 'fruit_diseased'
        }
        
    def detect_bboxes(self, img_path):
        """Detect bounding boxes using color segmentation."""
        
        img = cv2.imread(str(img_path))
        if img is None:
            return []
        
        h, w = img.shape[:2]
        bboxes = []
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define color ranges
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([75, 255, 255])
        
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([15, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        
        # Detect green objects (leaves)
        mask_green = cv2.inRange(hsv, green_lower, green_upper)
        
        # Detect red objects (ripe fruits)
        mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask_green, mask_red)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        min_area = int(0.05 * h * w)  # Minimum 5% of image area
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                
                # Calculate normalized coordinates
                cx_normalized = x / w
                cy_normalized = y / h
                w_normalized = bw / w
                h_normalized = bh / h
                
                valid_contours.append({
                    'bbox': [x, y, bw, bh],
                    'normalized_bbox': [cx_normalized, cy_normalized, w_normalized, h_normalized],
                    'area': area
                })
        
        # Sort by confidence (larger areas first)
        valid_contours.sort(key=lambda c: c['area'], reverse=True)
        
        # Limit to top 5 detections per image
        for contour in valid_contours[:5]:
            bbox = contour['normalized_bbox']
            x, y, bw, bh = bbox
            
            # Ensure reasonable bbox size (between 10% and 90% of image)
            if 0.05 <= bw <= 0.9 and 0.05 <= bh <= 0.9:
                # Adjust center to ensure bbox fits within image bounds
                new_x = max(0.05, min(x, 0.9 - bw))
                new_y = max(0.05, min(y, 0.9 - bh))
                bboxes.append([new_x, new_y, bw, bh])
        
        return bboxes
    
    def classify_content(self, img_path):
        """Classify whether image contains leaves or fruit based on dominant colors."""
        
        img = cv2.imread(str(img_path))
        if img is None:
            return 'unknown', 0
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Green ratio (leaves)
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([75, 255, 255])
        green_pixels = cv2.countNonZero(cv2.inRange(hsv, green_lower, green_upper))
        
        # Red ratio (fruits)
        red_lower = np.array([0, 100, 100])
        red_upper = np.array([15, 255, 255])
        red_pixels1 = cv2.countNonZero(cv2.inRange(hsv, red_lower, red_upper))
        
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        red_pixels2 = cv2.countNonZero(cv2.inRange(hsv, red_lower2, red_upper2))
        
        total_red = red_pixels1 + red_pixels2
        total_pixels = img.shape[0] * img.shape[1]
        
        green_ratio = green_pixels / total_pixels
        red_ratio = total_red / total_pixels
        
        if green_ratio > 0.1 and green_ratio > red_ratio:
            return 'leaf', green_ratio
        elif red_ratio > 0.05:
            return 'fruit', red_ratio
        else:
            return 'mixed', max(green_ratio, red_ratio)
    
    def annotate_dataset(self, dataset_root, output_dir):
        """Full auto-annotation pipeline for dataset."""
        
        print(f"🔍 Starting auto-annotation for {dataset_root}")
        
        # Create output directory
        out_path = Path(output_dir)
        for split in ['train', 'val']:
            img_out = out_path / split / 'images_auto'
            lbl_out = out_path / split / 'labels_auto'
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)
        
        stats = {'processed': 0, 'annotated': 0, 'failed': 0}
        
        for split in ['train', 'val']:
            src_img_dir = Path(dataset_root) / split / 'images'
            
            if not src_img_dir.exists():
                continue
            
            img_files = [f for f in src_img_dir.iterdir() 
                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
            
            print(f"\nProcessing {split}: {len(img_files)} images...")
            
            for idx, img_file in enumerate(img_files):
                try:
                    # Analyze content type
                    content_type, confidence = self.classify_content(str(img_file))
                    
                    # Generate auto-label
                    bboxes = self.detect_bboxes(str(img_file))
                    
                    # Create output filenames
                    base_name = img_file.stem
                    dest_img = out_path / split / 'images_auto' / img_file.name
                    dest_lbl = out_path / split / 'labels_auto' / f"{base_name}.txt"
                    
                    # Copy original image
                    import shutil
                    shutil.copy2(str(img_file), str(dest_img))
                    
                    # Write label file
                    with open(dest_lbl, 'w') as f:
                        for bbox in bboxes:
                            # Determine class_id based on content type and filename
                            fn_lower = img_file.name.lower()
                            
                            if 'tidak_sehat' in fn_lower or 'diseased' in fn_lower:
                                if content_type == 'leaf':
                                    cls_id = 1  # leaf_diseased
                                elif content_type == 'fruit':
                                    cls_id = 3  # fruit_diseased
                                else:
                                    # Keep original from filename
                                    cls_id = 1 if 'buah' in fn_lower or 'fruit' in fn_lower else 0
                            else:
                                if content_type == 'leaf':
                                    cls_id = 0  # leaf_healthy
                                elif content_type == 'fruit':
                                    cls_id = 2  # fruit_healthy
                                else:
                                    cls_id = 0 if 'buah' in fn_lower or 'fruit' in fn_lower else 0
                            
                            f.write(f"{cls_id} {' '.join(f'{x:.6f}' for x in bbox)}\n")
                    
                    stats['processed'] += 1
                    if bboxes:
                        stats['annotated'] += 1
                    
                    if (idx + 1) % 50 == 0:
                        print(f"  Progress: {idx+1}/{len(img_files)}")
                        
                except Exception as e:
                    stats['failed'] += 1
                    if (idx + 1) % 100 == 0:
                        print(f"  Error count: {stats['failed']}")
        
        print(f"\n✅ Auto-annotation complete!")
        print(f"   Processed: {stats['processed']}")
        print(f"   Successfully annotated: {stats['annotated']}")
        print(f"   Failed: {stats['failed']}")
        
        return stats


def main():
    """Run auto-annotation on improved dataset."""
    
    print("=" * 70)
    print("SMARTPEPPER AUTO-ANNOTATOR v1.0")
    print("Generating precise bounding boxes from existing images")
    print("=" * 70)
    
    # Check OpenCV availability
    import sys
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not available! Installing...")
        import subprocess
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'opencv-python-headless'],
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("Failed to install OpenCV. Try manually:")
            print("  pip install opencv-python-headless")
            return
        import cv2
        print(f"✅ OpenCV installed: {cv2.__version__}")
    
    # Run annotation
    annotator = SmartPepperAnnotator()
    
    dataset_paths = [
        ('Leaf Dataset', '/opt/smartpepper-yolo/dataset'),
        ('Fruit Dataset', '/opt/smartpepper-yolo/dataset_pepper_fruit')
    ]
    
    output_base = '/opt/smartpepper-yolo/dataset_v2_enhanced'
    
    for ds_name, ds_path in dataset_paths:
        if os.path.exists(ds_path):
            stats = annotator.annotate_dataset(ds_path, output_base)
            
            print(f"\n📊 Summary for {ds_name}:")
            for k, v in stats.items():
                print(f"   {k}: {v}")
    
    print("\n✅ Annotation complete! Check results in:")
    print(f"   {output_base}/train/images_auto/")
    print(f"   {output_base}/train/labels_auto/")


if __name__ == '__main__':
    main()
