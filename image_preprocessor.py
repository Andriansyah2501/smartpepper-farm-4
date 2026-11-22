import cv2
import numpy as np

def enhance_pepper_image(img_bgr):
    """
    Pipeline Pengolahan Citra Digital untuk Meningkatkan Akurasi Deteksi Penyakit Lada:
    1. Denoising: Gaussian Bilateral Filter (mereduksi noise kanopi tanpa mengaburkan tepi bercak/lesi).
    2. Adaptive Illumination: Contrast Limited Adaptive Histogram Equalization (CLAHE) pada channel L (LAB).
    3. Unsharp Masking: Menajamkan tekstur tepi luka nekrosis/bakteri.
    4. Color Vibrance: Kalibrasi saturasi dan balance warna klorofil hijau dan bercak cokelat.
    """
    # 1. Bilateral Filter Denoising
    denoised = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=50, sigmaSpace=50)

    # 2. CLAHE pada Color Space LAB (hanya menaikkan kontras lokal tanpa merusak warna daun)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # 3. Unsharp Masking (Penajaman Tepi Lesi Daun)
    gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(enhanced_bgr, 1.35, gaussian, -0.35, 0)

    # 4. Keseimbangan Saturasi (Peningkatan kontras klorofil vs nekrosis)
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 1.15)
    s = np.clip(s, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    final_bgr = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return final_bgr
