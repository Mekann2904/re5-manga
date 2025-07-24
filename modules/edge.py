import cv2
import numpy as np
from typing import List
from .logger import logger

def extract_canny(img_gray: np.ndarray, threshold1: int, threshold2: int) -> np.ndarray:
    """
    Canny法でエッジを抽出する（改良版）
    """
    logger.info(f"Extracting Canny edges with thresholds: {threshold1}, {threshold2}")
    
    # より鮮明な輪郭線のための前処理
    # 1. コントラスト強化
    enhanced = cv2.convertScaleAbs(img_gray, alpha=1.2, beta=10)
    
    # 2. Cannyエッジ検出（aperture_sizeを調整してより細かいエッジを検出）
    edges = cv2.Canny(enhanced, threshold1, threshold2, apertureSize=3, L2gradient=True)
    
    # 3. モルフォロジー処理で線を連続化
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return edges # 線は白(255), 背景が黒(0)

def extract_highpass(img: np.ndarray, kernel_size: int, thresh: int) -> np.ndarray:
    """
    ハイパスフィルタでエッジを抽出する（改良版）
    """
    logger.info(f"Extracting high-pass edges with kernel size: {kernel_size} and threshold: {thresh}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # より効果的なハイパスフィルタ
    # 1. ガウシアンブラーでローパス成分を作成
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    
    # 2. 元画像からローパス成分を引いてハイパス成分を取得
    highpass = cv2.subtract(gray, blurred)
    
    # 3. コントラストを強化
    highpass = cv2.convertScaleAbs(highpass, alpha=2.0, beta=0)
    
    # 4. ノイズ除去のための軽いぼかし
    highpass = cv2.medianBlur(highpass, 3)
    
    return highpass

def extract_binary(img_gray: np.ndarray, thresh: int) -> np.ndarray:
    """
    単純な二値化でエッジを抽出する
    """
    logger.info(f"Extracting binary edges with threshold: {thresh}")
    _, binary = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
    return binary

def combine_edges(edges: List[np.ndarray]) -> np.ndarray:
    """
    複数の線画レイヤーを合成する
    """
    logger.info(f"Combining {len(edges)} edge layers.")
    combined = np.full(edges[0].shape, 255, dtype=np.uint8)
    for edge in edges:
        combined = cv2.bitwise_and(combined, edge)
    return combined

def enhance_edges(img_gray: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """
    輪郭線を強化する
    """
    logger.info(f"Enhancing edges with strength: {strength}")
    
    # 1. Sobelフィルタで勾配を計算
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 2. 勾配の大きさを計算
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = np.uint8(np.clip(magnitude * strength, 0, 255))
    
    # 3. ラプラシアンフィルタも併用
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    laplacian = np.uint8(np.clip(np.abs(laplacian) * strength, 0, 255))
    
    # 4. 両方を組み合わせ
    enhanced = cv2.addWeighted(magnitude, 0.7, laplacian, 0.3, 0)
    
    return enhanced

def dilate_edges(edge_img: np.ndarray, kernel_size: int = 2, iterations: int = 1) -> np.ndarray:
    """
    線画を太くする（膨張処理）
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(edge_img, kernel, iterations=iterations)
    return dilated
