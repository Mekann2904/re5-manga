import cv2
import numpy as np
from typing import List
from .logger import logger

def extract_canny(img_gray: np.ndarray, threshold1: int, threshold2: int) -> np.ndarray:
    """
    Canny法でエッジを抽出する
    """
    logger.info(f"Extracting Canny edges with thresholds: {threshold1}, {threshold2}")
    edges = cv2.Canny(img_gray, threshold1, threshold2)
    return cv2.bitwise_not(edges) # 白黒反転

def extract_highpass(img: np.ndarray, kernel_size: int, thresh: int) -> np.ndarray:
    """
    ハイパスフィルタでエッジを抽出する
    """
    logger.info(f"Extracting high-pass edges with kernel size: {kernel_size} and threshold: {thresh}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    highpass = cv2.subtract(gray, blurred)
    _, binary = cv2.threshold(highpass, thresh, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_not(binary)

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

def dilate_edges(edge_img: np.ndarray, kernel_size: int = 2, iterations: int = 1) -> np.ndarray:
    """
    線画を太くする（膨張処理）
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(edge_img, kernel, iterations=iterations)
    return dilated
