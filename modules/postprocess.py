import cv2
import numpy as np
from .logger import logger

def remove_noise(img: np.ndarray, method: str = 'morphology', kernel: tuple = (3, 3)) -> np.ndarray:
    """
    ノイズを除去する
    """
    logger.info(f"Removing noise using {method} with kernel {kernel}")
    if method == 'morphology':
        # オープニング処理（収縮→膨張）で小さいノイズを除去
        k = np.ones(kernel, np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, k)
    return img

def cleanup_edges(img: np.ndarray, params: dict = None) -> np.ndarray:
    """
    線画をクリーンアップする（この例では何もしない）
    """
    logger.info("Cleaning up edges (not implemented).")
    return img
