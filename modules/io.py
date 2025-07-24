import cv2
import numpy as np
from .logger import logger

def load_image(path: str) -> np.ndarray:
    """
    画像を読み込む
    """
    logger.info(f"Loading image from: {path}")
    img = cv2.imread(path)
    if img is None:
        logger.error(f"Failed to load image: {path}")
        raise FileNotFoundError(f"Image not found at {path}")
    return img

def save_image(img: np.ndarray, path: str, dpi: int):
    """
    画像を保存する
    """
    logger.info(f"Saving image to: {path} with {dpi} DPI")
    # OpenCVはDPI情報を直接埋め込めないため、ここではフラグのみ
    # 必要であればPillowなどを使う
    cv2.imwrite(path, img)
