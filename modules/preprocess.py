import cv2
import numpy as np
from .logger import logger

def resize(img: np.ndarray, dpi: int) -> np.ndarray:
    """
    指定されたDPIに基づいて画像をリサイズする（この例では何もしない）
    """
    logger.info(f"Resizing image for {dpi} DPI (not implemented).")
    # DPIに基づいたリサイズは複雑なため、この例ではスキップ
    return img

def adjust_contrast(img: np.ndarray, method: str = 'clahe', params: dict = None) -> np.ndarray:
    """
    コントラストを調整する
    """
    logger.info(f"Adjusting contrast using {method}")
    if method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # カラー画像の場合は、輝度チャネルに適用
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        else:
            final_img = clahe.apply(img)
        return final_img
    else:
        # 他のコントラスト調整手法はここに追加
        return img
