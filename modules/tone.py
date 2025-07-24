import cv2
import numpy as np
from .logger import logger

def generate_posterized(img_gray: np.ndarray, levels: int) -> np.ndarray:
    """
    ポスタリゼーションでトーンを生成する
    """
    logger.info(f"Generating posterized tone with {levels} levels.")
    indices = np.arange(0, 256)
    divider = np.linspace(0, 255, levels + 1)[1] # find the divider
    quantiz = np.int_(np.linspace(0, 255, levels))
    color_levels = np.clip(np.int_(indices / divider) * divider, 0, 255)
    palette = quantiz
    
    # Create a lookup table
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = palette[np.argmin(np.abs(palette - color_levels[i]))]
        
    return cv2.LUT(img_gray, lut)

def generate_levels(img_gray: np.ndarray, level_params: dict) -> np.ndarray:
    """
    レベル補正でトーンを生成する（この例では単純な二値化）
    """
    logger.info(f"Generating level-based tone with params: {level_params}")
    thresh = level_params.get('thresh', 128)
    _, binary = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
    return binary

def generate_halftone(img_gray: np.ndarray, dot_size: int = 4) -> np.ndarray:
    """
    網点トーンを生成する（シンプルなドットパターン）
    """
    h, w = img_gray.shape
    halftone = np.full((h, w), 255, dtype=np.uint8)
    for y in range(0, h, dot_size):
        for x in range(0, w, dot_size):
            block = img_gray[y:y+dot_size, x:x+dot_size]
            mean = np.mean(block)
            if mean < 128:
                cv2.circle(halftone, (x + dot_size//2, y + dot_size//2), dot_size//3, 0, -1)
    return halftone

def apply_mask(base: np.ndarray, tone: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    マスクを使ってトーンを合成する
    """
    logger.info("Applying tone mask.")
    # マスクが3チャネルの場合、グレースケールに変換
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # マスクを二値化
    _, mask_binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # マスクの白領域にトーンを適用
    masked_tone = cv2.bitwise_and(tone, tone, mask=mask_binary)
    
    # ベース画像のマスク領域をクリア
    base_cleared = cv2.bitwise_and(base, base, mask=cv2.bitwise_not(mask_binary))
    
    # 合成
    return cv2.add(base_cleared, masked_tone)
