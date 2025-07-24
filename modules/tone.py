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

def generate_halftone(img_gray: np.ndarray, dot_size: int = 3) -> np.ndarray:
    """
    グレースケールトーンを生成する（漫画風）
    """
    # グレースケールでのトーン生成
    tone = img_gray.copy().astype(np.float32)
    
    # 段階的なグレーレベルに量子化（漫画風の階調）
    levels = [255, 220, 180, 140, 100, 60, 20, 0]  # 8段階のグレー
    
    quantized = np.zeros_like(tone)
    for i in range(len(levels) - 1):
        mask = (tone >= levels[i+1]) & (tone < levels[i])
        quantized[mask] = levels[i]
    
    # 最も暗い部分
    quantized[tone < levels[-1]] = levels[-1]
    
    # 軽いぼかしで滑らかに
    quantized = cv2.GaussianBlur(quantized.astype(np.uint8), (3, 3), 0)
    
    return quantized

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

def generate_grayscale_tone(img_gray: np.ndarray, levels: int = 6) -> np.ndarray:
    """
    漫画風のグレースケールトーンを生成する
    """
    # より細かい階調制御
    tone = img_gray.copy().astype(np.float32)
    
    # 指定されたレベル数で量子化
    step = 255 / (levels - 1)
    gray_levels = [int(i * step) for i in range(levels)]
    gray_levels.reverse()  # 明るい順に並べる
    
    quantized = np.zeros_like(tone)
    for i in range(len(gray_levels) - 1):
        lower = gray_levels[i+1]
        upper = gray_levels[i]
        mask = (tone >= lower) & (tone < upper)
        quantized[mask] = upper
    
    # 最も暗い部分
    quantized[tone < gray_levels[-1]] = gray_levels[-1]
    
    return quantized.astype(np.uint8)

def make_dot_pattern(shape, dot_size=6, spacing=12):
    """
    丸点のドットパターン画像を生成
    """
    pattern = np.full(shape, 255, np.uint8)
    for y in range(0, shape[0], spacing):
        for x in range(0, shape[1], spacing):
            cv2.circle(pattern, (x, y), dot_size, 0, -1)
    return pattern

def make_grid_pattern(shape, line_width=2, spacing=12):
    """
    格子（縦横線）パターン画像を生成
    """
    pattern = np.full(shape, 255, np.uint8)
    for y in range(0, shape[0], spacing):
        pattern[y:y+line_width, :] = 0
    for x in range(0, shape[1], spacing):
        pattern[:, x:x+line_width] = 0
    return pattern

def make_stripe_pattern(shape, angle=45, line_width=2, spacing=12):
    """
    斜線パターン画像を生成
    """
    pattern = np.full(shape, 255, np.uint8)
    rad = np.deg2rad(angle)
    for i in range(-shape[1], shape[0], spacing):
        x0 = int(i * np.cos(rad))
        y0 = int(i * np.sin(rad))
        x1 = int((i + shape[1]) * np.cos(rad))
        y1 = int((i + shape[1]) * np.sin(rad))
        cv2.line(pattern, (x0, y0), (x1, y1), 0, line_width)
    return pattern

def make_speed_lines(shape, center=None, num_lines=60):
    """
    集中線（スピード線）パターン画像を生成
    """
    pattern = np.full(shape, 255, np.uint8)
    if center is None:
        center = (shape[1] // 2, shape[0] // 2)
    for i in range(num_lines):
        angle = 2 * np.pi * i / num_lines
        x2 = int(center[0] + shape[1] * np.cos(angle))
        y2 = int(center[1] + shape[0] * np.sin(angle))
        cv2.line(pattern, center, (x2, y2), 0, 1)
    return pattern
