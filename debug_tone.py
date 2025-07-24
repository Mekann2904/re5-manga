#!/usr/bin/env python3
import cv2
import numpy as np
from modules.io import load_image
from modules.preprocess import adjust_contrast
from modules.tone import generate_posterized, generate_halftone

# Load and process image
img = load_image('input.jpg')
img_contrast = adjust_contrast(img)
img_gray = cv2.cvtColor(img_contrast, cv2.COLOR_BGR2GRAY)

# Generate tones
tone_poster = generate_posterized(img_gray, 4)
tone_simple = cv2.GaussianBlur(img_gray, (3,3), 0)
tone_combined = cv2.addWeighted(tone_poster, 0.5, tone_simple, 0.5, 0)
tone_halftone = generate_halftone(tone_combined, dot_size=4)

# Save debug images
cv2.imwrite('debug_tone_poster.png', tone_poster)
cv2.imwrite('debug_tone_simple.png', tone_simple)
cv2.imwrite('debug_tone_combined.png', tone_combined)
cv2.imwrite('debug_tone_halftone.png', tone_halftone)

print("Debug tone images saved!")
print(f"Original gray mean: {np.mean(img_gray)}")
print(f"Poster tone mean: {np.mean(tone_poster)}")
print(f"Combined tone mean: {np.mean(tone_combined)}")
print(f"Halftone mean: {np.mean(tone_halftone)}")