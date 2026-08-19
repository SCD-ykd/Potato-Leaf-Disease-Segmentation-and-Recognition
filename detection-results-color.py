import os
import numpy as np
from PIL import Image

# 颜色映射（必须与训练时定义的顺序一致）
color_map = {
    0: (0, 0, 0),          # 背景
    1: (170, 85, 0),       # leaf
    2: (85, 170, 127),     # Late blight
    3: (255, 170, 255),    # Early blight
}

input_dir = "miou_out6/detection-results"
output_dir = "miou_out6/detection-results-color"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.endswith(".png"):
        continue
    mask = np.array(Image.open(os.path.join(input_dir, fname)))
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in color_map.items():
        color_img[mask == idx] = color
    Image.fromarray(color_img).save(os.path.join(output_dir, fname))
    print(f"已转换: {fname}")