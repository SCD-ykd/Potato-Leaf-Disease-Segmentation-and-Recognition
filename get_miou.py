import os
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
from deeplab import DeeplabV3
from utils.utils_metrics import fast_hist, per_class_iu, per_class_PA_Recall, per_class_Precision, per_Accuracy, show_results


# ==================== 数据收集函数 ====================
def collect_from_category(category_dir):
    images_dir = os.path.join(category_dir, "images")
    masks_dir = os.path.join(category_dir, "masks")
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        return []

    pairs = []
    for img_file in os.listdir(images_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        match = re.search(r'_(\d+)\.', img_file)
        if not match:
            print(f"警告：图片 {img_file} 命名格式不对，跳过")
            continue
        base_without_ext = re.sub(r'\.(jpg|jpeg|png)$', '', img_file, flags=re.I)
        mask_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = os.path.join(masks_dir, f"{base_without_ext}_pseudo{ext}")
            if os.path.exists(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            print(f"警告：未找到 {img_file} 对应的 mask")
            continue
        pairs.append((os.path.join(images_dir, img_file), mask_path))
    return pairs


def get_all_pairs(root_dir):
    all_pairs = []
    for cat_name in os.listdir(root_dir):
        cat_path = os.path.join(root_dir, cat_name)
        if os.path.isdir(cat_path):
            pairs = collect_from_category(cat_path)
            all_pairs.extend(pairs)
    return all_pairs


# ==================== Kappa 计算函数（自实现，不依赖 sklearn） ====================
def compute_kappa_from_confusion(hist):
    """
    从混淆矩阵计算 Cohen's Kappa 系数
    hist: (num_classes, num_classes) 混淆矩阵，行是真值，列是预测值
    """
    total = np.sum(hist)
    if total == 0:
        return 0.0
    p_o = np.trace(hist) / total                     # 总体准确率
    # 每类的真实样本比例（行和）
    p_true = np.sum(hist, axis=1) / total
    # 每类的预测样本比例（列和）
    p_pred = np.sum(hist, axis=0) / total
    p_e = np.sum(p_true * p_pred)                    # 期望一致概率
    kappa = (p_o - p_e) / (1 - p_e + 1e-12)          # 加极小值防止除零
    return kappa


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 测试集根目录
    test_root = "D:/seg/deeplabv3-plus-pytorch-main/split_dataset/test"
    test_pairs = get_all_pairs(test_root)

    test_img_paths = [p[0] for p in test_pairs]
    test_mask_paths = [p[1] for p in test_pairs]

    # 类别设置
    name_classes = ["background", "leaf", "Late blight", "Early blight"]
    num_classes = len(name_classes)

    # 预测结果保存目录
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, 'detection-results')
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    print("加载模型...")
    deeplab = DeeplabV3()

    print("模型加载完成。")

    print("对测试集进行预测...")
    image_ids = []
    for img_path in tqdm(test_img_paths):
        image = Image.open(img_path)
        pred_mask = deeplab.get_miou_png(image)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        image_ids.append(base_name)

        pred_mask.save(os.path.join(pred_dir, base_name + ".png"))
    print("预测完成。")

    # 构建预测 mask 的完整路径列表
    pred_paths = [os.path.join(pred_dir, img_id + ".png") for img_id in image_ids]

    print("计算 mIoU...")
    hist = np.zeros((num_classes, num_classes))
    for gt_path, pred_path in tqdm(zip(test_mask_paths, pred_paths), total=len(test_mask_paths)):
        if not os.path.exists(pred_path):
            print(f"警告：预测文件不存在 {pred_path}")
            continue
        pred = np.array(Image.open(pred_path))
        # ----------------------------------------------------#
        #   定义颜色到类别索引的映射（请根据您的真实 mask 颜色修改）
        # ----------------------------------------------------#
        COLOR_MAP = {
            (0, 0, 0): 0,               # 背景
            (170, 85, 0): 1,            # leaf - 棕色 (#aa5500)
            (85, 170, 127): 2,          # late blight - 绿色 (#55aa7f)
            (255, 170, 255): 3,         # early blight - 粉色 (#ffaaff)
        }

        def rgb_to_index(rgb_mask, color_map):
            """
            将 RGB 彩色 mask 转换为单通道索引 mask
            rgb_mask: numpy array (H, W, 3), dtype=uint8
            color_map: dict { (R,G,B): index }
            """
            h, w = rgb_mask.shape[:2]
            idx_mask = np.zeros((h, w), dtype=np.uint8)
            for color, idx in color_map.items():
                match = np.all(rgb_mask == color, axis=-1)
                idx_mask[match] = idx
            return idx_mask

        # 读取真实 mask 并转换为单通道索引
        label_img = Image.open(gt_path)
        if label_img.mode == 'RGB':
            label_rgb = np.array(label_img)
            label = rgb_to_index(label_rgb, COLOR_MAP)
        else:
            label = np.array(label_img)

        if label.shape != pred.shape:
            print(f"警告：形状不匹配 {gt_path} {label.shape} vs {pred.shape}，跳过")
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

    # 计算各项指标
    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)

    print("mIoU 计算完成。")
    show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

    # ==================== 新增：计算并保存 Kappa 系数 ====================
    kappa_value = compute_kappa_from_confusion(hist)
    print(f"\nKappa coefficient: {kappa_value:.4f}")

    # 保存 Kappa 结果到单独文件
    kappa_save_path = os.path.join(miou_out_path, "kappa_metric.txt")
    with open(kappa_save_path, 'w') as f:
        f.write("Evaluation Metrics - Cohen's Kappa\n")
        f.write("===================================\n")
        f.write(f"Kappa coefficient: {kappa_value:.6f}\n")

        total = np.sum(hist)
        p_o = np.trace(hist) / total
        f.write(f"Overall Accuracy (p_o): {p_o:.6f}\n")
        p_true = np.sum(hist, axis=1) / total
        p_pred = np.sum(hist, axis=0) / total
        p_e = np.sum(p_true * p_pred)
        f.write(f"Expected Agreement (p_e): {p_e:.6f}\n")
        f.write("\nInterpretation of Kappa:\n")
        f.write("  < 0.00: Poor\n")
        f.write("  0.00–0.20: Slight\n")
        f.write("  0.21–0.40: Fair\n")
        f.write("  0.41–0.60: Moderate\n")
        f.write("  0.61–0.80: Substantial\n")
        f.write("  0.81–1.00: Almost Perfect\n")

    # 如果 show_results 已经生成了 results.txt，则将其追加到该文件中
    results_file = os.path.join(miou_out_path, "results.txt")
    if os.path.exists(results_file):
        with open(results_file, 'a') as f:
            f.write(f"\nKappa coefficient: {kappa_value:.6f}\n")
    else:
        # 如果没有 results.txt，则创建一个包含完整指标的汇总文件
        summary_path = os.path.join(miou_out_path, "all_metrics_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("===== Full Metrics Summary =====\n\n")
            f.write("Per-class IoU:\n")
            for i, name in enumerate(name_classes):
                f.write(f"  {name}: {IoUs[i]:.4f}\n")
            f.write(f"\nmIoU: {np.nanmean(IoUs):.4f}\n\n")
            f.write("Per-class Precision:\n")
            for i, name in enumerate(name_classes):
                f.write(f"  {name}: {Precision[i]:.4f}\n")
            f.write(f"\nPer-class Recall (PA):\n")
            for i, name in enumerate(name_classes):
                f.write(f"  {name}: {PA_Recall[i]:.4f}\n")
            f.write(f"\nKappa coefficient: {kappa_value:.6f}\n")

    print(f"\nKappa 结果已保存至: {kappa_save_path}")
    if os.path.exists(results_file):
        print(f"同时已将 Kappa 追加到: {results_file}")
    else:
        print(f"完整指标汇总已保存至: {os.path.join(miou_out_path, 'all_metrics_summary.txt')}")