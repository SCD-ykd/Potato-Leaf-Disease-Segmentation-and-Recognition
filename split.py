import random
import shutil
from pathlib import Path
from typing import List, Dict, Optional

# ==================== 配置参数 ====================
source_root = r"D:\数据集\potato"  # 源根目录 (包含 Early blight, Late blight 等)
target_root = r"D:\数据集\split_dataset"  # 目标根目录 (将生成 train/val/test)
ratios = [7, 2, 1]  # train : val : test
random_seed = 42  # 随机种子
image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}  # 支持的图片扩展名
mask_exts = ['.png']  # 掩码文件扩展名（通常为 .png）
mask_suffix = "_pseudo"  # 掩码文件名后缀（如 image_123.png 对应 image_123_pseudo.png）；若不需要设为 None


# =================================================

def collect_images_and_masks(category_path: Path) -> List[tuple]:
    """
    收集一个类别下的所有图片及其对应的掩码文件路径。
    返回列表，每个元素为 (图片路径, 掩码路径或None)
    """
    images_dir = category_path / "images"
    masks_dir = category_path / "masks"

    if not images_dir.exists():
        print(f"警告：{category_path.name} 下没有 images 文件夹，跳过")
        return []
    if not masks_dir.exists():
        print(f"警告：{category_path.name} 下没有 masks 文件夹，掩码可能缺失")

    # 收集所有图片
    image_files = []
    for ext in image_exts:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))
    image_files = list(set(image_files))  # 去重

    result = []
    for img_path in image_files:
        stem = img_path.stem  # 不含扩展名的文件名
        mask_path = None

        # 1. 尝试直接匹配掩码（同名 + 扩展名）
        for ext in mask_exts:
            candidate = masks_dir / f"{stem}{ext}"
            if candidate.exists():
                mask_path = candidate
                break

        # 2. 如果失败且设置了 mask_suffix，尝试添加后缀
        if mask_path is None and mask_suffix is not None:
            for ext in mask_exts:
                candidate = masks_dir / f"{stem}{mask_suffix}{ext}"
                if candidate.exists():
                    mask_path = candidate
                    break

        # 3. 也可能图片名本身已包含后缀（如 image_pseudo.png），尝试去掉后缀匹配
        if mask_path is None and mask_suffix is not None and mask_suffix in stem:
            base_stem = stem.replace(mask_suffix, "")
            for ext in mask_exts:
                candidate = masks_dir / f"{base_stem}{ext}"
                if candidate.exists():
                    mask_path = candidate
                    break
            # 同时尝试带后缀的掩码文件
            if mask_path is None:
                for ext in mask_exts:
                    candidate = masks_dir / f"{base_stem}{mask_suffix}{ext}"
                    if candidate.exists():
                        mask_path = candidate
                        break

        if mask_path is None:
            print(f"  警告：未找到 {img_path.name} 对应的掩码文件，将只复制图片")

        result.append((img_path, mask_path))

    return result


def main():
    src_root = Path(source_root)
    if not src_root.exists():
        print(f"错误：源目录不存在 -> {src_root}")
        return

    # 获取所有类别（子文件夹）
    categories = [d for d in src_root.iterdir() if d.is_dir()]
    if not categories:
        print("错误：源目录下没有类别文件夹")
        return
    print(f"发现类别: {[c.name for c in categories]}")

    # 存储每个类别的 (图片路径, 掩码路径) 列表
    cat_data: Dict[str, List[tuple]] = {}
    for cat in categories:
        data = collect_images_and_masks(cat)
        if not data:
            print(f"跳过类别 {cat.name}（没有图片）")
            continue
        cat_data[cat.name] = data
        print(f"类别 {cat.name}: {len(data)} 张图片")

    if not cat_data:
        print("没有找到任何图片，退出")
        return

    random.seed(random_seed)
    target_root_path = Path(target_root)
    target_root_path.mkdir(parents=True, exist_ok=True)

    subset_names = ['train', 'val', 'test']
    ratio_sum = sum(ratios)
    total_counts = {s: 0 for s in subset_names}

    for cat_name, items in cat_data.items():
        # 随机打乱
        random.shuffle(items)
        total = len(items)
        n_train = int(total * ratios[0] / ratio_sum)
        n_val = int(total * ratios[1] / ratio_sum)
        n_test = total - n_train - n_val

        splits = {
            'train': items[:n_train],
            'val': items[n_train:n_train + n_val],
            'test': items[n_train + n_val:]
        }
        print(f"\n类别 {cat_name}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

        for subset in subset_names:
            # 创建目标子文件夹: target_root/subset/category_name/images/ 和 masks/
            subset_cat_dir = target_root_path / subset / cat_name
            img_dst_dir = subset_cat_dir / "images"
            mask_dst_dir = subset_cat_dir / "masks"
            img_dst_dir.mkdir(parents=True, exist_ok=True)
            mask_dst_dir.mkdir(parents=True, exist_ok=True)

            for img_path, mask_path in splits[subset]:
                # 复制图片
                dst_img = img_dst_dir / img_path.name
                try:
                    shutil.copy2(img_path, dst_img)
                    total_counts[subset] += 1
                except Exception as e:
                    print(f"  复制图片失败 {img_path.name}: {e}")
                    continue

                # 复制掩码（如果存在）
                if mask_path:
                    dst_mask = mask_dst_dir / mask_path.name
                    try:
                        shutil.copy2(mask_path, dst_mask)
                        # print(f"  已复制掩码: {mask_path.name}")
                    except Exception as e:
                        print(f"  复制掩码失败 {mask_path.name}: {e}")
                # else: 无掩码，已警告过，不复制

    print("\n========== 分配完成 ==========")
    for subset in subset_names:
        print(f"{subset}: {total_counts[subset]} 张图片")
    print(f"目标根目录: {target_root_path}")
    print("目录结构示例:")
    print("  train/类别名/images/")
    print("  train/类别名/masks/")
    print("  val/...")
    print("  test/...")


if __name__ == "__main__":
    main()