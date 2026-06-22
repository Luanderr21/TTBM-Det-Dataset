"""
将 Tower_defects_detection_renew 数据集按三种比例和不同随机种子重新划分。

划分配置:
  - 70/15/15, seed=42
  - 60/20/20, seed=123
  - 80/10/10, seed=7

输出目录: A:/ml_learning/data/Tower_3splits/
每个 split 包含完整的 YOLO 目录结构（images/ + labels/ 的 train/val/test 子目录）
以及 train.txt / val.txt / test.txt 索引文件。

注意：仅使用 images/ 和 labels/ 下的平铺文件，忽略已有的 train/val/test 子目录。
"""

import os
import shutil
import random
from pathlib import Path

# ============ 配置 ============
SRC_DIR = Path("A:/ml_learning/data/Tower_defects_detection_renew")
DST_DIR = Path("A:/ml_learning/data/Tower_3splits")

# 三个划分方案: (名称, train_ratio, val_ratio, test_ratio, seed)
SPLITS = [
    ("split_70_15_15", 0.70, 0.15, 0.15, 42),
    ("split_60_20_20", 0.60, 0.20, 0.20, 123),
    ("split_80_10_10", 0.80, 0.10, 0.10, 7),
]


def collect_samples(src_dir: Path):
    """收集 images/ 目录下所有平铺的 .jpg 文件（忽略子目录），返回 (stem, image_path, label_path) 列表"""
    samples = []
    images_flat = src_dir / "images"
    labels_flat = src_dir / "labels"

    seen_subdirs = {"train", "val", "test"}

    for img_path in images_flat.iterdir():
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue

        stem = img_path.stem  # e.g. "obj_0001"
        label_path = labels_flat / f"{stem}.txt"

        if label_path.exists():
            samples.append((stem, img_path, label_path))

    return samples


def split_and_copy(samples, split_name, train_r, val_r, test_r, seed, dst_dir):
    """按给定比例打乱划分并拷贝到目标目录"""
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train_r)
    n_val = round(n * val_r)
    # 余数归入 test，确保不丢样本
    n_test = n - n_train - n_val

    # 边界保护
    if n_train < 0:
        n_train = 0
    if n_val < 0:
        n_val = 0
    if n_test < 0:
        n_test = 0

    train_set = shuffled[:n_train]
    val_set = shuffled[n_train : n_train + n_val]
    test_set = shuffled[n_train + n_val :]

    subsets = {
        "train": train_set,
        "val": val_set,
        "test": test_set,
    }

    print(f"  [{split_name}] seed={seed}, total={n}, train={n_train}, val={n_val}, test={n_test}")

    # 为每个子集创建目录并拷贝文件
    for subset_name, subset_data in subsets.items():
        img_dst = dst_dir / split_name / "images" / subset_name
        lbl_dst = dst_dir / split_name / "labels" / subset_name
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for stem, img_path, label_path in subset_data:
            # 拷贝图片
            shutil.copy2(img_path, img_dst / img_path.name)
            # 拷贝标签
            shutil.copy2(label_path, lbl_dst / label_path.name)

    # 生成 train.txt / val.txt / test.txt
    for subset_name, subset_data in subsets.items():
        txt_path = dst_dir / split_name / f"{subset_name}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for stem, img_path, _label_path in subset_data:
                # 写入目标路径（绝对路径）
                target_img = dst_dir / split_name / "images" / subset_name / img_path.name
                f.write(f"{target_img.as_posix()}\n")

    # 拷贝 classes.txt 到每个 split 的 labels/ 目录
    src_classes = SRC_DIR / "labels" / "classes.txt"
    if src_classes.exists():
        shutil.copy2(src_classes, dst_dir / split_name / "labels" / "classes.txt")

    return {"train": n_train, "val": n_val, "test": n_test}


def main():
    print("=" * 60)
    print("数据集划分脚本")
    print(f"源目录: {SRC_DIR}")
    print(f"目标目录: {DST_DIR}")
    print("=" * 60)

    # 收集样本
    samples = collect_samples(SRC_DIR)
    print(f"\n收集到 {len(samples)} 个有效样本（图像+标签配对）")

    if len(samples) == 0:
        print("错误: 未找到有效样本！")
        return

    # 确认样本名示例
    print(f"示例样本: {samples[:3]}")

    # 清空目标目录（如果存在）
    if DST_DIR.exists():
        print(f"\n清空已有目标目录: {DST_DIR}")
        shutil.rmtree(DST_DIR)
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # 执行三种划分
    print("\n开始划分...\n")
    results = {}
    for split_name, train_r, val_r, test_r, seed in SPLITS:
        results[split_name] = split_and_copy(
            samples, split_name, train_r, val_r, test_r, seed, DST_DIR
        )

    # 打印汇总
    print("\n" + "=" * 60)
    print("划分完成! 汇总:")
    print("=" * 60)
    for split_name, counts in results.items():
        print(f"  {split_name}: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    print(f"\n输出目录: {DST_DIR}")
    print("目录结构:")
    for item in sorted(DST_DIR.rglob("*")):
        if item.is_dir():
            rel = item.relative_to(DST_DIR)
            depth = len(rel.parts)
            if depth <= 3:
                print(f"  {'  ' * depth}{rel}/")


if __name__ == "__main__":
    main()
