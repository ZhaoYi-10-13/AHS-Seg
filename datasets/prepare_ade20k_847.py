#!/usr/bin/env python3
# ==========================================
# ADE20K-847 数据集准备脚本（适配AHS-Seg）
# ==========================================
# 基于H-CLIP的prepare_ade20k_full.py修改
# 用于准备最难的评估数据集（847个类别）

import os
import pickle as pkl
from pathlib import Path

import cv2
import numpy as np
import tqdm
from PIL import Image

# 从H-CLIP复制847个类别的定义
# 这里只包含前100行作为示例，完整版本需要从H-CLIP复制
ADE20K_SEM_SEG_FULL_CATEGORIES = [
    {"name": "wall", "id": 2978, "trainId": 0},
    {"name": "building, edifice", "id": 312, "trainId": 1},
    {"name": "sky", "id": 2420, "trainId": 2},
    {"name": "tree", "id": 2855, "trainId": 3},
    {"name": "road, route", "id": 2131, "trainId": 4},
    # ... 完整列表需要从H-CLIP复制，共847个类别
]

def loadAde20K(file):
    """加载ADE20K标注文件"""
    fileseg = file.replace(".jpg", "_seg.png")
    with Image.open(fileseg) as io:
        seg = np.array(io)

    R = seg[:, :, 0]
    G = seg[:, :, 1]
    ObjectClassMasks = (R / 10).astype(np.int32) * 256 + (G.astype(np.int32))

    return {"img_name": file, "segm_name": fileseg, "class_mask": ObjectClassMasks}


if __name__ == "__main__":
    print("=" * 50)
    print("ADE20K-847 数据集准备脚本")
    print("=" * 50)
    
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "/root/datasets"))
    index_file = dataset_dir / "ADE20K_2021_17_01" / "index_ade20k.pkl"
    
    # 检查索引文件是否存在
    if not index_file.exists():
        print(f"❌ 错误: 找不到索引文件 {index_file}")
        print("请先下载ADE20K-847数据集并解压")
        print("运行: bash /root/datasets/download_ade20k_847.sh")
        exit(1)
    
    print(f"读取索引文件: {index_file}")
    print('注意: 我们只生成验证集（validation set）!')
    
    # 从H-CLIP复制完整的847个类别定义
    # 这里需要完整复制，暂时使用占位符
    print("⚠️  警告: 需要从H-CLIP复制完整的847个类别定义")
    print("请从以下文件复制:")
    print("  /root/H-CLIP/datasets/prepare_ade20k_full.py")
    print("  第13-929行的 ADE20K_SEM_SEG_FULL_CATEGORIES")
    
    # 临时解决方案：尝试从H-CLIP导入
    try:
        import sys
        sys.path.insert(0, '/root/H-CLIP')
        from datasets.prepare_ade20k_full import ADE20K_SEM_SEG_FULL_CATEGORIES
        print(f"✅ 成功从H-CLIP导入847个类别定义")
    except ImportError:
        print("❌ 无法从H-CLIP导入，请确保H-CLIP项目存在")
        print("或者手动复制类别定义到此文件")
        exit(1)
    
    with open(index_file, "rb") as f:
        index_ade20k = pkl.load(f)

    # 创建ID映射
    id_map = {}
    for cat in ADE20K_SEM_SEG_FULL_CATEGORIES:
        id_map[cat["id"]] = cat["trainId"]
    
    print(f"✅ 类别映射创建完成，共 {len(id_map)} 个类别")

    # 创建输出目录
    for name in ["training", "validation"]:
        image_dir = dataset_dir / "ADE20K_2021_17_01" / "images_detectron2" / name
        image_dir.mkdir(parents=True, exist_ok=True)
        annotation_dir = dataset_dir / "ADE20K_2021_17_01" / "annotations_detectron2" / name
        annotation_dir.mkdir(parents=True, exist_ok=True)

    # 处理图像和标注
    print("开始处理图像和标注...")
    for i, (folder_name, file_name) in tqdm.tqdm(
        enumerate(zip(index_ade20k["folder"], index_ade20k["filename"])),
        total=len(index_ade20k["filename"]),
    ):
        split = "validation" if file_name.split("_")[1] == "val" else "training"
        if split == 'training':
            # 只处理验证集
            continue
        
        info = loadAde20K(str(dataset_dir / folder_name / file_name))

        # 调整图像和标签大小
        img = np.asarray(Image.open(info["img_name"]))
        lab = np.asarray(info["class_mask"])

        h, w = img.shape[0], img.shape[1]
        max_size = 512
        resize = True
        if w >= h > max_size:
            h_new, w_new = max_size, round(w / float(h) * max_size)
        elif h >= w > max_size:
            h_new, w_new = round(h / float(w) * max_size), max_size
        else:
            resize = False

        if resize:
            img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
            lab = cv2.resize(lab, (w_new, h_new), interpolation=cv2.INTER_NEAREST)

        assert img.dtype == np.uint8
        assert lab.dtype == np.int32

        # 应用标签转换并保存为uint16图像
        output = np.zeros_like(lab, dtype=np.uint16) + 65535
        for obj_id in np.unique(lab):
            if obj_id in id_map:
                output[lab == obj_id] = id_map[obj_id]

        output_img = dataset_dir / "ADE20K_2021_17_01" / "images_detectron2" / split / file_name
        output_lab = (
            dataset_dir
            / "ADE20K_2021_17_01"
            / "annotations_detectron2"
            / split
            / file_name.replace(".jpg", ".tif")
        )
        Image.fromarray(img).save(output_img)
        Image.fromarray(output).save(output_lab)

    print("=" * 50)
    print("✅ ADE20K-847 数据集准备完成！")
    print("=" * 50)
    print(f"输出目录: {dataset_dir / 'ADE20K_2021_17_01'}")
    print("下一步: 可以开始评估了！")
