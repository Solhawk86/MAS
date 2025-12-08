import os
from PIL import Image
import numpy as np


def _dump_pixel_matrix(mask_array, mask_path, matrix_output_dir):
    """将像素矩阵写入 txt 文件，不在终端输出。"""
    if matrix_output_dir:
        os.makedirs(matrix_output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(mask_path))[0]
        out_path = os.path.join(matrix_output_dir, f"{base_name}_matrix.txt")
    else:
        out_path = f"{mask_path}_matrix.txt"

    array_str = np.array2string(
        mask_array,
        separator=" ",
        threshold=mask_array.size,
        max_line_width=max(mask_array.size * 6, 120),
    )

    with open(out_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(f"mask_path: {mask_path}\n")
        txt_file.write("像素矩阵:\n")
        txt_file.write(array_str)
        txt_file.write("\n")


def check_and_fix_resolution(pred_dir, gt_dir, pred_suffix=".jpg", gt_suffix=".png"):
    """
    检查预测图和GT的分辨率是否一致，如果不一致，将GT resize到预测图尺寸。

    参数:
        pred_dir (str): 预测图所在文件夹路径
        gt_dir (str): GT图所在文件夹路径
        pred_suffix (str): 预测图的文件后缀名（默认 .jpg）
        gt_suffix (str): GT图的文件后缀名（默认 .png）
    """
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(pred_suffix)])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(gt_suffix)])

    if len(pred_files) != len(gt_files):
        print(f"文件数量不一致：pred = {len(pred_files)}, gt = {len(gt_files)}")

    mismatch_count = 0

    for pred_name, gt_name in zip(pred_files, gt_files):
        pred_path = os.path.join(pred_dir, pred_name)
        gt_path = os.path.join(gt_dir, gt_name)

        try:
            pred_img = Image.open(pred_path)
            gt_img = Image.open(gt_path)

            if pred_img.size != gt_img.size:
                mismatch_count += 1
                print(f"分辨率不匹配: {pred_name} / {gt_name}")
                print(f"pred: {pred_img.size}, gt: {gt_img.size}")

                # resize GT 到预测图大小
                gt_resized = gt_img.resize(pred_img.size, Image.NEAREST)
                gt_resized.save(gt_path)
                print(f"  已将 GT resize 到 {pred_img.size}")

        except Exception as e:
            print(f"⚠️ 无法读取 {pred_name} 或 {gt_name}: {e}")

    if mismatch_count == 0:
        print("所有图像分辨率一致！")
    else:
        print(f"检查完成，共发现 {mismatch_count} 个不匹配的图像，并已修正。")
        
def value_check(
    path,
    suffixes=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    show_pixel_matrix=False,
    matrix_output_dir=None,
):
    """
    批量检查 mask 图像，仅输出异常数量及总数量。
    """
    if os.path.isfile(path):
        inspect_mask(
            path,
            show_pixel_matrix=show_pixel_matrix,
            matrix_output_dir=matrix_output_dir,
        )
        return

    if not os.path.isdir(path):
        print(f"⚠️ 路径 {path} 不存在")
        return

    paths_to_check = [
        os.path.join(path, fname)
        for fname in sorted(os.listdir(path))
        if os.path.isfile(os.path.join(path, fname)) and fname.lower().endswith(suffixes)
    ]

    if not paths_to_check:
        print(f"⚠️ 文件夹 {path} 中没有匹配的图像: {suffixes}")
        return

    invalid_image_count = 0
    image_count = 0

    for mask_path in paths_to_check:
        image_count += 1
        try:
            mask = Image.open(mask_path).convert("L")
            mask_array = np.array(mask)
            unique_vals = np.unique(mask_array)

            if not np.all(np.isin(unique_vals, [0, 255])):
                invalid_image_count += 1

            if show_pixel_matrix:
                _dump_pixel_matrix(mask_array, mask_path, matrix_output_dir)
        except Exception as exc:
            print(f"⚠️ 无法读取 {mask_path}: {exc}")

    print(f"包含除0和255以外像素值的图片数量: {invalid_image_count}")
    print(f"总图片数量: {image_count}")


def inspect_mask(mask_path, show_pixel_matrix=False, matrix_output_dir=None):
    """
    单张图像检查，输出详细的像素统计信息。
    """
    if not os.path.isfile(mask_path):
        print(f"⚠️ 路径 {mask_path} 不存在或不是文件")
        return

    try:
        mask = Image.open(mask_path).convert("L")
        mask_array = np.array(mask)
    except Exception as exc:
        print(f"⚠️ 无法读取 {mask_path}: {exc}")
        return

    print(f"检查文件: {mask_path}")
    print("mask形状:", mask_array.shape)
    print("数据类型:", mask_array.dtype)
    print("像素最小值:", mask_array.min())
    print("像素最大值:", mask_array.max())
    print("唯一像素值:", np.unique(mask_array))

    if show_pixel_matrix:
        _dump_pixel_matrix(mask_array, mask_path, matrix_output_dir)


def binarize_masks(
    input_path,
    output_dir=None,
    suffixes=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
    threshold=128,
):
    """
    将灰度图二值化：像素值 < threshold 置为 0，其余置为 255。
    支持单张图片或整个文件夹批量处理，可自定义输出文件夹。
    """
    if os.path.isdir(input_path):
        image_paths = sorted(
            [
                os.path.join(input_path, fname)
                for fname in os.listdir(input_path)
                if os.path.isfile(os.path.join(input_path, fname))
                and fname.lower().endswith(suffixes)
            ]
        )
        if not image_paths:
            print(f"⚠️ 文件夹 {input_path} 中没有匹配的图像: {suffixes}")
            return
        target_dir = output_dir or input_path
        os.makedirs(target_dir, exist_ok=True)
    elif os.path.isfile(input_path):
        image_paths = [input_path]
        target_dir = output_dir or os.path.dirname(input_path) or "."
        os.makedirs(target_dir, exist_ok=True)
    else:
        print(f"⚠️ 输入路径 {input_path} 不存在")
        return

    processed = 0

    for mask_path in image_paths:
        try:
            mask = Image.open(mask_path).convert("L")
            mask_array = np.array(mask, dtype=np.uint8)

            # 严格按照“大于 threshold 为 255，否则为 0”的规则
            bin_array = np.zeros_like(mask_array, dtype=np.uint8)
            bin_array[mask_array > threshold] = 255
            bin_image = Image.fromarray(bin_array, mode="L")

            base_name, ext = os.path.splitext(os.path.basename(mask_path))
            ext_lower = ext.lower()
            renamed_for_png = False

            if ext_lower in {".jpg", ".jpeg"}:
                save_name = f"{base_name}.png"
                renamed_for_png = True
            else:
                save_name = f"{base_name}{ext}"

            if output_dir is None and os.path.isfile(input_path):
                if renamed_for_png:
                    save_path = os.path.join(
                        os.path.dirname(mask_path), f"{base_name}.png"
                    )
                else:
                    save_path = mask_path
            else:
                save_path = os.path.join(target_dir, save_name)

            save_kwargs = {}
            if save_path.lower().endswith((".jpg", ".jpeg")):
                # 对于仍需 JPEG 格式的情况，尽可能降低失真
                save_kwargs = {"quality": 100, "subsampling": 0, "optimize": False}

            bin_image.save(save_path, **save_kwargs)

            if renamed_for_png and os.path.splitext(save_path)[1].lower() == ".png":
                print(f"提示: {mask_path} 为 JPEG，已转存为 PNG -> {save_path}")

            processed += 1
        except Exception as exc:
            print(f"⚠️ 无法处理 {mask_path}: {exc}")

    print(f"共处理 {processed} 张图片，输出路径: {target_dir}")

if __name__ == "__main__":
    # binarize_masks("/mnt/mydata/hong/MAS3K/test/Mask", output_dir="/mnt/mydata/hong/MAS3K/test/binary_label")
    value_check("/mnt/mydata/hong/MAS3K/test/binary_label")
    # inspect_mask("/mnt/mydata/hong/RMAS_small/test/binary_label/49.png")