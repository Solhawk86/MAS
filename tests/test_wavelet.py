import argparse
import cv2
import numpy as np
import pywt
import os


def read_image_color(path):
    """
    读取彩色图像 (BGR)，返回 float32
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img.astype(np.float32)


def resize_if_needed(img, size):
    """
    size: (width, height) or None
    """
    if size is None:
        return img
    w, h = size
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def wavelet_decompose_channel(channel, wavelet, level):
    """
    单通道小波分解
    """
    return pywt.wavedec2(channel, wavelet=wavelet, level=level)


def wavelet_reconstruct_channel(coeffs, wavelet):
    """
    单通道小波重建
    """
    return pywt.waverec2(coeffs, wavelet=wavelet)


def clip_and_uint8(img):
    """
    限幅并转 uint8
    """
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="彩色图像小波 LL 分量交换并重建"
    )
    parser.add_argument("--level", type=int, required=True, help="小波分解层数")
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        help="统一 resize 到的分辨率 (宽 高)"
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="db4",
        help="小波类型（默认 haar）"
    )

    args = parser.parse_args()

    # 读取图像
    image1 = "/mnt/mydata/hong/RMAS_small/test/img/50.jpg"
    image2 = "/mnt/mydata/hong/RMAS_small/test/img/54.jpg"
    img1_array = read_image_color(image1)
    img2_array = read_image_color(image2)

    # resize（如需要）
    img1_array = resize_if_needed(img1_array, args.size)
    img2_array = resize_if_needed(img2_array, args.size)

    # 尺寸检查
    if img1_array.shape != img2_array.shape:
        raise ValueError("两幅图像尺寸不一致，请使用 --size 参数统一尺寸")

    # 拆分通道（OpenCV: BGR）
    channels1 = cv2.split(img1_array)
    channels2 = cv2.split(img2_array)

    rec_channels1 = []
    rec_channels2 = []

    # 对每个通道做小波处理
    for ch1, ch2 in zip(channels1, channels2):
        coeffs1 = wavelet_decompose_channel(ch1, args.wavelet, args.level)
        coeffs2 = wavelet_decompose_channel(ch2, args.wavelet, args.level)

        print(f"交换前 LL 大小: {coeffs1[0].shape} vs {coeffs2[0].shape}")
        # 交换 LL 分量
        coeffs1[0], coeffs2[0] = coeffs2[0], coeffs1[0]

        # 重建
        rec1 = wavelet_reconstruct_channel(coeffs1, args.wavelet)
        rec2 = wavelet_reconstruct_channel(coeffs2, args.wavelet)

        rec_channels1.append(rec1)
        rec_channels2.append(rec2)

    # 合并通道
    img1_rec = cv2.merge(rec_channels1)
    img2_rec = cv2.merge(rec_channels2)

    # 转换为 uint8
    img1_rec = clip_and_uint8(img1_rec)
    img2_rec = clip_and_uint8(img2_rec)

    # 输出文件名
    base1 = os.path.splitext(os.path.basename(image1))[0]
    base2 = os.path.splitext(os.path.basename(image2))[0]

    out1 = f"{base1}_swapLL_color.png"
    out2 = f"{base2}_swapLL_color.png"

    cv2.imwrite(out1, img1_rec)
    cv2.imwrite(out2, img2_rec)

    print("处理完成：")
    print(f"输出图像 1: {out1}")
    print(f"输出图像 2: {out2}")


if __name__ == "__main__":
    main()
