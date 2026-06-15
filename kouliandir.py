# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u kouliandir.py > koulianlog1226.txt 2>&1 &

import os
import cv2
from PIL import Image

# 这里假设 detector 已经在外部初始化，比如：
# from mtcnn import MTCNN
# detector = MTCNN()

def detect_and_crop_face(image_bgr):
    """
    检测人脸并返回裁剪后的人脸RGB图像（numpy数组）。若未检测到返回 None。
    """

    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_image)

    if faces:
        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)
        cropped = rgb_image[y:y+h, x:x+w]
        return cropped
    else:
        return None


def process_images(input_dir, output_dir):
    """
    遍历 input_dir 下的所有 JPG 文件，裁剪人脸并保存到 output_dir
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".jpg"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            img = cv2.imread(input_path)
            if img is None:
                print(f"无法读取文件: {filename}")
                continue

            cropped_face = detect_and_crop_face(img)
            if cropped_face is not None:
                # 转回 BGR 再保存
                cropped_bgr = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, cropped_bgr)
            else:
                print(f"没检测到人脸: {filename}")


if __name__ == "__main__":
    from mtcnn import MTCNN
    detector = MTCNN()

    input_folder = "TMC_1222_4wei_huanlian2"   # 原始图片文件夹
    output_folder = "TMC_1222_4wei_koulian2_test" # 保存裁剪后人脸的文件夹

    process_images(input_folder, output_folder)
