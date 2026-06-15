# nohup python handledata_normal2.py > log_handledata_normal2_0731.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup python handledata_TMC_koulian.py > log.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=5,6,7 nohup python -u handledata_TMC_koulian_4wei_1222_nokoulian.py > log1222_4wei_nokoulian.txt 2>&1 &
import cv2
import os
import numpy as np
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from mtcnn import MTCNN
from PIL import Image
# 初始化人脸检测器（只初始化一次）
detector = MTCNN()
print("detector初始化完毕")
random.seed(42)  # 任意固定整数
count = 1
delta_counter = defaultdict(float) 
dim_counters = [Counter() for _ in range(12)]

log_lines = []
output_dir = "TMC_1222_4wei_nokoulian"
output_dir1 = os.path.join(output_dir,"images")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir1, exist_ok=True)

def quantize_delta(diff, threshold=0.02):
    if diff >= threshold:
        return 0.1
    elif diff <= -threshold:
        return -0.1
    else:
        return 0.0

def process_pair(i, timestamps_exposure, frame_list):
    prev_row = timestamps_exposure[i - 1]
    curr_row = timestamps_exposure[i]
    delta = [0]*4

    flag = 0

    deltat = 0
    for k in range(6):
        deltat +=  quantize_delta(curr_row[k+1] - prev_row[k+1])
    delta[0] = quantize_delta(deltat,0.1) * 10 
    if (delta[0] != 0 ):
        flag = 1

    
    deltat = 0
    for k in range(3):
        deltat +=  quantize_delta(curr_row[k+1+6] - prev_row[k+1+6])
    delta[1] = quantize_delta(deltat,0.1) * 10 
    if (delta[1] != 0 ):
        flag = 1
    
    deltat = 0
    for k in range(3):
        deltat +=  quantize_delta(curr_row[k+1+9] - prev_row[k+1+9])
    delta[2] = quantize_delta(deltat,0.1) * 10 
    if (delta[2] != 0 ):
        flag = 1
    
    deltat = 0
    for k in range(7):
        deltat +=  quantize_delta(curr_row[k+1+12] - prev_row[k+1+12])
    delta[3] = quantize_delta(deltat,0.1) * 10 
    if (delta[3] != 0 ):
        flag = 1


    if (flag == 1 or random.random() < 0.01):
        prev_frame = frame_list[i - 1]
        curr_frame = frame_list[i]

        prev_face = detect_and_crop_face(prev_frame)
        curr_face = detect_and_crop_face(curr_frame)

        if prev_face is not None and curr_face is not None:
            return (i, prev_face, curr_face, delta)
    return None



def detect_and_crop_face(image_bgr):
    """
    检测人脸并返回裁剪后的人脸RGB图像（numpy数组）。若未检测到返回 None。
    """
    rotated = cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rgb_image = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    # faces = detector.detect_faces(rgb_image)

    # if faces:
    #     x, y, w, h = faces[0]['box']
    #     x, y = max(0, x), max(0, y)
    #     cropped = rgb_image[y:y+h, x:x+w]
    #     return cropped
    # else:
    #     print("没找到人脸")
    #     # input()
    #     return None
    return rgb_image



# ---------- 参数 ----------
def func(video_path,txt_path):
    global count,log_lines ,delta_counter,dim_counters


    
    # ---------- 步骤1: 读取 txt ----------
    timestamps_exposure = []
    num_1000000 = 0
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) == 2:
                ts, exp = parts
                timestamps_exposure.append((int(ts), int(exp)))
                num_1000000 = num_1000000 + 1
            else:
                tmp = []
                tmp.append(int(parts[0]))
                for i in range(1,len(parts)):
                    tmp.append(float(parts[i]))
                timestamps_exposure.append(tmp)


    # ---------- 步骤2: 丢弃第一个1000000以前的数据 ----------
    def trim_txt_to_first_1000000(data):
        for i,exp in enumerate(data):
            if exp[1] == 1000000:
                return data[i:]
        return []

    timestamps_exposure = trim_txt_to_first_1000000(timestamps_exposure)
    print(f"Trimmed TXT 总长度: {len(timestamps_exposure)}")

    # ---------- 步骤3: 读取视频帧 ----------
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_list.append(frame)
    cap.release()
    total_frames = len(frame_list)
    print(f"原始视频帧数: {total_frames}")

    # ---------- 步骤3.1: 找最暗的8帧，保留其中index最小的帧为起点 ----------
    def compute_brightness(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)

    half_len = total_frames // 2
    brightness_scores = [(i, compute_brightness(frame_list[i])) for i in range(half_len)]
    brightness_scores.sort(key=lambda x: x[1])
    darkest_8 = brightness_scores[:8]
    min_dark_index = min(idx for idx, _ in darkest_8)
    print(darkest_8)
    print(min_dark_index)
    # ---------- 步骤3.5: 统计有多少个1000000，并同步丢弃前x帧和前x个txt行 ----------
    
    print(f"TXT 中曝光值为1000000的数量: {num_1000000}")

    frame_trim_start = min_dark_index + num_1000000
    txt_trim_start = num_1000000

    frame_list = frame_list[frame_trim_start:]
    timestamps_exposure = timestamps_exposure[txt_trim_start:]

    print(f"同步丢弃前 {num_1000000} 个帧和行后：")
    print(f"视频帧剩余: {len(frame_list)}, TXT 剩余: {len(timestamps_exposure)}")

    # ---------- 步骤4: 曝光变化检测 ----------
    aligned_count = min(len(frame_list), len(timestamps_exposure))
    print("aligned_count",aligned_count)


    with ThreadPoolExecutor(max_workers=8) as executor:  # 根据你机器的线程数可调
        futures = [
            executor.submit(process_pair, i, timestamps_exposure, frame_list)
            for i in range(1, aligned_count)
        ]

        for future in futures:
            result = future.result()
            if result is not None:
                # print(result)
                i, prev_face, curr_face, delta = result
                # delta_counter[tuple(delta)] += 1
                for i, v in enumerate(delta):
                    dim_counters[i][v] += 1

                fname = f"img_{count:05d}"
                Image.fromarray(prev_face).save(os.path.join(output_dir1, f"{fname}_before.jpg"))
                Image.fromarray(curr_face).save(os.path.join(output_dir1, f"{fname}_after.jpg"))
                strtmp = f"{fname}_before.jpg,{fname}_after.jpg"
                for i in delta:
                    strtmp=strtmp+","+str(i)
                # log_lines.append(f"{fname}_before.jpg,{fname}_after.jpg,{delta1*10},{delta2*10},{delta4*10}")
                log_lines.append(strtmp)
                count += 1






def scan_file_and_handle(dcim_dir,txt_dir):
    for filename in os.listdir(dcim_dir):
        if filename.endswith('.mp4') and filename.startswith('VID_'):
            # 构造完整路径
            video_path = os.path.join(dcim_dir, filename)

            # 假设对应 txt 文件名为 xxx.txt，其中 xxx 是视频名（不带扩展名）
            base_name = os.path.splitext(filename)[0]  # 去除 .mp4
            txt_name = base_name[4:] + '.txt'
            txt_path = os.path.join(txt_dir, txt_name)

            if os.path.exists(txt_path):
                print("处理文件"+video_path+" "+txt_path)
                func(video_path, txt_path)
            else:
                print(f"找不到对应的文本文件: {txt_path}")


base_dir = os.path.abspath(os.path.dirname(__file__))


# dcim_dir = os.path.join(base_dir, 'DCIM_1222')
# txt_dir = os.path.join(base_dir, 'files_1222')

# scan_file_and_handle(dcim_dir,txt_dir)

func("VID_1766420410514.mp4","1766420410514.txt")



# # ---------- 步骤5: 保存日志 ----------
with open(os.path.join(output_dir, "annotations.csv"), 'w') as f:
    for line in log_lines:
        f.write(line + "\n")

# for delta, count in sorted(delta_counter.items(), reverse=True):
#     print(f"Delta: {delta}, Count: {count}")


for i, counter in enumerate(dim_counters):
    print(f"第 {i+1} 维：{dict(counter)}")


print(f"完成！共检测曝光变化 {len(log_lines)} 次。")
