# CUDA_VISIBLE_DEVICES=4,5,6,7 python -u ./test_resnet34_normal_4wei.py
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
from main_resnet34_normal_save_4wei import ConcatResNet34  # 确保导入正确
from main_resnet34_normal_save_4wei import parse_csv
import os
import csv
import pandas as pd
from collections import Counter
import numpy as np

gt_counter = Counter()
gt_all_counter =Counter()


def evaluate_vector_classification(preds, gts):
    preds = np.asarray(preds, dtype=np.float32)
    gts = np.asarray(gts, dtype=np.float32)
    assert preds.shape == gts.shape, "preds 与 gts 形状必须相同"

    # ===== 1️⃣ 映射到最近的 {-1, 0, 1} =====
    choices = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    idx = np.argmin(np.abs(preds[..., None] - choices[None, None, :]), axis=-1)
    preds_discrete = choices[idx]

    # ===== 2️⃣ 每维准确率 =====
    per_dim_correct = (preds_discrete == gts).astype(np.float32)
    per_dim_acc_each = per_dim_correct.mean(axis=0)
    per_dim_acc_mean = per_dim_acc_each.mean()

    # ===== 3️⃣ 严格准确率（全部四维都对） =====
    exact_match_acc = (per_dim_correct.prod(axis=1).mean())

    # ===== 4️⃣ 平均错维数 =====
    mean_wrong_dims = np.mean(np.sum(preds_discrete != gts, axis=1))

    # ===== 5️⃣ 每一维的错误分布（-1、0、1 各错多少次） =====
    num_dims = preds_discrete.shape[1]
    per_dim_error_detail = []

    for d in range(num_dims):
        # 该维度的预测和标签
        pd = preds_discrete[:, d]
        gt = gts[:, d]

        # 只统计错的
        wrong_idx = (pd != gt)

        # 错误值的分布
        unique, counts = np.unique(pd[wrong_idx], return_counts=True)

        # 存成 dict：{-1: x, 0: y, 1: z}
        error_count = {k: 0 for k in [-1.0, 0.0, 1.0]}
        for u, c in zip(unique, counts):
            error_count[float(u)] = int(c)

        per_dim_error_detail.append(error_count)

    return {
        "per_dim_acc_each": per_dim_acc_each.tolist(),
        "per_dim_acc_mean": float(per_dim_acc_mean),
        "exact_match_acc": float(exact_match_acc),
        "mean_wrong_dims": float(mean_wrong_dims),
        "per_dim_error_detail": per_dim_error_detail,   # ⭐ 新增字段
    }





class ExposureChangeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.samples = parse_csv(csv_file)
        self.root = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # 保持和训练一致的尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img1_path, img2_path, delta = self.samples[idx]
        img1 = Image.open(os.path.join(self.root, img1_path)).convert('RGB')
        img2 = Image.open(os.path.join(self.root, img2_path)).convert('RGB')
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2, torch.tensor(delta, dtype=torch.float32) 

def find_high_error_samples(preds_raw, gts, samples):
    choices = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    idx = np.argmin(np.abs(preds_raw[..., None] - choices[None, None, :]), axis=-1)
    preds_discrete = choices[idx]

    error_3 = []
    error_4 = []

    for i in range(len(gts)):
        wrong_dims = np.sum(preds_discrete[i] != gts[i])

        if wrong_dims in (3, 4):
            img1_path, img2_path, _ = samples[i]

            record = {
                "before": img1_path,
                "after": img2_path,
                "gt": gts[i].tolist(),
                "pred_raw": preds_raw[i].tolist(),
                "pred_discrete": preds_discrete[i].tolist(),
                "wrong_dims": int(wrong_dims)
            }

            if wrong_dims == 3:
                error_3.append(record)
            else:
                error_4.append(record)

    return error_3, error_4



def test_model(model_path, test_csv, test_root, output_csv, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载测试数据集
    test_dataset = ExposureChangeDataset(test_csv, test_root, transform=None)
    samples = test_dataset.samples
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 加载模型
    model = ConcatResNet34(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    preds = []
    gts = []
    with torch.no_grad():
        for img1, img2, delta in test_loader:
            img1, img2 = img1.to(device), img2.to(device)
            output = model(img1, img2).cpu().numpy()  # [batch, 3]
            gt = delta.numpy()                        # [batch, 3]
            preds.extend(output.tolist())
            gts.extend(gt.tolist())

    preds = np.array(preds)
    gts = np.array(gts)

    # MSE
    mse = np.mean((preds - gts) ** 2)
    print(f"\n✅ Test MSE: {mse:.4f}")

    # 计算欧氏距离准确率
    threshold = 1
    distances = np.linalg.norm(preds - gts, axis=1)  # 每个样本的向量差
    correct = np.sum(distances < threshold)
    total = len(gts)
    acc = correct / total * 100
    print(f"🎯 欧氏距离准确率: {acc:.2f}%  ({correct}/{total})")



    tol = 0.3  # 容差范围
    sign_equal = np.sign(preds) == np.sign(gts)
    zero_close = ((preds == 0) & (np.abs(gts) <= tol)) | ((gts == 0) & (np.abs(preds) <= tol))
    same_sign = np.all(sign_equal | zero_close, axis=1)

    # same_sign = np.all(np.sign(preds) == np.sign(gts), axis=1)
    correct = np.sum(same_sign)
    total = len(gts)
    acc = correct / total * 100
    # print(f"🎯 准确率: {acc:.2f}%  ({correct}/{total})")
    # 保存预测结果
    # with open(output_csv, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['index', 'prediction', 'ground_truth', 'distance'])
    #     for idx, (p, gt, dist) in enumerate(zip(preds, gts, distances)):
    #         writer.writerow([idx, p.tolist(), gt.tolist(), dist])



    metrics = evaluate_vector_classification(preds, gts)
    for k, v in metrics.items():
        print(f"{k}: {v}")

    error_3, error_4 = find_high_error_samples(preds, gts, samples)

    print("\n🚨 错 3 维样本（文件名）")
    for e in error_3[:5]:
        print(
            f"before={e['before']} | after={e['after']} | "
            f"GT={e['gt']} | "
            f"Pred(raw)={[round(x,3) for x in e['pred_raw']]} | "
            f"Pred(discrete)={e['pred_discrete']}"
        )

    print("\n🔥 错 4 维样本（文件名）")
    for e in error_4[:5]:
        print(
            f"before={e['before']} | after={e['after']} | "
            f"GT={e['gt']} | "
            f"Pred(raw)={[round(x,3) for x in e['pred_raw']]} | "
            f"Pred(discrete)={e['pred_discrete']}"
        )


    return metrics


if __name__ == '__main__':

    # for i in ["CCG_0815_2"]:
    #     TEST_CSV = i +"/annotations.csv"
    #     TEST_ROOT = i + "/images"
    #     MODEL_PATH = './save_0815/resnet34_ccg_epoch800_testloss0.0551.pth'
    #     output_csv = 'prediction_results_resnet34_koulian.csv'
    #     test_model(MODEL_PATH, TEST_CSV, TEST_ROOT, output_csv)


    # for i in ["CCG_0815_cjz"]:
    #     TEST_CSV = i +"/annotations.csv"
    #     TEST_ROOT = i + "/images"
    #     MODEL_PATH = './save_0815/resnet34_ccg_epoch800_testloss0.0551.pth'
    #     # MODEL_PATH = 'resnet34_normal_0724.pth'
    #     output_csv = 'prediction_results_resnet34_koulian.csv'
    #     test_model(MODEL_PATH, TEST_CSV, TEST_ROOT, output_csv)

    # for i in ["CCG_0911_test","CCG_0911_3"]:
    #     TEST_CSV = i +"/annotations.csv"
    #     TEST_ROOT = i + "/images"
    #     MODEL_PATH = './save_0911/resnet34_ccg_epoch300_testloss0.0010.pth'
    #     # MODEL_PATH = 'resnet34_normal_0724.pth'
    #     output_csv = 'prediction_results_resnet34_koulian.csv'
    #     test_model(MODEL_PATH, TEST_CSV, TEST_ROOT, output_csv)


# CCG_0911_test_huanlian_koulian

    # TEST_CSV = "TMC_1126_4wei_test/annotations.csv"
    # TEST_ROOT = "TMC_1126_4wei_test/images"
    # MODEL_PATH = 'save_1126_4wei/resnet34_TMC_epoch1200_testloss0.0492.pth'
    # # MODEL_PATH = 'resnet34_normal_0724.pth'
    # output_csv = 'prediction_results_resnet34_koulian.csv'
    # test_model(MODEL_PATH, TEST_CSV, TEST_ROOT, output_csv)


    # TEST_CSV = "TMC_1103_test/annotations1.csv"
    # TEST_ROOT = "TMC_1103_test_koulian"
    # MODEL_PATH = './save_1104/resnet34_TMC_epoch1050_testloss0.5058.pth'
    # # MODEL_PATH = 'resnet34_normal_0724.pth'
    # output_csv = 'prediction_results_resnet34_koulian.csv'
    # test_model(MODEL_PATH, TEST_CSV, TEST_ROOT, output_csv)



    TEST_CSV = "TMC_1222_4wei_nokoulian/annotations.csv"
    # TEST_ROOT = "TMC_1222_4wei_test/images"
    TEST_ROOT = "TMC_1222_4wei_koulian2_test"
    MODEL_PATH = 'save_1222_4wei/resnet34_TMC_epoch1440_testloss0.0440.pth'
    # MODEL_PATH = 'resnet34_normal_0724.pth'
    output_csv = 'prediction_results_resnet34_koulian.csv'
    test_model(MODEL_PATH, TEST_CSV, TEST_ROOT, output_csv)


def evaluate_all_checkpoints(
    model_dir,
    test_csv,
    test_root,
    batch_size=64
):
    results = []

    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith(".pth"):
            continue

        model_path = os.path.join(model_dir, fname)
        print(f"\n🚀 Testing {fname}")

        metrics = test_model(
            model_path=model_path,
            test_csv=test_csv,
            test_root=test_root,
            output_csv=None,
            batch_size=batch_size
        )

        per_dim = np.array(metrics["per_dim_acc_each"])
        record = {
            "model": fname,
            "per_dim_acc_each": per_dim,
            "per_dim_acc_mean": metrics["per_dim_acc_mean"],
            "exact_match_acc": metrics["exact_match_acc"],
            "min_dim_acc": per_dim.min(),   # ⭐ 最关键指标
        }
        results.append(record)

    return results





def select_best_model(results):
    # 先按 最差维度准确率 排序，再按 平均准确率
    results_sorted = sorted(
        results,
        key=lambda x: (x["min_dim_acc"], x["per_dim_acc_mean"]),
        reverse=True
    )

    best = results_sorted[0]

    print("\n🏆 Best Model Selected")
    print(f"Model: {best['model']}")
    print(f"Per-dim acc: {best['per_dim_acc_each']}")
    print(f"Min dim acc: {best['min_dim_acc']:.4f}")
    print(f"Mean acc: {best['per_dim_acc_mean']:.4f}")
    print(f"Exact match acc: {best['exact_match_acc']:.4f}")

    return best, results_sorted


def predict_4wei(model_path,before_img_path,after_img_path,device=None,discrete=True):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ===== 1️⃣ 和训练 / 测试一致的 transform =====
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ===== 2️⃣ 加载图片 =====
    before_img = Image.open(before_img_path).convert("RGB")
    after_img  = Image.open(after_img_path).convert("RGB")

    before_tensor = transform(before_img).unsqueeze(0).to(device)  # [1,3,224,224]
    after_tensor  = transform(after_img).unsqueeze(0).to(device)

    # ===== 3️⃣ 加载模型 =====
    model = ConcatResNet34(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # ===== 4️⃣ 前向推理 =====
    with torch.no_grad():
        pred = model(before_tensor, after_tensor)  # [1,4]
        pred = pred.cpu().numpy()[0]               # (4,)

    # ===== 5️⃣ 映射到 {-1, 0, 1}（可选） =====
    if discrete:
        choices = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        idx = np.argmin(np.abs(pred[:, None] - choices[None, :]), axis=1)
        pred_discrete = choices[idx]
        return pred, pred_discrete

    return pred






if __name__ == '__main__':

    # TEST_CSV = "TMC_1222_4wei_test/annotations.csv"
    # TEST_ROOT = "TMC_1222_4wei_test/images"
    # MODEL_DIR = "./save_1222_4wei"

    # results = evaluate_all_checkpoints(
    #     model_dir=MODEL_DIR,
    #     test_csv=TEST_CSV,
    #     test_root=TEST_ROOT,
    #     batch_size=64
    # )

    # best, all_sorted = select_best_model(results)



# before=img_00409_before.jpg | after=img_00409_after.jpg | GT=[0.0, 1.0, -1.0, -1.0] | Pred(raw)=[0.014, 0.298, -0.04, -0.212] | Pred(discrete)=[0.0, 0.0, 0.0, 0.0]
# before=img_00458_before.jpg | after=img_00458_after.jpg | GT=[-1.0, 1.0, -1.0, 1.0] | Pred(raw)=[-0.999, 0.011, -0.484, 0.015] | Pred(discrete)=[-1.0, 0.0, 0.0, 0.0]


# img_00386_before.jpg,img_00386_after.jpg,0.0,1.0,-1.0,-1.0
# img_00431_before.jpg,img_00431_after.jpg,-1.0,1.0,-1.0,1.0


    MODEL_PATH = "save_1222_4wei/resnet34_TMC_epoch1440_testloss0.0440.pth"
# img_00007_before.jpg,img_00007_after.jpg
    before_img = "TMC_1222_4wei_huanlian2/img_00458_before.jpg"
    after_img  = "TMC_1222_4wei_huanlian2/img_00458_after.jpg"

    before_img = "TMC_1222_4wei_test/images/img_00431_before.jpg"
    before_img = "TMC_1222_4wei_test/images/img_00431_after.jpg"

    # before_img = "./1.jpg"
    # after_img  = "./0.jpg"

    raw_pred, discrete_pred = predict_4wei(
        MODEL_PATH,
        before_img,
        after_img
    )

    raw_pred_str = "[" + ", ".join(f"{x:.4f}" for x in raw_pred) + "]"
    disc_pred_str = "[" + ", ".join(f"{int(x)}" for x in discrete_pred) + "]"

    print("Raw prediction:", raw_pred_str)
    print("Discrete prediction:", disc_pred_str)
