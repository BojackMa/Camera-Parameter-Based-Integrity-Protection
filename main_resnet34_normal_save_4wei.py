# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup python main_resnet34_normal_save.py > main_resnet34_save_1103.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u main_resnet34_normal_save_4wei.py > main_resnet34_save_1222_4wei.txt 2>&1 &
import os
import csv
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from custom_resnet34 import resnet34

# ========== 测试集路径 ==========
TEST_CSV = 'TMC_1222_4wei_test/annotations.csv'
TEST_ROOT = 'TMC_1222_4wei_test/images'
# ========== 数据集路径 ==========
PTH_FILE_NAME = 'resnet34_tmc_1222_4wei.pth'
ROOT_DIR = 'TMC_1222_4wei/images'
CSV_FILE = 'TMC_1222_4wei/annotations.csv'
# ========== 保存路径 ============
SAVE_PATH = './save_1222_4wei'

def parse_csv(csv_path):
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            img1, img2, *delta = row
            delta = [float(d) for d in delta]  # 支持多维
            samples.append((img1, img2, delta))
    return samples

class ExposureChangeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.samples = parse_csv(csv_file)
        self.root = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
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
        delta = torch.tensor(delta, dtype=torch.float32)  # 3维
        return img1, img2, delta

class ConcatResNet34(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = resnet34(pretrained=pretrained)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512, 4)  # 输出改为3维
        )

    def forward(self, img1, img2):
        x = torch.cat([img1, img2], dim=1)  # 仍然 concat
        return self.backbone(x)



# ========== 测试集评估函数 ==========
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for img1, img2, delta in test_loader:
            img1, img2, delta = img1.to(device), img2.to(device), delta.to(device)
            outputs = model(img1, img2)
            loss = criterion(outputs, delta)
            total_loss += loss.item() * img1.size(0)
            count += img1.size(0)
    return total_loss / count


if __name__ == '__main__':

    batch_size = 512
    lr = 1e-4
    num_epochs = 1500

    # 数据集加载
    dataset = ExposureChangeDataset(CSV_FILE, ROOT_DIR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=min(32, os.cpu_count()))

    # 测试集加载（无数据增强）
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_dataset = ExposureChangeDataset(TEST_CSV, TEST_ROOT, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=min(16, os.cpu_count()))

    # 模型定义
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConcatResNet34(pretrained=True)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    try:
        for epoch in range(1, num_epochs + 1):
            model.train()
            running_loss = 0.0
            for img1, img2, delta in dataloader:
                img1, img2, delta = img1.to(device), img2.to(device), delta.to(device)
                optimizer.zero_grad()
                outputs = model(img1, img2)
                loss = criterion(outputs, delta)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * img1.size(0)

            epoch_loss = running_loss / len(dataset)
            print(f'[Train] Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}')

            # 每100个epoch评估一次测试集
            if epoch % 10 == 0:
                test_loss = evaluate(model, test_loader, criterion, device)
                print(f'[Eval ] Epoch [{epoch}], Test Loss: {test_loss:.4f}')
                # 保存模型
                save_name = SAVE_PATH + f'/resnet34_TMC_epoch{epoch}_testloss{test_loss:.4f}.pth'
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), save_name)
                else:
                    torch.save(model.state_dict(), save_name)
                print(f'✅ 模型已保存为 {save_name}')

        # 最终模型保存
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), PTH_FILE_NAME)
        else:
            torch.save(model.state_dict(), PTH_FILE_NAME)
    except KeyboardInterrupt:
        print("⛔️ 手动中断训练，正在保存模型...")
        torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), PTH_FILE_NAME)
        print("✅ 模型已保存为 " + PTH_FILE_NAME)
