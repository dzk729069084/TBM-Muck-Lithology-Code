import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm

# ----------------------
# 配置参数
# ----------------------
train_data_dir = r"G:\key_data\2000x2000\2_8\2"  # 需替换为实际路径
test_data_dir = r"G:\key_data\2000x2000\2_8\8"   # 需替换为实际路径
num_classes = 6
batch_size = 32
num_epochs = 200
learning_rate = 1e-3
result_csv = "training_metrics.csv"
confusion_matrix_csv = "confusion_matrix.csv"

early_stop_patience = 10
best_model_path = "best_rock_model.pth"

if __name__ == '__main__':
    # ----------------------
    # 设备配置
    # ----------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ----------------------
    # 数据预处理
    # ----------------------
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_data_dir, train_transform)
    test_dataset = datasets.ImageFolder(test_data_dir, test_transform)
    class_names = train_dataset.classes

    # 注意：tqdm(DataLoader(...)) 这种写法会让同一个 tqdm 在多个 epoch 重复用时表现怪异
    # 这里仍然保持你的写法，但更推荐每个 epoch 内再包 tqdm（你若需要我也可帮你改）
    train_loader_tqdm = tqdm(DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4))
    test_loader_tqdm  = tqdm(DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4))

    # ----------------------
    # 模型定义
    # ----------------------
    model = torchvision.models.resnet50(weights=None)
    # model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # ----------------------
    # 损失函数与优化器
    # ----------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ----------------------
    # 训练循环（早停+保存）
    # ----------------------
    # 🌟 增加两列：Silhouette、DBI（训练阶段先留空，最终 test 时填）
    results = pd.DataFrame(columns=["Epoch", "Time", "Loss", "Accuracy", "F1", "Silhouette", "DaviesBouldin"])
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        # ===== 训练阶段 =====
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)

        # ===== 验证/测试阶段（你这里用 test_loader 当验证集）=====
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        epoch_time = time.time() - start_time

        # ===== 早停与保存 =====
        if f1 > best_f1:
            print(f"[Epoch {epoch + 1}] 🚀 F1 improved {best_f1:.4f} → {f1:.4f}")
            torch.save(model.state_dict(), best_model_path)
            best_f1 = f1
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"[Epoch {epoch + 1}] ⏳ No improvement ({patience_counter}/{early_stop_patience})")
            if patience_counter >= early_stop_patience:
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                break

        # 记录结果（训练过程不算 silhouette/DBI，先写 NaN）
        new_row = pd.DataFrame([[
            epoch + 1,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            epoch_loss,
            accuracy,
            f1,
            np.nan,
            np.nan
        ]], columns=results.columns)
        results = pd.concat([results, new_row], ignore_index=True)

        if (epoch + 1) % 10 == 0:
            results.to_csv(result_csv, index=False)
            print(f"[Epoch {epoch + 1}] 💾 Checkpoint saved")

        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {accuracy:.4f} | "
              f"F1: {f1:.4f} | Time: {epoch_time:.2f}s")

    # ===== 最终处理 =====
    results.to_csv(result_csv, index=False)

    # ----------------------
    # 🌟 加载最优模型 + 最终评估 + 计算 Silhouette / DBI
    # ----------------------
    print("\n🔍 Loading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # 用 hook 提取 fc 前的特征（resnet50: avgpool 后 flatten 的 2048 维）
    features_list = []

    def _feat_hook(module, inp, out):
        # inp 是一个 tuple，inp[0] shape: (B, 2048)
        x = inp[0].detach()
        features_list.append(x.cpu())

    hook_handle = model.fc.register_forward_hook(_feat_hook)

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # forward 时 hook 会自动把特征存进 features_list

            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 关闭 hook
    hook_handle.remove()

    test_loss = total_loss / len(test_dataset)
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="weighted")

    # 拼接特征 (N, 2048)
    X = torch.cat(features_list, dim=0).numpy()
    y = np.array(all_labels)

    # 标准化后计算距离类指标
    X_scaled = StandardScaler().fit_transform(X)

    # Silhouette：要求至少 2 个类别且每类不为空
    # DBI：同样要求至少 2 个类别
    unique_labels = np.unique(y)
    if unique_labels.size >= 2:
        sil = silhouette_score(X_scaled, y, metric="euclidean")
        dbi = davies_bouldin_score(X_scaled, y)
    else:
        sil = np.nan
        dbi = np.nan
        print("⚠️ Cannot compute Silhouette/DBI: need at least 2 classes in y.")

    # 写入 results 的 test 行
    test_row = pd.DataFrame([[
        "test",
        "test",
        test_loss,
        test_accuracy,
        test_f1,
        sil,
        dbi
    ]], columns=results.columns)
    results = pd.concat([results, test_row], ignore_index=True)
    results.to_csv(result_csv, index=False)

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(confusion_matrix_csv, encoding="utf-8-sig")

    print(f"✅ Final Test Results - Loss: {test_loss:.4f} | Acc: {test_accuracy:.4f} | F1: {test_f1:.4f}")
    print(f"✅ Feature Separability - Silhouette: {sil:.4f} (higher is better) | DBI: {dbi:.4f} (lower is better)")
    print(f"✅ Results saved: {result_csv}, {confusion_matrix_csv}")
