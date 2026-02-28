
####改进：
####1.训练结果和测试集合保存至指定文件；
####2.输入参数整合在code开始
####3.保存训练时间

import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from datetime import datetime
from simclr import SimCLR
from simclr.modules import LogisticRegression, get_resnet      ###LogisticRegression???
from simclr.modules.transformations import TransformsSimCLR
import torchvision.datasets as datasets
from utils import yaml_config_hook
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import time
import random
path_matrix = r'G:\岩性SimCLR模型\SimCLR_code\Results\embedding\confusion_matrix4.csv'
path_result = r"G:\岩性SimCLR模型\SimCLR_code\Results\embedding\results4.csv"
down_train = r'G:\key_data\200x200\2_8\8'
down_test = r'G:\key_data\200x200\2_8\2'
feature_path = r'G:\岩性SimCLR模型\SimCLR_code\Results\embedding\tsne_features.npz'
model_pth="checkpoint_best(4000+20%).tar"

def inference(loader, simclr_model, device):               #获取XY
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        # get encoding
        with torch.no_grad():
            h, _, z, _ = simclr_model(x, x)
        h = h.detach()


        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 1 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)

    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):   #将特征数据转换为PyTorch的DataLoader
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def train(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    predicted_labels = []
    true_labels = []
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(y.cpu().numpy())

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )
    accuracy_overall = accuracy_score(true_labels, predicted_labels)
    F1 = f1_score(true_labels, predicted_labels, average='weighted')
    return loss_epoch, accuracy_overall,F1


# def test(args, loader, simclr_model, model, criterion, optimizer,path_matrix):
#     loss_epoch = 0
#     model.eval()
#     predicted_labels = []
#     true_labels = []
#
#     for step, (x, y) in enumerate(loader):
#         model.zero_grad()
#         x = x.to(args.device)
#         y = y.to(args.device)
#         output = model(x)
#         loss = criterion(output, y)
#         predicted = output.argmax(1)
#         predicted_labels.extend(predicted.cpu().numpy())
#         true_labels.extend(y.cpu().numpy())
#         loss_epoch += loss.item()
#     accuracy_overall = accuracy_score(true_labels, predicted_labels)
#     F1 = f1_score(true_labels, predicted_labels, average='weighted')
#     # 计算混淆矩阵
#     confusion_mat = confusion_matrix(true_labels , predicted_labels)
#     # 将混淆矩阵转换为DataFrame
#     confusion_df = pd.DataFrame(confusion_mat)
#
#     # 保存DataFrame为CSV文件
#     confusion_df.to_csv(path_matrix, mode='a',index=False)
#
#     return loss_epoch, accuracy_overall,F1

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def test(args, loader, simclr_model, model, criterion, optimizer, path_matrix, feat_save_path=None):
    """
    增强版 test：
    - 维持原有功能：计算 loss / acc / F1、并保存混淆矩阵
    - 新增功能：收集所有样本的特征向量 + 标签，并可选保存到文件，用于 t-SNE 可视化
    """

    loss_epoch = 0.0
    model.eval()
    if simclr_model is not None:
        simclr_model.eval()

    predicted_labels = []
    true_labels = []

    all_features = []   # 用于保存每个样本的特征向量
    all_targets = []    # 用于保存对应的真实标签

    with torch.no_grad():
        for step, (x, y) in enumerate(loader):
            x = x.to(args.device)
            y = y.to(args.device)

            # ------- 1. 正常分类前向 + 损失 -------
            output = model(x)                  # 分类输出 (logits)
            loss = criterion(output, y)

            predicted = output.argmax(1)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(y.cpu().numpy())
            loss_epoch += loss.item()

            # ------- 2. 提取用于 t-SNE 的特征 -------
            # 你可以有两种选择：
            # (A) 用 SimCLR 的特征嵌入（推荐）
            # (B) 用当前分类模型的某层输出（比如 logits）

            if simclr_model is not None:
                # 假设 simclr_model 有 encoder，可以根据你的代码改成 backbone / f / encoder 等
                # h = simclr_model.encoder(x)          # [N, D]
                # 如果你的 simclr_model 直接前向返回 (h, z)，也可以这样：
                # h, z = simclr_model(x)
                # 这里给出一个通用写法，你按自己模型结构替换：
                h = simclr_model.encoder(x)           # TODO: 按你的实际模型改名字
                features = h
            else:
                # 如果不想用 simclr_model，就先用 logits 作为特征（也能画 t-SNE，只是语义略差）
                features = output                     # [N, num_classes]

            all_features.append(features.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # ------- 3. 计算指标 -------
    accuracy_overall = accuracy_score(true_labels, predicted_labels)
    F1 = f1_score(true_labels, predicted_labels, average='weighted')

    # ------- 4. 保存混淆矩阵到 CSV -------
    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    confusion_df = pd.DataFrame(confusion_mat)
    confusion_df.to_csv(path_matrix, mode='a', index=False)

    # ------- 5. 合并并可选保存特征，用于 t-SNE -------
    all_features = np.concatenate(all_features, axis=0)  # [N_total, D]
    all_targets = np.concatenate(all_targets, axis=0)    # [N_total]

    if feat_save_path is not None:
        # 推荐保存为 npz：包含 features 和 labels，后续直接加载做 t-SNE
        np.savez(feat_save_path,
                 features=all_features,
                 labels=all_targets)

        # 如果你更喜欢 CSV，也可以这样（但 CSV 会比较大，这里注释给你参考）：
        # df_feat = pd.DataFrame(all_features)
        # df_feat["label"] = all_targets
        # df_feat.to_csv(feat_save_path, index=False)

    return loss_epoch, accuracy_overall, F1


def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

if __name__ == "__main__":##用目标数据进行微调，先尝试少量有标签数据来训练，剩下的测试






    #setup_seed(3407)

    start_time1 = time.time()  # 记录epoch开始时间

    #############参数配置与设备初始化##########################
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("args.device",args.device)


    #############数据加载与预处理##########################
    if args.dataset == "STL10":
        # train_dataset = torchvision.datasets.STL10(
        #     args.dataset_dir,
        #     split="train",
        #     download=True,
        #     transform=TransformsSimCLR(size=args.image_size).test_transform,
        # )
        train_dataset = datasets.ImageFolder(down_train, TransformsSimCLR(size=224).test_transform)   ##TransformsSimCLR？？？
        # test_dataset = torchvision.datasets.STL10(
        #     args.dataset_dir,
        #     split="test",
        #     download=True,
        #     transform=TransformsSimCLR(size=args.image_size).test_transform,
        # )
        test_dataset = datasets.ImageFolder(down_test, TransformsSimCLR(size=224).test_transform)
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(args.dataset_dir,train=True,download=True,transform=TransformsSimCLR(size=args.image_size).test_transform,)
        test_dataset = torchvision.datasets.CIFAR10(args.dataset_dir,train=False,download=True,transform=TransformsSimCLR(size=args.image_size).test_transform,)
    else:
        raise NotImplementedError






    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.logistic_batch_size,shuffle=True,drop_last=True,num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.logistic_batch_size,shuffle=False,drop_last=False,num_workers=args.workers,)


    #############预训练模型加载##########################
    encoder = get_resnet(args.resnet, pretrained=False)    ####没有发现resnet的tar    # 不加载ImageNet预训练
    n_features = encoder.fc.in_features  # get dimensions of fc layer  # 提取特征维度

    # load pre-trained model from checkpoint
    simclr_model = SimCLR(encoder, args.projection_dim, n_features)    #????
    model_fp = os.path.join(args.model_path,model_pth)          # 加载自监督预训练权重
    simclr_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    simclr_model = simclr_model.to(args.device)
    simclr_model.eval()    #冻结编码器  # 冻结参数


    #############下游分类器构建##########################
    ## Logistic Regression
    n_classes = 6  # CIFAR-10 / STL-10
    model = LogisticRegression(simclr_model.n_features, n_classes)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    #############特征提取与数据重组##########################
    (train_X, train_y, test_X, test_y) = get_features(simclr_model, train_loader, test_loader, args.device)
    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(train_X, train_y, test_X, test_y, args.logistic_batch_size)
    #csv_path=r"G:\岩性SimCLR模型\SimCLR_code\Results\03280.2tune_train_index.csv"   #混淆矩阵路径

    for epoch in range(args.logistic_epochs):
        start_time = time.time()  # 记录epoch开始时间
        loss_epoch, accuracy_overall,f1_tr = train(args, arr_train_loader, simclr_model, model, criterion, optimizer)
        epoch_time = time.time() - start_time
        epoch_time = round(epoch_time, 2)

        loss_epoch = loss_epoch / len(arr_train_loader)
        #print(f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_loader)}\t F1:{f1_tr}")
        #print("accuracy_overall",accuracy_overall)
        #accuracy_epoch = accuracy_epoch / len(arr_train_loader)
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        step = f"Step[{epoch}]"
        row = [time_str, step,epoch_time, loss_epoch, accuracy_overall, f1_tr]
        df = pd.DataFrame([row], columns=["Time", "Step","epoch_time", "Loss","Accuracy", "F1"])
        df.to_csv(path_result, mode='a', header=not os.path.exists(path_result), index=False)


    # final testing
    loss_epoch, accuracy_overall,f1 = test(args, arr_test_loader, simclr_model, model, criterion, optimizer,path_matrix)
    py_time = time.time()-start_time1
    loss_epoch = loss_epoch / len(arr_test_loader)
    print(f"[FINAL]\t Loss: {loss_epoch}\t Accuracy: {accuracy_overall }\t F1:{f1}")
    #print("accuracy_overall",accuracy_overall)

    row_final= ["test","test",py_time, loss_epoch, accuracy_overall, f1]
    df = pd.DataFrame([row_final])
    df.to_csv(path_result, mode='a',header=not os.path.exists(path_result), index=False)
