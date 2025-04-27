import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from itertools import product
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter
from torchvision.models import resnet18, ResNet18_Weights
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import json

def get_top_k_classes(df, k=8):
    all_labels = []
    for labels in df['Finding Labels']:
        all_labels.extend([x.strip() for x in labels.split('|')])
    counter = Counter(all_labels)
    top_classes = [item[0] for item in counter.most_common(k)]
    return top_classes

# ================================
# 1️⃣ Dataset 类：仅增强小类图像
# ================================
class ChestXrayDataset(Dataset):
    def __init__(self, df, image_dir, class_list, augmentation='light', top_classes=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.class_list = class_list
        self.augmentation = augmentation
        self.top_classes = top_classes or []

        self.light_transform = transforms.Compose([
            transforms.RandomRotation(5),
        ])
        self.strong_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image Index'].replace('.png', '.pt')
        img_path = os.path.join(self.image_dir, img_name)
        image = torch.load(img_path)

        labels = str(row['Finding Labels']).split('|')
        enhance = not any(lbl.strip() in self.top_classes for lbl in labels)

        if self.augmentation == 'strong' and enhance:
            image = self.strong_transform(image)
        elif self.augmentation == 'light' and enhance:
            image = self.light_transform(image)
        # else: 不增强

        label = torch.zeros(len(self.class_list))
        for disease in labels:
            disease = disease.strip()
            if disease in self.class_list:
                label[self.class_list.index(disease)] = 1.0

        return image, label

# ================================
# 2️⃣ 数据划分：确保ID不重复 + 类别全覆盖
# ================================


def has_all_classes(df, class_list):
    present_classes = set()
    for labels in df['Finding Labels']:
        for label in labels.split('|'):
            present_classes.add(label.strip())
    return all(cls in present_classes for cls in class_list)


def split_train_eval(df, class_list, test_ratio=0.2, seed=42, max_tries=100):
    random.seed(seed)
    patient_ids = df['Patient ID'].unique().tolist()

    for _ in range(max_tries):
        random.shuffle(patient_ids)
        split_idx = int(len(patient_ids) * (1 - test_ratio))
        train_ids = set(patient_ids[:split_idx])
        test_ids = set(patient_ids[split_idx:])

        train_df = df[df['Patient ID'].isin(train_ids)].copy()
        test_df = df[df['Patient ID'].isin(test_ids)].copy()

        if has_all_classes(train_df, class_list) and has_all_classes(test_df, class_list):
            return train_df, test_df

    raise ValueError("无法在 max_tries 次随机中找到满足所有类别覆盖的训练/测试集划分")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target):
        bce_loss = self.bce(input, target)
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
def create_model(num_classes, dropout, fc_hidden_size, kernel_size, num_blocks):

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.layer4 = model.layer4[:num_blocks]
    model.conv1 = nn.Conv2d(3, 64, kernel_size=kernel_size,
                            stride=2, padding=kernel_size//2, bias=False)
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(512, fc_hidden_size),
        nn.ReLU(),
        nn.Linear(fc_hidden_size, num_classes)
    )

    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=False)

    return model

def get_optimizer(opt_name, model, lr, weight_decay):
    if opt_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

def train_and_validate(model, train_loader, val_loader, optimizer, criterion, device, early_stop=5):
    best_f1 = 0
    no_improve = 0
    best_metrics = {}
    history = {'f1': [], 'acc': [], 'loss': [],'recall': []}

    for epoch in range( early_stop):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float().cpu()
                all_preds.append(preds)
                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())


        y_true = torch.cat(all_labels, dim=0).numpy()
        y_pred = torch.cat(all_preds, dim=0).numpy()
        y_prob = torch.cat(all_probs, dim=0)

        f1 = f1_score(y_true, y_pred, average='micro')
        precision = precision_score(
            y_true, y_pred, average='micro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
        soft_acc = ((y_prob - torch.cat(all_labels, dim=0)).abs() < 0.5).float().mean().item()

        history['f1'].append(f1)
        #history['precision'].append(precision)
        history['recall'].append(recall)
        #history['loss'].append(val_loss / len(val_loader))
        history['acc'].append(soft_acc)

        print(f"Epoch {
              epoch+1} | F1: {f1:.4f} | Acc: {soft_acc:.4f} | Recall: {recall:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {'f1': f1, 'acc': soft_acc, 'recall': recall}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop:
                print("⏹️ 触发早停！")
                break

    return best_metrics, history


def run_all(csv_path, output_dir, df_train_path, model_path):
    # print("CUDA Available:", torch.cuda.is_available())
    # print("Number of GPUs:", torch.cuda.device_count())
    # print("Current Device:", torch.cuda.current_device())
    # print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device(
    #    "mps" if torch.backends.mps.is_available() else "cpu")
    full_df = pd.read_csv(df_train_path)
    class_list = sorted(set('|'.join(full_df['Finding Labels']).split('|')))

    train_df, temp_eval_df = split_train_eval(
        full_df, class_list, test_ratio=0.2)
    top_classes = get_top_k_classes(train_df, k=3)
    class_list = top_classes

    search_space = {
        'dropout': [0.3],
        'lr': [0.0006],
        'batch_size': [128],
        'optimizer': ['adam'],
        'kernel_size': [3],
        'num_blocks': [2],
        'fc_hidden_size': [1024],
        'weight_decay': [0.0005],
        'augmentation': ['light'],
        'loss_type': ['focal'], 
        'gamma': [1.5]  
    }

    keys, values = zip(*search_space.items())
    all_combinations = [dict(zip(keys, v)) for v in product(*values)]

    results = []
    best_model_metrics = None
    best_config = None
    best_f1 = 0
    best_history = None

    for config in all_combinations:
        print(f"\n🚀 正在训练配置: {config}")

        model = create_model(len(
            class_list), config['dropout'], config['fc_hidden_size'], config['kernel_size'], config['num_blocks']).to(device)
        optimizer = get_optimizer(
            config['optimizer'], model, config['lr'], config['weight_decay'])
        # criterion = nn.BCEWithLogitsLoss()
        if config['loss_type'] == 'focal':
            criterion = FocalLoss(gamma=config['gamma'])
        else:
            criterion = nn.BCEWithLogitsLoss()

        train_dataset = ChestXrayDataset(
            train_df, output_dir, class_list, augmentation=config['augmentation'], top_classes=top_classes)
        val_dataset = ChestXrayDataset(
            temp_eval_df, output_dir, class_list, augmentation='none')

        train_loader = DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

        metrics, history = train_and_validate(
            model, train_loader, val_loader, optimizer, criterion, device)
        row = {**config, **metrics}
        results.append(row)

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_config = config
            best_model_metrics = metrics
            best_history = history
            torch.save(model.state_dict(), model_path)
            print("✅ 已保存当前最佳模型")

    df_result = pd.DataFrame(results)
    df_result.to_csv("grid_search_results.csv", index=False)
    print("\n🏁 所有训练完成！")
    print("🥇 最佳配置:", best_config)
    print("📊 指标:", best_model_metrics)

def train_and_validate_with_dp(model, train_loader, val_loader, optimizer, criterion, config, device, early_stop=5):
    from opacus import PrivacyEngine
    privacy_engine = PrivacyEngine(
        secure_mode=config.get("secure_mode", False)
    )
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=config["noise_multiplier"],
        max_grad_norm=config["max_grad_norm"]
    )

    best_f1 = 0
    no_improve = 0
    best_metrics = {}
    history = {'f1': [], 'precision': [], 'recall': [], 'loss': [], 'soft_acc': [], 'epsilon': []}

    for epoch in range(early_stop):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float().cpu()
                all_preds.append(preds)
                all_labels.append(labels.cpu())

        y_true = torch.cat(all_labels, dim=0).numpy()
        y_pred = torch.cat(all_preds, dim=0).numpy()

        from sklearn.metrics import f1_score, precision_score, recall_score
        f1 = f1_score(y_true, y_pred, average='micro')
        precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
        soft_acc = ((torch.cat(all_preds, dim=0) - torch.cat(all_labels, dim=0)).abs() < 0.5).float().mean().item()

        try:
            epsilon = privacy_engine.get_epsilon(delta=1e-5)
        except Exception:
            epsilon = -1

        history['f1'].append(f1)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['loss'].append(val_loss / len(val_loader))
        history['soft_acc'].append(soft_acc)
        history['epsilon'].append(epsilon)

        print(f"[DP Train] Epoch {epoch+1} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | Soft Acc: {soft_acc:.4f} | ε: {epsilon:.4f}")

    return best_metrics, history

def run_all_with_dp(csv_path, output_dir, df_train_path, model_path, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_df = pd.read_csv(df_train_path)
    class_list = sorted(set('|'.join(full_df['Finding Labels']).split('|')))
    train_df, temp_eval_df = split_train_eval(full_df, class_list, test_ratio=0.2)
    top_classes = get_top_k_classes(train_df, k=8)

    model = create_model(
        num_classes=len(class_list),
        dropout=config['dropout'],
        fc_hidden_size=config['fc_hidden_size'],
        kernel_size=config['kernel_size'],
        num_blocks=config['num_blocks']
    ).to(device)

    optimizer = get_optimizer(
        config['optimizer'], model, config['lr'], config['weight_decay']
    )

    criterion = FocalLoss(gamma=config['gamma']) if config['loss_type'] == 'focal' else nn.BCEWithLogitsLoss()

    train_dataset = ChestXrayDataset(train_df, output_dir, class_list, augmentation=config['augmentation'], top_classes=top_classes)
    val_dataset = ChestXrayDataset(temp_eval_df, output_dir, class_list, augmentation='none')

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    metrics, history = train_and_validate_with_dp(
        model, train_loader, val_loader, optimizer, criterion, config, device
    )

    torch.save(model.state_dict(), model_path)

    final_state_dict = torch.load(model_path)
    vector = torch.cat([
        v.flatten() for v in final_state_dict.values()
        if torch.is_floating_point(v)
    ])

    dp_params = {
        'epsilon': metrics.get('epsilon', 0.0),
        'delta': 1e-5,
        'noise_multiplier': config.get('noise_multiplier', 1.0),
        'max_grad_norm': config.get('max_grad_norm', 1.0)
    }

    print(f"📤 ε = {metrics.get('epsilon', 0.0):.2f}，向量维度 = {len(vector)}")
    return vector.cpu().tolist(), json.dumps(dp_params)
