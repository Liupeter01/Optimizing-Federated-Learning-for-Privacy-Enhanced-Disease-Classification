# ✅ 8️⃣ 使用验证集评估最终模型（保持结构一致 + 手动传入最佳参数）
def evaluate_on_val():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")

    # ✅ 手动输入最佳参数（请替换为你最终选出的最佳组合）
    best_config = {
        'dropout': 0.4,
        'lr': 0.00015,
        'batch_size': 100,
        'optimizer': 'adam',
        'kernel_size': 4,
        'num_blocks': 2,
        'fc_hidden_size': 515,
        'weight_decay': 0.0008,
        'augmentation': 'none',  # 🔹 验证集不增强
        'loss_type': 'focal',
        'gamma': 2.0
    }

    # ✅ 载入验证集
    val_df = pd.read_csv("val_group1_smart.csv")
    class_list = sorted(set('|'.join(val_df['Finding Labels']).split('|')))

    val_dataset = ChestXrayDataset(
        val_df, "Preprocessed_Images/Group2", class_list, augmentation='none')
    val_loader = DataLoader(val_dataset, batch_size=best_config['batch_size'])

    # ✅ 构建模型结构并加载权重
    model = create_model(len(class_list), best_config['dropout'], best_config['fc_hidden_size'],
                         best_config['kernel_size'], best_config['num_blocks']).to(device)
    model.load_state_dict(torch.load(
        "best_resnet18_custom.pth", map_location=device))
    model.eval()

    # ✅ 计算评估指标
    all_preds, all_labels = [], []
    criterion = nn.BCEWithLogitsLoss()
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

    f1 = f1_score(y_true, y_pred, average='micro')
    precision = precision_score(
        y_true, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='micro', zero_division=0)

    print("\n🧪 验证集评估结果：")
    print(f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {
          recall:.4f} | Loss: {val_loss / len(val_loader):.4f}")


# ✅ 运行验证评估（请手动运行）
evaluate_on_val()
