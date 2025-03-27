import pandas as pd
import random
from collections import Counter
import matplotlib.pyplot as plt
from collections import Counter

# 提取每位患者所有标签（集合形式）
def get_unique_labels(label_lists):
    unique = set()
    for label_group in label_lists:
        for label in label_group:
            unique.add(label.strip())
    return unique

# === 智能划分函数（按图像总数）===
def smart_split_by_image_count(patients_df, val_ratio=0.2, min_classes=15):
    patients = list(patients_df.itertuples(index=False, name="PatientRecord"))
    total_images = sum(len(p.Images) for p in patients)
    target_val_images = total_images * val_ratio

    for _ in range(1000):
        random.shuffle(patients)
        val_group = []
        val_total = 0
        for p in patients:
            if val_total < target_val_images:
                val_group.append(p)
                val_total += len(p.Images)

        train_group = [p for p in patients if p not in val_group]

        train_labels = set.union(*(p.Label_Set for p in train_group))
        val_labels = set.union(*(p.Label_Set for p in val_group))

        if len(train_labels) >= min_classes and len(val_labels) >= min_classes:
            return train_group, val_group

    raise ValueError("无法在1000次尝试内划分出满足条件的集合")


# === 展开为图像级标签表 ===
def expand_patient_group(patient_group):
    rows = []
    for p in patient_group:
        pid = p.Patient_ID
        labels = p.Labels
        for i, img in enumerate(p.Images):
            lbl = '|'.join(labels[i]) if isinstance(labels[i], list) else labels[i]
            rows.append({'Image Index': img, 'Finding Labels': lbl, 'Patient ID': pid})
    return pd.DataFrame(rows)

def count_class_distribution(df):
    label_counter = Counter()
    for labels in df['Finding Labels']:
        for label in labels.split('|'):
            label_counter[label.strip()] += 1
    return dict(label_counter)

# === 多标签计数函数 ===
def count_class_distribution(df):
    label_counter = Counter()
    for labels in df['Finding Labels']:
        for label in labels.split('|'):
            label_counter[label.strip()] += 1
    return dict(label_counter)

def split(csv_path):

    # === 读取 CSV 文件(Group2 labels ) ===
    df = pd.read_csv(csv_path)
    df['Image Index'] = df['Image Index'].astype(str)
    df['Patient ID'] = df['Patient ID'].astype(str)
    df['Labels List'] = df['Finding Labels'].apply(lambda x: x.split('|'))

    # === Step 2: 构建以 Patient ID 为单位的数据表 ===
    patient_to_images = df.groupby('Patient ID')['Image Index'].apply(list)
    patient_to_labels = df.groupby('Patient ID')['Labels List'].apply(list)

    patient_df = pd.DataFrame({
        'Patient_ID': patient_to_images.index,
        'Images': patient_to_images.values,
        'Labels': patient_to_labels.values
    })

    patient_df['Label_Set'] = patient_df['Labels'].apply(get_unique_labels)
    patient_df['Num_Images'] = patient_df['Images'].apply(len)

    # === Step 5: 执行划分和导出 ===
    train_group, val_group = smart_split_by_image_count(patient_df, val_ratio=0.2)
    df_train = expand_patient_group(train_group)
    df_val = expand_patient_group(val_group)

    df_train.to_csv("train_group2_smart.csv", index=False)
    df_val.to_csv("val_group2_smart.csv", index=False)

    print("✅ 划分完成！")
    print(f"训练集图像数: {len(df_train)}，患者数: {len(train_group)}")
    print(f"验证集图像数: {len(df_val)}，患者数: {len(val_group)}")

    # 示例：统计训练/验证集中类别分布
    train_dist = count_class_distribution(df_train)
    val_dist = count_class_distribution(df_val)

    print("训练集类别分布：")
    for cls, count in train_dist.items():
        print(f"{cls:<25} {count}")

    print("\n验证集类别分布：")
    for cls, count in val_dist.items():
        print(f"{cls:<25} {count}")

    df_train = pd.read_csv("train_group2_smart.csv")
    df_val = pd.read_csv("val_group2_smart.csv")

    # === 分别统计训练和验证集中每类数量 ===
    train_dist = count_class_distribution(df_train)
    val_dist = count_class_distribution(df_val)

    # === 对齐类别顺序并准备画图数据 ===
    all_labels = sorted(set(train_dist.keys()) | set(val_dist.keys()))
    train_counts = [train_dist.get(label, 0) for label in all_labels]
    val_counts = [val_dist.get(label, 0) for label in all_labels]

    # === 绘制条形图 ===
    plt.figure(figsize=(12, 6))
    x = range(len(all_labels))
    width = 0.4

    plt.bar([i - width/2 for i in x], train_counts, width=width, label='Train', color='orange')
    plt.bar([i + width/2 for i in x], val_counts, width=width, label='Validation', color='orangered')

    plt.xticks(x, all_labels, rotation=45, ha='right')
    plt.xlabel("Class")
    plt.ylabel("Image Count")
    plt.title("Class Distribution in Train and Validation Sets")
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y')

    # === Step 6: 保存图像 ===
    plt.savefig("class_distribution.png")
    plt.close()

    print("✅ 图像保存为 class_distribution_group2.png")