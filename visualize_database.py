"""
نمایش گرافیکی ۳ بعدی از وضعیت پایگاه‌داده و تحلیل همپوشانی اطلاعات
مطابق با درخواست کارفرما:
- نمایش توزیع مکانی ۱۰۰ سطر پایگاه‌داده در فضای ۳ بعدی ویژگی‌ها (Feature1, Feature2, Feature3)
- نمایش گراف ارتباطی همپوشانی بین سطرها در فضای ۳ بعدی
- رسم ماتریس گرمایی (Heatmap) همپوشانی دو‌به‌دوی سطرها (سقف ۳۰٪ مطابق صورت‌مسئله)
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from typing import List, Tuple


def parse_intervals(interval_str: str) -> List[Tuple[float, float]]:
    """تبدیل رشته بازه‌ها به لیست تاپل‌های (start, end)"""
    if not interval_str or pd.isna(interval_str):
        return []
    intervals = []
    for part in str(interval_str).split(","):
        part = part.strip()
        if "-" in part:
            try:
                s, e = part.split("-")
                intervals.append((float(s), float(e)))
            except ValueError:
                continue
    return intervals


def calc_overlap_ratio(ints1: List[Tuple[float, float]], ints2: List[Tuple[float, float]]) -> float:
    """محاسبه نسبت همپوشانی بین دو مجموعه بازه مطابق فرمول صورت‌مسئله"""
    if not ints1 or not ints2:
        return 0.0
    
    def total_len(ints):
        return sum(e - s for s, e in ints)
    
    overlap = 0.0
    for s1, e1 in ints1:
        for s2, e2 in ints2:
            os = max(s1, s2)
            oe = min(e1, e2)
            if os < oe:
                overlap += (oe - os)
                
    l1, l2 = total_len(ints1), total_len(ints2)
    return overlap / ((l1 + l2) / 2.0) if (l1 + l2) > 0 else 0.0


def generate_3d_visualization(db_path: str = "database.pkl", output_image: str = "database_3d_visualization.png"):
    """تولید و ذخیره نمودارهای ۳ بعدی وضعیت پایگاه‌داده و ماتریس همپوشانی"""
    print(f"📂 بارگذاری پایگاه‌داده از {db_path}...")
    df = pd.read_pickle(db_path)
    n_rows = len(df)
    
    # استخراج مراکز سطرها در فضای ۳ بعدی و تعداد بازه‌ها
    centers_f1 = []
    centers_f2 = []
    centers_f3 = []
    num_intervals = []
    
    parsed_all = []
    for _, row in df.iterrows():
        f1 = parse_intervals(row["Feature1"])
        f2 = parse_intervals(row["Feature2"])
        f3 = parse_intervals(row["Feature3"])
        parsed_all.append((f1, f2, f3))
        
        c1 = np.mean([(s + e) / 2.0 for s, e in f1]) if f1 else 0.0
        c2 = np.mean([(s + e) / 2.0 for s, e in f2]) if f2 else 0.0
        c3 = np.mean([(s + e) / 2.0 for s, e in f3]) if f3 else 0.0
        
        centers_f1.append(c1)
        centers_f2.append(c2)
        centers_f3.append(c3)
        num_intervals.append(len(f1) + len(f2) + len(f3))
        
    centers_f1 = np.array(centers_f1)
    centers_f2 = np.array(centers_f2)
    centers_f3 = np.array(centers_f3)
    num_intervals = np.array(num_intervals)
    
    # محاسبه ماتریس همپوشانی دوبه‌دو
    print("⏳ در حال محاسبه ماتریس همپوشانی ۱۰۰ سطر...")
    overlap_matrix = np.zeros((n_rows, n_rows))
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            o1 = calc_overlap_ratio(parsed_all[i][0], parsed_all[j][0])
            o2 = calc_overlap_ratio(parsed_all[i][1], parsed_all[j][1])
            o3 = calc_overlap_ratio(parsed_all[i][2], parsed_all[j][2])
            avg_o = (o1 + o2 + o3) / 3.0
            overlap_matrix[i, j] = avg_o
            overlap_matrix[j, i] = avg_o
            
    # ترسیم گرافیکی
    fig = plt.figure(figsize=(16, 7))
    
    # ۱. نمودار ۳ بعدی توزیع داده‌ها و یال‌های همپوشانی
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    sc = ax1.scatter(
        centers_f1, centers_f2, centers_f3,
        c=np.arange(n_rows), cmap="viridis",
        s=num_intervals * 7, alpha=0.85,
        edgecolors="black", linewidth=0.5
    )
    
    # رسم خطوط ارتباطی برای سطرهایی که با هم همپوشانی دارند
    overlap_connections = 0
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            if overlap_matrix[i, j] > 0.03:
                ax1.plot(
                    [centers_f1[i], centers_f1[j]],
                    [centers_f2[i], centers_f2[j]],
                    [centers_f3[i], centers_f3[j]],
                    color="red", alpha=min(0.8, overlap_matrix[i, j] * 4.0), linewidth=0.8
                )
                overlap_connections += 1
                
    ax1.set_xlabel("Feature 1 (1000 - 10000)", fontsize=10, labelpad=8)
    ax1.set_ylabel("Feature 2 (20000 - 100000)", fontsize=10, labelpad=8)
    ax1.set_zlabel("Feature 3 (0.1 - 1000)", fontsize=10, labelpad=8)
    ax1.set_title(
        f"3D Distribution of 100 Rows & Overlap Graph ({overlap_connections} Connections)",
        fontsize=11, fontweight="bold"
    )
    plt.colorbar(sc, ax=ax1, shrink=0.6, label="Row ID (0 - 99)")
    
    # ۲. ماتریس گرمایی همپوشانی دوبه‌دو
    ax2 = fig.add_subplot(1, 2, 2)
    max_ov = float(overlap_matrix.max() * 100.0)
    im = ax2.imshow(overlap_matrix * 100.0, cmap="magma", aspect="auto", origin="lower")
    ax2.set_xlabel("Row Index", fontsize=10)
    ax2.set_ylabel("Row Index", fontsize=10)
    ax2.set_title(
        f"Pairwise Overlap Matrix Heatmap (Max Overlap = {max_ov:.2f}%)",
        fontsize=11, fontweight="bold"
    )
    plt.colorbar(im, ax=ax2, label="Overlap Percentage (%)")
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"✅ نمودار ۳ بعدی و ماتریس همپوشانی با موفقیت ذخیره شد: {output_image}")


if __name__ == "__main__":
    generate_3d_visualization()

