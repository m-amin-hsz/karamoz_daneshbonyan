"""
قدم ۳: آموزش شبکه عصبی و ارزیابی عملکرد (حل گلوگاه ورودی طول متغیر با Padding & Masking)
مطابق با آخرین اصلاحات کارفرما:
۱. هر یک از اعضای بردار ویژگی به طور مستقل با احتمال ۳۰٪ می‌تواند مفقود باشد (Independent Dropout).
۲. ذخیره‌سازی وزن‌های مدل آموزش‌دیده در فایل trained_model.joblib جهت استفاده مستقیم در اسکریپت مجزای inference.py.
۳. رسم و ذخیره نمودارهای ارزیابی دقت آموزش و آزمون (accuracy_evaluation_plot.png).
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import json
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# ==============================================================================
# ۱. مشخصات بازه‌ها و محدوده‌های فیزیکی ویژگی‌ها
# ==============================================================================

MAX_LEN_F1 = 10
MAX_LEN_F2 = 20
MAX_LEN_F3 = 5

MIN_VAL_F1, MAX_VAL_F1 = 1000.0, 10000.0
MIN_VAL_F2, MAX_VAL_F2 = 20000.0, 100000.0
MIN_VAL_F3, MAX_VAL_F3 = 0.1, 1000.0


def parse_intervals(interval_str: str) -> List[Tuple[float, float]]:
    """تبدیل رشته بازه‌ها در پایگاه‌داده به لیست تاپل‌های (start, end)"""
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


def parse_query_points(query_dict: Dict, feature_name: str) -> List[float]:
    """استخراج مقادیر عددی ورودی برای یک ویژگی"""
    values_key = f"{feature_name}_values"
    if values_key in query_dict and isinstance(query_dict[values_key], list):
        return [float(x) for x in query_dict[values_key]]
    
    if feature_name in query_dict:
        raw_val = query_dict[feature_name]
        if isinstance(raw_val, list):
            return [float(x) for x in raw_val]
        pts = []
        for p in str(raw_val).split(","):
            p = p.strip()
            if p:
                try:
                    pts.append(float(p))
                except ValueError:
                    continue
        return pts
    return []


# ==============================================================================
# ۲. انکودر ورودی طول متغیر با رویکرد پدینگ و ماسک اعتبار (Padding & Masking)
# ==============================================================================

def encode_feature_vector(points: List[float], max_len: int, min_val: float, max_val: float) -> Tuple[List[float], List[float]]:
    """نرمال‌سازی داده خام و ساخت بردار پدینگ به همراه بردار ماسک اعتبار دودویی"""
    if not points:
        return [0.0] * max_len, [0.0] * max_len
    
    sorted_pts = sorted(points)
    span = max_val - min_val
    norm_pts = [(x - min_val) / span for x in sorted_pts]
    
    actual_len = len(norm_pts)
    pad_len = max(0, max_len - actual_len)
    
    values = norm_pts[:max_len] + [0.0] * pad_len
    mask = [1.0] * min(actual_len, max_len) + [0.0] * pad_len
    return values, mask


def encode_variable_length_query(q1: List[float], q2: List[float], q3: List[float]) -> np.ndarray:
    """تبدیل ورودی با طول متغیر به تنسور ورودی ثابت ۷۰ بعدی (۳۵ مقدار نرمال‌شده + ۳۵ ماسک)"""
    v1, m1 = encode_feature_vector(q1, MAX_LEN_F1, MIN_VAL_F1, MAX_VAL_F1)
    v2, m2 = encode_feature_vector(q2, MAX_LEN_F2, MIN_VAL_F2, MAX_VAL_F2)
    v3, m3 = encode_feature_vector(q3, MAX_LEN_F3, MIN_VAL_F3, MAX_VAL_F3)
    return np.array(v1 + v2 + v3 + m1 + m2 + m3, dtype=np.float32)


# ==============================================================================
# ۳. تولید داده آموزشی با حذف مستقل ۳۰٪ برای هر یک از اعضا
# ==============================================================================

def sample_query_points(intervals: List[Tuple[float, float]], drop_prob: float = 0.30, noise_rate: float = 0.01) -> List[float]:
    """
    تولید بردار اعداد نقطه‌ای از بازه‌ها:
    - هر یک از اعضا به صورت کاملاً مستقل با شانس ۳۰٪ می‌تواند مفقود باشد (Independent Dropout)
    - اعمال خطای ۱٪ به هر عدد باقی‌مانده
    """
    if not intervals:
        return []
    
    pts = [float(np.random.uniform(s, e)) for s, e in intervals]
    
    # اصلاحیه کارفرما: هر عضو مستقلاً ۳۰٪ احتمال حذف دارد
    kept = [p for p in pts if np.random.random() >= drop_prob]
    if not kept and pts:
        kept = [pts[np.random.randint(0, len(pts))]]
        
    noisy_pts = [p * (1.0 + np.random.uniform(-noise_rate, noise_rate)) for p in kept]
    return noisy_pts


def generate_training_data(df: pd.DataFrame, samples_per_class: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    """تولید ۸۰۰۰ نمونه آموزشی (۸۰ نمونه برای هر یک از ۱۰۰ سطر پایگاه‌داده)"""
    parsed_db = []
    for _, row in df.iterrows():
        parsed_db.append((
            parse_intervals(row["Feature1"]),
            parse_intervals(row["Feature2"]),
            parse_intervals(row["Feature3"])
        ))
    
    n_rows = len(df)
    X = []
    y = []
    
    for row_idx in range(n_rows):
        d1, d2, d3 = parsed_db[row_idx]
        for _ in range(samples_per_class):
            q1 = sample_query_points(d1)
            q2 = sample_query_points(d2)
            q3 = sample_query_points(d3)
            
            x_vec = encode_variable_length_query(q1, q2, q3)
            X.append(x_vec)
            y.append(row_idx)
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ==============================================================================
# ۴. آموزش شبکه عصبی و رسم نمودارهای دقت عملکرد
# ==============================================================================

def plot_performance_evaluation(model: MLPClassifier, X_train, y_train, X_test, y_test, output_path: str = "accuracy_evaluation_plot.png"):
    """رسم نمودارهای کامل دقت آموزش و تست (تابع هزینه، منحنی یادگیری و ستون‌های دقت)"""
    train_acc = accuracy_score(y_train, model.predict(X_train)) * 100.0
    test_acc = accuracy_score(y_test, model.predict(X_test)) * 100.0
    
    test_probs = model.predict_proba(X_test)
    top2_acc = np.mean([y_test[i] in np.argsort(test_probs[i])[-2:] for i in range(len(y_test))]) * 100.0
    top3_acc = np.mean([y_test[i] in np.argsort(test_probs[i])[-3:] for i in range(len(y_test))]) * 100.0
    top5_acc = np.mean([y_test[i] in np.argsort(test_probs[i])[-5:] for i in range(len(y_test))]) * 100.0
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ۱. منحنی همگرایی تابع هزینه آموزش
    axes[0].plot(model.loss_curve_, color="#1f77b4", lw=2.5, label="Training Loss")
    axes[0].set_title("Training Loss Convergence Curve", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epochs / Iterations", fontsize=10)
    axes[0].set_ylabel("Categorical Cross-Entropy", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()
    
    # ۲. منحنی دقت اعتبارسنجی طی ایپاک‌ها
    if model.validation_scores_:
        val_accs = [v * 100.0 for v in model.validation_scores_]
        axes[1].plot(val_accs, color="#2ca02c", lw=2.5, marker="o", markersize=4, label="Validation Accuracy")
        axes[1].set_title("Validation Accuracy over Epochs", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Epochs", fontsize=10)
        axes[1].set_ylabel("Accuracy (%)", fontsize=10)
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "Early Stopping Validation Disabled", ha="center", va="center")
        
    # ۳. نمودار میله‌ای مقایسه عملکرد در آموزش و آزمون
    metrics = ["Train Acc", "Test Top-1", "Test Top-2", "Test Top-3", "Test Top-5"]
    values = [train_acc, test_acc, top2_acc, top3_acc, top5_acc]
    colors = ["#3498db", "#e74c3c", "#e67e22", "#2ecc71", "#1abc9c"]
    
    bars = axes[2].bar(metrics, values, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    axes[2].set_ylim(0, 115)
    axes[2].set_title("Algorithm Performance: Train vs Unseen Test Data", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Accuracy (%)", fontsize=10)
    axes[2].grid(axis="y", linestyle="--", alpha=0.6)
    
    for bar in bars:
        yval = bar.get_height()
        axes[2].text(
            bar.get_x() + bar.get_width() / 2.0, yval + 1.5,
            f"{yval:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"📊 نمودار عملکرد الگوریتم با موفقیت ذخیره شد: {output_path}")
    
    return train_acc, test_acc, top3_acc


def train_neural_network(X: np.ndarray, y: np.ndarray, model_save_path: str = "trained_model.joblib") -> MLPClassifier:
    """آموزش شبکه عصبی، ارزیابی، رسم نمودارها و ذخیره وزن‌ها"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=15
    )
    
    print("⏳ در حال آموزش شبکه عصبی عمیق روی ۸۰۰۰ نمونه (با حذف مستقل ۳۰٪ اعضا)...")
    model.fit(X_train, y_train)
    
    # رسم و ذخیره نمودارهای ارزیابی
    train_acc, test_acc, top3_acc = plot_performance_evaluation(model, X_train, y_train, X_test, y_test)
    
    print("\n📊 نتایج ارزیابی عملکرد شبکه عصبی:")
    print("-" * 55)
    print(f"✅ دقت روی داده‌های آموزش (Train Accuracy)     : {train_acc:.2f}%")
    print(f"✅ دقت پیش‌بینی دقیق (Test Top-1 Accuracy)    : {test_acc:.2f}%")
    print(f"✅ دقت در ۳ سطر برتر (Test Top-3 Accuracy)    : {top3_acc:.2f}%")
    print("-" * 55)
    
    # ذخیره وزن‌های مدل آموزش‌دیده
    joblib.dump(model, model_save_path)
    print(f"💾 وزن‌های مدل آموزش‌دیده با موفقیت ذخیره شد: {model_save_path}")
    print("   (کاربر/کارفرما می‌تواند با اسکریپت inference.py بدون نیاز به ترین مجدد، در کسری از ثانیه خروجی بگیرد)")
    
    return model


# ==============================================================================
# ۵. استنتاج و استخراج ۳ سطر با بیشترین شباهت
# ==============================================================================

def find_top3(query: Dict, df: pd.DataFrame, model: MLPClassifier) -> List[Dict[str, Any]]:
    """استخراج ۳ سطر برتر با بیشترین شباهت از طریق خروجی Softmax شبکه"""
    q1 = parse_query_points(query, "Feature1")
    q2 = parse_query_points(query, "Feature2")
    q3 = parse_query_points(query, "Feature3")
    
    x_input = encode_variable_length_query(q1, q2, q3)
    probs = model.predict_proba([x_input])[0]
    top3_indices = np.argsort(probs)[-3:][::-1]
    
    results = []
    for rank, idx in enumerate(top3_indices, 1):
        row = df.iloc[idx]
        results.append({
            "rank": rank,
            "row_id": int(idx),
            "probability": float(probs[idx]),
            "Feature1": row["Feature1"],
            "Feature2": row["Feature2"],
            "Feature3": row["Feature3"],
        })
    return results


# ==============================================================================
# ۶. اجرای اصلی
# ==============================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("🚀 قدم ۳: آموزش شبکه عصبی و ثبت وزن‌ها (همراه با نمودارهای ۳ بعدی و دقت)")
    print("=" * 75)
    
    # ۱. بارگذاری پایگاه‌داده
    df = pd.read_pickle("database.pkl")
    print(f"📂 پایگاه‌داده با موفقیت بارگذاری شد ({len(df)} سطر).")
    
    # ۲. تولید داده آموزشی
    print("\n📊 تولید داده‌های آموزشی با حذف مستقل ۳۰٪ برای هر یک از اعضا...")
    X, y = generate_training_data(df, samples_per_class=80)
    print(f"   ابعاد ماتریس ورودی X: {X.shape} | برچسب‌ها y: {y.shape}")
    
    # ۳. آموزش شبکه و ذخیره وزن‌ها و رسم نمودار
    model = train_neural_network(X, y, model_save_path="trained_model.joblib")
    
    # ۴. بارگذاری query.json و تست استنتاج
    with open("query.json", "r", encoding="utf-8") as f:
        query = json.load(f)
        
    print(f"\n📥 تست استنتاج روی query.json (سطر مبدأ اصلی: {query.get('source_row_id')})...")
    top3 = find_top3(query, df, model)
    
    print("\n" + "=" * 75)
    print("🏆 ۳ سطر برتر پیش‌بینی‌شده توسط شبکه عصبی:")
    print("=" * 75)
    for item in top3:
        print(f"🥇 رتبه {item['rank']}: سطر شماره {item['row_id']:<3} | احتمال شباهت: {item['probability']*100:.2f}%")
        
    with open("top3_results.json", "w", encoding="utf-8") as f:
        json.dump(top3, f, ensure_ascii=False, indent=2)
    print("\n✅ نتایج در فایل top3_results.json ذخیره شد.")
