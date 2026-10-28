"""
قدم ۳: آموزش شبکه عصبی، ارزیابی عملکرد و شباهت‌یابی مطلق
مطابق با تمام اصلاحات و الزامات جدید کارفرما:
۱. پارامترها و ثوابت از config.txt خوانده می‌شوند.
۲. داده‌ها بر اساس پدینگ و ماسکینگ وارد شبکه عصبی می‌شوند و گلوگاه طول متغیر حل شده است.
۳. معیار شباهت خروجی دقیقاً بیانگر شباهت مطلق فیزیکی به پایگاه‌داده است (نه توزیع نسبی).
۴. شرط آستانه ۳۰٪ برای تشخیص داده‌های خارج از پایگاه‌داده اعمال شده است.
۵. وزن‌های مدل در trained_model.joblib و نمودار ارزیابی در accuracy_evaluation_plot.png ذخیره می‌شوند.
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
from config_utils import CONFIG
from inference import (
    parse_intervals,
    parse_query_points,
    encode_feature_vector,
    encode_variable_length_query,
    compute_exact_absolute_similarity,
    run_inference_on_query
)

warnings.filterwarnings("ignore")
np.random.seed(CONFIG["RANDOM_SEED"])


# ==============================================================================
# ۱. تولید داده آموزشی با حذف مستقل ۳۰٪ برای هر یک از اعضا
# ==============================================================================

def sample_query_points(intervals: List[Tuple[float, float]], 
                        drop_prob: float = None, 
                        noise_rate: float = None) -> List[float]:
    """تولید نمونه نقطه‌ای با حذف مستقل ۳۰٪ و خطای ۱٪ بر اساس config.txt"""
    if drop_prob is None: drop_prob = CONFIG["DROPOUT_PROB"]
    if noise_rate is None: noise_rate = CONFIG["NOISE_RATE"]
    
    if not intervals:
        return []
    
    pts = [float(np.random.uniform(s, e)) for s, e in intervals]
    
    # حذف مستقل برنولی برای هر عضو
    kept = [p for p in pts if np.random.random() >= drop_prob]
    if not kept and pts:
        kept = [pts[np.random.randint(0, len(pts))]]
        
    noisy_pts = [p * (1.0 + np.random.uniform(-noise_rate, noise_rate)) for p in kept]
    return noisy_pts


def generate_training_data(df: pd.DataFrame, samples_per_class: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """تولید دیتاست آموزشی برای ۱۰۰ سطر پایگاه‌داده"""
    if samples_per_class is None:
        samples_per_class = CONFIG["SAMPLES_PER_CLASS"]
        
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
# ۲. آموزش شبکه عصبی و رسم نمودارهای دقت عملکرد
# ==============================================================================

def plot_performance_evaluation(model: MLPClassifier, X_train, y_train, X_test, y_test, output_path: str = None):
    """رسم نمودارهای کامل دقت آموزش و تست"""
    if output_path is None:
        output_path = CONFIG["ACCURACY_PLOT"]
        
    train_acc = accuracy_score(y_train, model.predict(X_train)) * 100.0
    test_acc = accuracy_score(y_test, model.predict(X_test)) * 100.0
    
    test_probs = model.predict_proba(X_test)
    top2_acc = np.mean([y_test[i] in np.argsort(test_probs[i])[-2:] for i in range(len(y_test))]) * 100.0
    top3_acc = np.mean([y_test[i] in np.argsort(test_probs[i])[-3:] for i in range(len(y_test))]) * 100.0
    top5_acc = np.mean([y_test[i] in np.argsort(test_probs[i])[-5:] for i in range(len(y_test))]) * 100.0
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
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
        axes[1].text(0.5, 0.5, "Early Stopping Validation Enabled", ha="center", va="center")
        
    # ۳. نمودار میله‌ای مقایسه عملکرد آموزش و آزمون
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


def train_neural_network(X: np.ndarray, y: np.ndarray, model_save_path: str = None) -> MLPClassifier:
    """آموزش شبکه عصبی بر اساس پارامترهای config.txt و ذخیره وزن‌ها"""
    if model_save_path is None:
        model_save_path = CONFIG["MODEL_SAVE_PATH"]
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=CONFIG["RANDOM_SEED"], stratify=y
    )
    
    model = MLPClassifier(
        hidden_layer_sizes=(CONFIG["HIDDEN_LAYER_1"], CONFIG["HIDDEN_LAYER_2"]),
        activation="relu",
        solver="adam",
        learning_rate_init=CONFIG["LEARNING_RATE"],
        max_iter=CONFIG["MAX_ITER"],
        random_state=CONFIG["RANDOM_SEED"],
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=15
    )
    
    total_samples = len(X)
    print(f"⏳ در حال آموزش شبکه عصبی عمیق روی {total_samples} نمونه...")
    model.fit(X_train, y_train)
    
    train_acc, test_acc, top3_acc = plot_performance_evaluation(model, X_train, y_train, X_test, y_test)
    
    print("\n📊 نتایج ارزیابی عملکرد شبکه عصبی:")
    print("-" * 55)
    print(f"✅ دقت آموزش (Train Accuracy)     : {train_acc:.2f}%")
    print(f"✅ دقت پیش‌بینی دقیق (Test Top-1 Accuracy)    : {test_acc:.2f}%")
    print(f"✅ دقت در ۳ سطر برتر (Test Top-3 Accuracy)    : {top3_acc:.2f}%")
    print("-" * 55)
    
    joblib.dump(model, model_save_path)
    print(f"💾 وزن‌های مدل با موفقیت در {model_save_path} ذخیره شد.")
    
    return model


# ==============================================================================
# ۳. اجرای اصلی
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 قدم ۳: آموزش شبکه عصبی و شباهت‌یابی مطلق (تنظیمات از config.txt)")
    print("=" * 80)
    
    df = pd.read_pickle(CONFIG["DATABASE_PKL"])
    print(f"📂 پایگاه‌داده بارگذاری شد ({len(df)} سطر).")
    
    print("\n📊 تولید داده‌های آموزشی...")
    X, y = generate_training_data(df)
    print(f"   ابعاد ماتریس ورودی X: {X.shape} | برچسب‌ها y: {y.shape}")
    
    model = train_neural_network(X, y)
    
    # تست استنتاج با بررسی شرط ۳۰٪
    with open(CONFIG["QUERY_JSON"], "r", encoding="utf-8") as f:
        query = json.load(f)
        
    print(f"\n📥 اجرای استنتاج بر روی {CONFIG['QUERY_JSON']}...")
    result = run_inference_on_query(query, df)
    top3 = result["top_matches"]
    
    print("\n" + "=" * 80)
    print(f"📌 وضعیت تشخیص: {result['status_text']}")
    print("🏆 ۳ سطر برتر با بیشترین شباهت مطلق به پایگاه‌داده:")
    print("=" * 80)
    for item in top3:
        print(f"🥇 رتبه: سطر شماره {item['row_id']:<3} | شباهت مطلق به پایگاه‌داده: {item['absolute_similarity_percent']:.2f}%")
        
    formatted = []
    for rank, item in enumerate(top3, 1):
        formatted.append({
            "rank": rank,
            "row_id": item["row_id"],
            "absolute_similarity": item["absolute_similarity"],
            "absolute_similarity_percent": f"{item['absolute_similarity_percent']:.2f}%",
            "is_in_database": result["is_in_database"],
            "status": result["status_text"],
            "Feature1": item["Feature1"],
            "Feature2": item["Feature2"],
            "Feature3": item["Feature3"]
        })
        
    with open(CONFIG["TOP3_RESULTS_JSON"], "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)
    print(f"\n✅ نتایج در {CONFIG['TOP3_RESULTS_JSON']} ذخیره گردید.")
