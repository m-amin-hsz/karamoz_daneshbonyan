"""
قدم ۳: شباهت‌یابی با شبکه عصبی ساده (گزینه A)
- معیار شباهت: همپوشانی بازه‌ها
- شبکه عصبی: MLP برای یادگیری وزن ویژگی‌ها
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)


# ============================================================
# ۱. توابع کمکی برای کار با بازه‌ها
# ============================================================

def parse_intervals(interval_str: str) -> List[Tuple[float, float]]:
    """تبدیل رشته به لیست بازه‌ها"""
    if not interval_str or pd.isna(interval_str):
        return []
    intervals = []
    for part in str(interval_str).split(","):
        part = part.strip()
        if "-" in part:
            try:
                start, end = part.split("-")
                intervals.append((float(start), float(end)))
            except:
                continue
    return intervals


def interval_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """محاسبه طول همپوشانی دو بازه"""
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    return max(0.0, end - start)


def total_length(intervals: List[Tuple[float, float]]) -> float:
    """مجموع طول بازه‌ها"""
    return sum(e - s for s, e in intervals) + 1e-8


def similarity_score(intervals1: List[Tuple[float, float]], 
                     intervals2: List[Tuple[float, float]]) -> float:
    """
    محاسبه شباهت بین دو مجموعه بازه (معیار Jaccard-مانند)
    مقدار بین ۰ تا ۱
    """
    if not intervals1 or not intervals2:
        return 0.0
    
    overlap = 0.0
    for a in intervals1:
        for b in intervals2:
            overlap += interval_overlap(a, b)
    
    len1 = total_length(intervals1)
    len2 = total_length(intervals2)
    
    # Jaccard-like
    union = len1 + len2 - overlap
    return min(1.0, overlap / union) if union > 0 else 0.0


def compute_feature_similarities(query_row: Dict, db_row: pd.Series) -> np.ndarray:
    """محاسبه شباهت سه ویژگی بین query و یک سطر پایگاه‌داده"""
    q1 = parse_intervals(query_row["Feature1"])
    q2 = parse_intervals(query_row["Feature2"])
    q3 = parse_intervals(query_row["Feature3"])
    
    d1 = parse_intervals(db_row["Feature1"])
    d2 = parse_intervals(db_row["Feature2"])
    d3 = parse_intervals(db_row["Feature3"])
    
    s1 = similarity_score(q1, d1)
    s2 = similarity_score(q2, d2)
    s3 = similarity_score(q3, d3)
    
    return np.array([s1, s2, s3])


# ============================================================
# ۲. تولید داده آموزشی
# ============================================================

def apply_1_percent_error(value: float) -> float:
    error = value * 0.01 * np.random.uniform(-1, 1)
    return value + error


def remove_random_member(intervals: List[Tuple[float, float]], prob: float = 0.30):
    if len(intervals) <= 1:
        return intervals
    if np.random.random() < prob:
        idx = np.random.randint(0, len(intervals))
        return intervals[:idx] + intervals[idx+1:]
    return intervals


def make_noisy_version(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    noisy = []
    for s, e in intervals:
        ns = apply_1_percent_error(s)
        ne = apply_1_percent_error(e)
        if ns > ne:
            ns, ne = ne, ns
        noisy.append((ns, ne))
    return remove_random_member(noisy, 0.30)


def intervals_to_string(intervals: List[Tuple[float, float]]) -> str:
    return ",".join([f"{s:.4f}-{e:.4f}" for s, e in intervals])


def generate_training_data(df: pd.DataFrame, n_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    تولید داده آموزشی:
    - نمونه‌های مثبت: query ساخته‌شده از یک سطر + همان سطر (برچسب ۱)
    - نمونه‌های منفی: query ساخته‌شده از یک سطر + سطر دیگر (برچسب ۰)
    """
    X = []
    y = []
    
    n_rows = len(df)
    
    for _ in range(n_samples):
        # انتخاب سطر مبدأ
        src_idx = np.random.randint(0, n_rows)
        src_row = df.iloc[src_idx]
        
        # ساخت query نویزی از این سطر
        f1 = make_noisy_version(parse_intervals(src_row["Feature1"]))
        f2 = make_noisy_version(parse_intervals(src_row["Feature2"]))
        f3 = make_noisy_version(parse_intervals(src_row["Feature3"]))
        
        query = {
            "Feature1": intervals_to_string(f1),
            "Feature2": intervals_to_string(f2),
            "Feature3": intervals_to_string(f3)
        }
        
        # نمونه مثبت (همان سطر)
        sims_pos = compute_feature_similarities(query, src_row)
        X.append(sims_pos)
        y.append(1)
        
        # نمونه منفی (سطر تصادفی دیگر)
        neg_idx = src_idx
        while neg_idx == src_idx:
            neg_idx = np.random.randint(0, n_rows)
        neg_row = df.iloc[neg_idx]
        
        sims_neg = compute_feature_similarities(query, neg_row)
        X.append(sims_neg)
        y.append(0)
    
    return np.array(X), np.array(y)


# ============================================================
# ۳. آموزش شبکه عصبی
# ============================================================

def train_model(X: np.ndarray, y: np.ndarray) -> MLPClassifier:
    """آموزش یک MLP ساده"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16),   # دو لایه مخفی
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    print("⏳ در حال آموزش شبکه عصبی...")
    model.fit(X_train, y_train)
    
    # ارزیابی
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"✅ دقت روی داده آموزش: {train_acc*100:.1f}%")
    print(f"✅ دقت روی داده تست:   {test_acc*100:.1f}%")
    
    return model


# ============================================================
# ۴. پیدا کردن ۳ سطر شبیه‌ترین
# ============================================================

def find_top3(query: Dict, df: pd.DataFrame, model: MLPClassifier) -> List[Dict]:
    """
    برای یک query، شباهت با تمام سطرها را حساب می‌کند
    و ۳ تای برتر را همراه با احتمال برمی‌گرداند
    """
    results = []
    
    for idx, row in df.iterrows():
        sims = compute_feature_similarities(query, row)  # [s1, s2, s3]
        
        # احتمال تعلق به کلاس مثبت (شبیه بودن)
        proba = model.predict_proba(sims.reshape(1, -1))[0][1]
        
        results.append({
            "row_id": int(idx),
            "probability": float(proba),
            "sim_feature1": float(sims[0]),
            "sim_feature2": float(sims[1]),
            "sim_feature3": float(sims[2]),
            "Feature1": row["Feature1"],
            "Feature2": row["Feature2"],
            "Feature3": row["Feature3"]
        })
    
    # مرتب‌سازی بر اساس احتمال (نزولی)
    results.sort(key=lambda x: x["probability"], reverse=True)
    
    return results[:3]


# ============================================================
# ۵. اجرای اصلی
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 قدم ۳: شباهت‌یابی با شبکه عصبی")
    print("=" * 70)
    
    # بارگذاری پایگاه‌داده
    print("\n📂 بارگذاری پایگاه‌داده...")
    df = pd.read_pickle("database.pkl")
    print(f"   تعداد سطرها: {len(df)}")
    
    # تولید داده آموزشی
    print("\n📊 تولید داده آموزشی (نمونه‌های مثبت و منفی)...")
    X, y = generate_training_data(df, n_samples=1500)
    print(f"   تعداد کل نمونه‌ها: {len(X)} (مثبت + منفی)")
    
    # آموزش مدل
    print()
    model = train_model(X, y)
    
    # بارگذاری query
    print("\n📥 بارگذاری داده دریافتی (query)...")
    with open("query.json", "r", encoding="utf-8") as f:
        query = json.load(f)
    
    print(f"   منبع اصلی query: سطر شماره {query['source_row_id']}")
    print(f"   Feature1: {query['Feature1'][:80]}...")
    print(f"   Feature2: {query['Feature2'][:80]}...")
    print(f"   Feature3: {query['Feature3']}")
    
    # پیدا کردن ۳ سطر برتر
    print("\n🔍 در حال جستجوی ۳ سطر شبیه‌ترین...")
    top3 = find_top3(query, df, model)
    
    # نمایش نتایج
    print("\n" + "=" * 70)
    print("🏆 نتیجه نهایی: ۳ سطر با بیشترین شباهت")
    print("=" * 70)
    
    for i, item in enumerate(top3, 1):
        print(f"\n--- رتبه {i} ---")
        print(f"شماره سطر       : {item['row_id']}")
        print(f"احتمال شباهت    : {item['probability']*100:.2f}%")
        print(f"شباهت Feature1  : {item['sim_feature1']:.4f}")
        print(f"شباهت Feature2  : {item['sim_feature2']:.4f}")
        print(f"شباهت Feature3  : {item['sim_feature3']:.4f}")
    
    # ذخیره نتیجه
    with open("top3_results.json", "w", encoding="utf-8") as f:
        json.dump(top3, f, ensure_ascii=False, indent=2)
    
    print("\n✅ نتایج در فایل top3_results.json ذخیره شد.")
    print("=" * 70)