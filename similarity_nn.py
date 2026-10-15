"""
قدم ۳: الگوریتم شباهت‌یابی با شبکه عصبی (MLP)
مطابق با مشخصات پروژه کارآموزی و بازخورد کارفرما:
- داده‌های ورودی به شبکه، مقادیر عددی (نقاط) هستند که از بازه‌های پایگاه‌داده انتخاب شده‌اند.
- پایگاه‌داده دارای ۱۰۰ سطر و ۳ ویژگی برداری بازه‌ای است.
- مدل شبکه عصبی (MLP) وظیفه یادگیری وزن‌ها و الگوی شباهت بین داده ورودی و سطرهای پایگاه‌داده را بر عهده دارد.
- ۳ سطر با بیشترین شباهت به همراه درصد احتمال شباهت هر یک استخراج و ذخیره می‌شوند.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)


# ==============================================================================
# ۱. توابع پردازش بازه‌ها و داده‌های ورودی نقطه‌ای
# ==============================================================================

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
    """
    استخراج مقادیر عددی (نقاط) داده ورودی برای یک ویژگی
    (پشتیبانی از هر دو فرمت لیست عددی و رشته کاما-جدا)
    """
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
# ۲. فرمول‌بندی تطابق نقطه با بازه (Point-to-Interval Matching)
# ==============================================================================

def point_interval_metrics(point: float, interval: Tuple[float, float], sigma: float = 0.01) -> Tuple[float, float]:
    """
    محاسبه فاصله نسبی و امتیاز عضویت نرم نقطه در بازه با در نظر گرفتن تلرانس خطای ۱٪:
    - اگر نقطه درون بازه باشد: فاصله نسبی صفر و شباهت ۱.۰ است.
    - اگر خارج از بازه باشد: فاصله نسبی نسبت به مرز بازه محاسبه می‌شود.
    - شباهت نرم از تابع گاوسی با انحراف معیار sigma=0.01 (منطبق بر ۱٪ نویز مجاز) پیروی می‌کند.
    """
    s, e = interval
    if s <= point <= e:
        rel_dist = 0.0
    elif point < s:
        rel_dist = (s - point) / s if s != 0 else abs(s - point)
    else:
        rel_dist = (point - e) / e if e != 0 else abs(point - e)
    
    # تلرانس خطای ۱٪ (نویز مجاز بند ۳ PDF)
    if rel_dist <= 0.005:
        sim = 1.0
    else:
        sim = float(np.exp(-0.5 * (rel_dist / sigma) ** 2))
    
    return rel_dist, sim


def compute_feature_match(points: List[float], intervals: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """
    محاسبه ویژگی‌های تطابق بردار اعداد ورودی با بردار بازه‌های پایگاه‌داده:
    1. Precision (نرخ برخورد): چه درصدی از نقاط ورودی با بازه‌های سطر همخوانی دارند.
    2. Recall (نرخ پوشش): چه درصدی از بازه‌های سطر توسط نقاط ورودی پوشش داده شده‌اند.
    3. F1-Score: میانگین هارمونیک دقت و پوشش.
    4. AvgDistance: میانگین حداقل فاصله نسبی نقاط ورودی از بازه‌های سطر.
    """
    if not points or not intervals:
        return 0.0, 0.0, 0.0, 1.0
    
    # ۱. ارزیابی تطابق هر نقطه با بهترین بازه ممکن
    point_sims = []
    point_dists = []
    for p in points:
        best_sim = 0.0
        best_dist = float("inf")
        for inv in intervals:
            d, s = point_interval_metrics(p, inv)
            if s > best_sim:
                best_sim = s
            if d < best_dist:
                best_dist = d
        point_sims.append(best_sim)
        point_dists.append(best_dist)
    
    precision = float(np.mean(point_sims))
    avg_dist = float(np.mean(point_dists))
    
    # ۲. ارزیابی پوشش بازه‌ها توسط نقاط ورودی
    interval_sims = []
    for inv in intervals:
        best_sim = 0.0
        for p in points:
            _, s = point_interval_metrics(p, inv)
            if s > best_sim:
                best_sim = s
        interval_sims.append(best_sim)
    
    recall = float(np.mean(interval_sims))
    f1 = float((2.0 * precision * recall) / (precision + recall + 1e-8))
    
    return precision, recall, f1, avg_dist


def compute_pair_features(query_points_tuple: Tuple[List[float], List[float], List[float]], 
                          db_row_tuple: Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]]) -> List[float]:
    """
    استخراج بردار ورودی ۱۲ بعدی نرمال‌شده برای ورودی به شبکه عصبی MLP:
    شامل [F1, Precision, Recall, Distance] برای هر یک از ۳ ویژگی (Feature1, Feature2, Feature3)
    """
    q1, q2, q3 = query_points_tuple
    d1, d2, d3 = db_row_tuple
    
    p1, r1, f1, dist1 = compute_feature_match(q1, d1)
    p2, r2, f2, dist2 = compute_feature_match(q2, d2)
    p3, r3, f3, dist3 = compute_feature_match(q3, d3)
    
    return [f1, p1, r1, dist1, f2, p2, r2, dist2, f3, p3, r3, dist3]


# ==============================================================================
# ۳. تولید داده آموزشی جهت یادگیری شبکه عصبی
# ==============================================================================

def sample_query_points_from_intervals(intervals: List[Tuple[float, float]], 
                                       drop_prob: float = 0.30, 
                                       noise_rate: float = 0.01) -> List[float]:
    """
    تولید بردار اعداد از بازه‌ها دقیقا منطبق بر قوانین مرحله ۲:
    ۱. انتخاب عدد تصادفی از درون هر بازه
    ۲. احتمال ۳۰٪ حذف یک عضو
    ۳. اعمال خطای تصادفی ۱٪
    """
    if not intervals:
        return []
    
    # انتخاب عدد از هر بازه
    pts = [float(np.random.uniform(s, e)) for s, e in intervals]
    
    # حذف تصادفی با شانس ۳۰٪
    if len(pts) > 1 and np.random.random() < drop_prob:
        drop_idx = np.random.randint(0, len(pts))
        pts.pop(drop_idx)
    
    # اعمال خطای ۱٪
    noisy_pts = [p * (1.0 + np.random.uniform(-noise_rate, noise_rate)) for p in pts]
    return noisy_pts


def generate_training_data(df: pd.DataFrame, n_samples: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
    """
    تولید دیتاست آموزشی برای شبکه عصبی:
    - نمونه‌های مثبت (برچسب ۱): کوئری ساخته‌شده از سطر i + همان سطر i در پایگاه‌داده
    - نمونه‌های منفی تصادفی (برچسب ۰): کوئری ساخته‌شده از سطر i + سطر تصادفی j (که j != i)
    - نمونه‌های منفی دشوار (Hard Negatives): کوئری سطر i + سطرهایی که همپوشانی جزئی با سطر i دارند
    """
    n_rows = len(df)
    
    # پیش‌پردازش تمام بازه‌های پایگاه‌داده
    parsed_db = []
    for _, row in df.iterrows():
        parsed_db.append((
            parse_intervals(row["Feature1"]),
            parse_intervals(row["Feature2"]),
            parse_intervals(row["Feature3"])
        ))
    
    X = []
    y = []
    
    for _ in range(n_samples):
        # انتخاب تصادفی یک سطر منبع
        src_idx = np.random.randint(0, n_rows)
        d1, d2, d3 = parsed_db[src_idx]
        
        # تولید داده ورودی نقطه‌ای از این سطر
        q1 = sample_query_points_from_intervals(d1)
        q2 = sample_query_points_from_intervals(d2)
        q3 = sample_query_points_from_intervals(d3)
        query_points = (q1, q2, q3)
        
        # ۱. ساخت نمونه مثبت (کوئری با سطر واقعی خود) -> برچسب ۱
        pos_features = compute_pair_features(query_points, (d1, d2, d3))
        X.append(pos_features)
        y.append(1)
        
        # ۲. ساخت نمونه منفی تصادفی -> برچسب ۰
        neg_idx = np.random.randint(0, n_rows)
        while neg_idx == src_idx:
            neg_idx = np.random.randint(0, n_rows)
        
        neg_features = compute_pair_features(query_points, parsed_db[neg_idx])
        X.append(neg_features)
        y.append(0)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ==============================================================================
# ۴. ساخت و آموزش شبکه عصبی MLP
# ==============================================================================

def train_model(X: np.ndarray, y: np.ndarray) -> MLPClassifier:
    """
    آموزش شبکه عصبی MLP برای ارزیابی شباهت:
    - لایه ورودی: ۱۲ ویژگی تطابق استخراج‌شده
    - لایه‌های مخفی: (64, 32) نورون با فعال‌ساز ReLU
    - بهینه‌ساز: Adam با توقف زودهنگام (Early Stopping)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=15
    )
    
    print("⏳ در حال آموزش شبکه عصبی MLP...")
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]
    
    print("\n📊 نتایج ارزیابی عملکرد شبکه عصبی:")
    print("-" * 50)
    print(f"✅ دقت روی داده‌های آموزش (Train Accuracy): {accuracy_score(y_train, y_pred_train)*100:.2f}%")
    print(f"✅ دقت روی داده‌های آزمون (Test Accuracy) : {accuracy_score(y_test, y_pred_test)*100:.2f}%")
    print(f"✅ معیار Precision روی داده‌های آزمون    : {precision_score(y_test, y_pred_test)*100:.2f}%")
    print(f"✅ معیار Recall روی داده‌های آزمون       : {recall_score(y_test, y_pred_test)*100:.2f}%")
    print(f"✅ معیار F1-Score روی داده‌های آزمون     : {f1_score(y_test, y_pred_test)*100:.2f}%")
    print(f"✅ شاخص تفکیک‌پذیری ROC-AUC             : {roc_auc_score(y_test, y_prob_test):.4f}")
    print("-" * 50)
    
    return model


# ==============================================================================
# ۵. جستجو و استخراج ۳ سطر با بیشترین شباهت
# ==============================================================================

def find_top3(query: Dict, df: pd.DataFrame, model: MLPClassifier) -> List[Dict[str, Any]]:
    """
    مقایسه داده ورودی عددی با تمامی ۱۰۰ سطر پایگاه‌داده و استخراج ۳ سطر برتر به همراه احتمال شباهت
    """
    q1 = parse_query_points(query, "Feature1")
    q2 = parse_query_points(query, "Feature2")
    q3 = parse_query_points(query, "Feature3")
    query_points = (q1, q2, q3)
    
    results = []
    
    for idx, row in df.iterrows():
        d1 = parse_intervals(row["Feature1"])
        d2 = parse_intervals(row["Feature2"])
        d3 = parse_intervals(row["Feature3"])
        db_tuple = (d1, d2, d3)
        
        # استخراج ویژگی‌های تطابق
        pair_feats = compute_pair_features(query_points, db_tuple)
        
        # محاسبه احتمال تعلق به کلاس شباهت کامل (کلاس ۱)
        proba = float(model.predict_proba([pair_feats])[0][1])
        
        results.append({
            "row_id": int(idx),
            "probability": proba,
            "f1_feature1": round(pair_feats[0], 4),
            "precision_feature1": round(pair_feats[1], 4),
            "recall_feature1": round(pair_feats[2], 4),
            "f1_feature2": round(pair_feats[4], 4),
            "precision_feature2": round(pair_feats[5], 4),
            "recall_feature2": round(pair_feats[6], 4),
            "f1_feature3": round(pair_feats[8], 4),
            "precision_feature3": round(pair_feats[9], 4),
            "recall_feature3": round(pair_feats[10], 4),
            "Feature1": row["Feature1"],
            "Feature2": row["Feature2"],
            "Feature3": row["Feature3"],
        })
    
    # مرتب‌سازی بر اساس احتمال شباهت (نزولی)
    results.sort(key=lambda item: item["probability"], reverse=True)
    
    return results[:3]


# ==============================================================================
# ۶. اجرای اصلی
# ==============================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("🚀 قدم ۳: شباهت‌یابی در پایگاه‌داده با شبکه عصبی بر اساس داده ورودی نقطه‌ای")
    print("=" * 75)
    
    # بارگذاری پایگاه‌داده
    print("\n📂 بارگذاری پایگاه‌داده...")
    df = pd.read_pickle("database.pkl")
    print(f"   تعداد سطرهای موجود در پایگاه‌داده: {len(df)}")
    
    # تولید داده‌های آموزشی
    print("\n📊 تولید ۳۰۰۰ نمونه داده آموزشی (زوج‌های مثبت و منفی)...")
    X, y = generate_training_data(df, n_samples=3000)
    print(f"   ابعاد ماتریس داده‌های آموزشی X: {X.shape} | برچسب‌ها y: {y.shape}")
    
    # آموزش مدل شبکه عصبی
    model = train_model(X, y)
    
    # بارگذاری داده دریافتی (query)
    print("\n📥 بارگذاری داده ورودی به شبکه (query.json)...")
    with open("query.json", "r", encoding="utf-8") as f:
        query = json.load(f)
    
    print(f"   شماره سطر منبع اصلی کوئری: سطر {query.get('source_row_id')}")
    print(f"   داده‌های عددی Feature1   : {str(query.get('Feature1'))[:75]}...")
    print(f"   داده‌های عددی Feature2   : {str(query.get('Feature2'))[:75]}...")
    print(f"   داده‌های عددی Feature3   : {str(query.get('Feature3'))}")
    
    # جستجوی ۳ سطر برتر
    print("\n🔍 در حال جستجو و ارزیابی شباهت بین داده ورودی و ۱۰۰ سطر پایگاه‌داده...")
    top3 = find_top3(query, df, model)
    
    # نمایش نتایج
    print("\n" + "=" * 75)
    print("🏆 ۳ سطر از پایگاه‌داده با بیشترین میزان شباهت و احتمال تعلق:")
    print("=" * 75)
    
    for rank, item in enumerate(top3, 1):
        print(f"\n🥇 رتبه {rank}: سطر شماره {item['row_id']}")
        print(f"   درصد احتمال شباهت   : {item['probability'] * 100:.2f}%")
        print(f"   تطابق ویژگی ۱ (F1)  : {item['f1_feature1']:.4f} (دقت: {item['precision_feature1']:.3f} | پوشش: {item['recall_feature1']:.3f})")
        print(f"   تطابق ویژگی ۲ (F1)  : {item['f1_feature2']:.4f} (دقت: {item['precision_feature2']:.3f} | پوشش: {item['recall_feature2']:.3f})")
        print(f"   تطابق ویژگی ۳ (F1)  : {item['f1_feature3']:.4f} (دقت: {item['precision_feature3']:.3f} | پوشش: {item['recall_feature3']:.3f})")
    
    # ذخیره نتایج در فایل JSON
    with open("top3_results.json", "w", encoding="utf-8") as f:
        json.dump(top3, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 75)
    print("✅ نتایج نهایی با موفقیت در فایل top3_results.json ذخیره گردید.")
    print("=" * 75)