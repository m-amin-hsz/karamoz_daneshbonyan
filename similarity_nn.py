"""
قدم ۳: الگوریتم شباهت‌یابی با شبکه عصبی (حل گلوگاه ورودی با طول متغیر)
مطابق با بازخورد کارفرما:
۱. از محاسبه دستی شباهت قبل از شبکه و سرچ خطی روی پایگاه‌داده خودداری شده است.
۲. از تبدیل داده به پارامترهای آماری (مانند میانگین، انحراف معیار، F1 و ...) خودداری شده است.
۳. گلوگاه ورودی داده با طول متغیر از طریق رویکرد اصیل شبکه‌های عصبی عمیق:
   «پدینگ با ظرفیت ماکزیمم همراه با بردار ماسک اعتبار (Padding & Validity Masking)»
   حل شده است.
۴. داده ورودی مستقیماً و یک‌باره وارد شبکه عصبی شده و خروجی Softmax آن توزیع احتمال شباهت
   روی ۱۰۰ سطر پایگاه‌داده را فراهم می‌کند و ۳ سطر با بیشترین شباهت استخراج می‌گردند.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# ==============================================================================
# ۱. مشخصات بازه‌ها و محدوده‌های فیزیکی داده‌ها
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
# ۲. حل گلوگاه ورودی با طول متغیر در شبکه عصبی: رویکرد Padding & Masking
# ==============================================================================

def encode_feature_vector(points: List[float], max_len: int, min_val: float, max_val: float) -> Tuple[List[float], List[float]]:
    """
    نرمال‌سازی داده خام و ساخت بردار پدینگ همراه با بردار ماسک اعتبار:
    - مقادیر خام مستقیماً به بازه [0, 1] نرمال می‌شوند (بدون تبدیل به آماره‌هایی مثل میانگین یا انحراف معیار).
    - آرایه تا طول ماکزیمم با مقدار 0.0 پد می‌شود (Padding).
    - یک بردار ماسک دودویی هم‌طول ایجاد می‌شود: ۱ برای داده‌های واقعی و ۰ برای موقعیت‌های پدشده (Validity Mask).
    """
    if not points:
        return [0.0] * max_len, [0.0] * max_len
    
    # مرتب‌سازی نقاط برای حفظ یکپارچگی ساختار ترتیب بازه‌ها
    sorted_pts = sorted(points)
    
    # نرمال‌سازی داده‌ها بر اساس محدوده فیزیکی دامنه
    span = max_val - min_val
    norm_pts = [(x - min_val) / span for x in sorted_pts]
    
    actual_len = len(norm_pts)
    pad_len = max(0, max_len - actual_len)
    
    values = norm_pts[:max_len] + [0.0] * pad_len
    mask = [1.0] * min(actual_len, max_len) + [0.0] * pad_len
    
    return values, mask


def encode_variable_length_query(q1: List[float], q2: List[float], q3: List[float]) -> np.ndarray:
    """
    تبدیل ورودی با طول متغیر به تنسور ورودی ثابت با استفاده از ایده Padding و Masking:
    - ورودی خام ویژگی ۱ (حداکثر ۱۰ عضو) + ماسک اعتبار (۱۰ عضو)
    - ورودی خام ویژگی ۲ (حداکثر ۲۰ عضو) + ماسک اعتبار (۲۰ عضو)
    - ورودی خام ویژگی ۳ (حداکثر ۵ عضو) + ماسک اعتبار (۵ عضو)
    مجموعاً یک بردار ۷۰ بعدی شامل ۳۵ مقدار نرمال‌شده و ۳۵ مقدار ماسک اعتبارسنجی
    """
    v1, m1 = encode_feature_vector(q1, MAX_LEN_F1, MIN_VAL_F1, MAX_VAL_F1)
    v2, m2 = encode_feature_vector(q2, MAX_LEN_F2, MIN_VAL_F2, MAX_VAL_F2)
    v3, m3 = encode_feature_vector(q3, MAX_LEN_F3, MIN_VAL_F3, MAX_VAL_F3)
    
    full_vector = v1 + v2 + v3 + m1 + m2 + m3
    return np.array(full_vector, dtype=np.float32)


# ==============================================================================
# ۳. تولید داده آموزشی جهت یادگیری بازنمایی سطرهای پایگاه‌داده
# ==============================================================================

def sample_query_points(intervals: List[Tuple[float, float]], drop_prob: float = 0.30, noise_rate: float = 0.01) -> List[float]:
    """تولید نمونه عددی از بازه‌ها مطابق قوانین PDF (انتخاب از بازه، ۳۰٪ حذف، ۱٪ نویز)"""
    if not intervals:
        return []
    pts = [float(np.random.uniform(s, e)) for s, e in intervals]
    if len(pts) > 1 and np.random.random() < drop_prob:
        drop_idx = np.random.randint(0, len(pts))
        pts.pop(drop_idx)
    noisy_pts = [p * (1.0 + np.random.uniform(-noise_rate, noise_rate)) for p in pts]
    return noisy_pts


def generate_training_data(df: pd.DataFrame, samples_per_class: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    تولید داده‌های آموزشی برای یادگیری مستقیم ۱۰۰ سطر پایگاه‌داده توسط شبکه:
    به ازای هر سطر از ۱۰۰ سطر پایگاه‌داده، ۶۰ نمونه کوئری با شرایط واقعی (طول متغیر، نویز و حذف عضو)
    تولید شده و به تنسورهای ورودی همراه با ماسک اعتبار تبدیل می‌شوند.
    مجموعاً ۶۰۰۰ نمونه آموزشی برای ۱۰۰ کلاس.
    """
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
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y


# ==============================================================================
# ۴. معماری و آموزش شبکه عصبی
# ==============================================================================

def train_neural_network(X: np.ndarray, y: np.ndarray) -> MLPClassifier:
    """
    آموزش شبکه عصبی MLP برای نگاشت مستقیم داده ورودی به سطرهای پایگاه‌داده:
    - ورودی: ۷۰ گره (۳۵ داده نرمال‌شده + ۳۵ ماسک اعتبارسنجی)
    - لایه‌های مخفی: لایه اول ۲۵۶ نورون، لایه دوم ۱۲۸ نورون با فعال‌ساز ReLU
    - خروجی: ۱۰۰ نورون با تابع Softmax (احتمال تعلق به هر یک از ۱۰۰ سطر)
    - بهینه‌ساز: Adam همراه با Early Stopping جهت جلوگیری از بیش‌برازش
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=350,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=15
    )
    
    print("⏳ در حال آموزش شبکه عصبی عمیق روی ورودی‌های طول متغیر (Padding + Masking)...")
    model.fit(X_train, y_train)
    
    # ارزیابی
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    test_probs = model.predict_proba(X_test)
    top3_acc = np.mean([y_test[i] in np.argsort(test_probs[i])[-3:] for i in range(len(y_test))])
    
    print("\n📊 نتایج ارزیابی شبکه عصبی روی داده‌های آزمون (داده‌های کاملاً جدید):")
    print("-" * 55)
    print(f"✅ دقت آموزش (Train Accuracy)     : {train_acc * 100:.2f}%")
    print(f"✅ دقت پیش‌بینی دقیق (Top-1 Accuracy): {test_acc * 100:.2f}%")
    print(f"✅ دقت در ۳ سطر برتر (Top-3 Accuracy): {top3_acc * 100:.2f}%")
    print("-" * 55)
    
    return model


# ==============================================================================
# ۵. استنتاج و استخراج ۳ سطر با بیشترین شباهت (بدون سرچ در پایگاه‌داده)
# ==============================================================================

def find_top3(query: Dict, df: pd.DataFrame, model: MLPClassifier) -> List[Dict[str, Any]]:
    """
    استخراج ۳ سطر با بیشترین شباهت از طریق یک‌بار ارسال داده ورودی به شبکه عصبی:
    هیچ‌گونه حلقه سرچ یا مقایسه دستی با پایگاه‌داده انجام نمی‌شود!
    شبکه عصبی مستقیماً احتمال تعلق به تک‌تک ۱۰۰ سطر را در قالب توزیع Softmax برمی‌گرداند.
    """
    q1 = parse_query_points(query, "Feature1")
    q2 = parse_query_points(query, "Feature2")
    q3 = parse_query_points(query, "Feature3")
    
    # تبدیل داده ورودی به بردار ۷۰ بعدی (مقادیر + ماسک)
    x_input = encode_variable_length_query(q1, q2, q3)
    
    # ارسال به شبکه عصبی (One-pass Inference)
    probs = model.predict_proba([x_input])[0]
    
    # استخراج ۳ اندیس با بیشترین احتمال
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
    print("🚀 قدم ۳: شباهت‌یابی مستقیم با شبکه عصبی بر روی داده‌های با طول متغیر")
    print("   (رویکرد Padding + Masking بدون استفاده از پارامترهای آماری یا سرچ دستی)")
    print("=" * 75)
    
    # بارگذاری پایگاه‌داده
    print("\n📂 بارگذاری پایگاه‌داده...")
    df = pd.read_pickle("database.pkl")
    print(f"   تعداد سطرهای موجود در پایگاه‌داده: {len(df)}")
    
    # تولید داده آموزشی
    print("\n📊 تولید داده‌های آموزشی (۶۰ نمونه برای هر یک از ۱۰۰ سطر = ۶۰۰۰ نمونه)...")
    X, y = generate_training_data(df, samples_per_class=60)
    print(f"   ابعاد ماتریس ورودی X: {X.shape} (شامل ۳۵ داده نرمال‌شده + ۳۵ ماسک) | برچسب‌ها y: {y.shape}")
    
    # آموزش شبکه عصبی
    model = train_neural_network(X, y)
    
    # بارگذاری داده دریافتی (query.json)
    print("\n📥 بارگذاری داده ورودی به شبکه (query.json)...")
    with open("query.json", "r", encoding="utf-8") as f:
        query = json.load(f)
    
    print(f"   شماره سطر منبع اصلی کوئری: سطر {query.get('source_row_id')}")
    print(f"   داده‌های عددی Feature1   : {str(query.get('Feature1'))[:75]}...")
    print(f"   داده‌های عددی Feature2   : {str(query.get('Feature2'))[:75]}...")
    print(f"   داده‌های عددی Feature3   : {str(query.get('Feature3'))}")
    
    # استنتاج مستقیم توسط شبکه عصبی
    print("\n🔍 در حال محاسبه احتمالات خروجی شبکه عصبی (توزیع Softmax روی ۱۰۰ کلاس)...")
    top3 = find_top3(query, df, model)
    
    # نمایش نتایج
    print("\n" + "=" * 75)
    print("🏆 ۳ سطر برتر پیش‌بینی‌شده توسط شبکه عصبی به همراه احتمال شباهت:")
    print("=" * 75)
    
    for item in top3:
        print(f"\n🥇 رتبه {item['rank']}: سطر شماره {item['row_id']}")
        print(f"   درصد احتمال شباهت (Softmax Probability): {item['probability'] * 100:.2f}%")
        print(f"   بازه‌های Feature1 در پایگاه‌داده: {item['Feature1'][:75]}...")
        print(f"   بازه‌های Feature2 در پایگاه‌داده: {item['Feature2'][:75]}...")
        print(f"   بازه‌های Feature3 در پایگاه‌داده: {item['Feature3']}")
    
    # ذخیره نتایج در فایل JSON
    with open("top3_results.json", "w", encoding="utf-8") as f:
        json.dump(top3, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 75)
    print("✅ نتایج نهایی با موفقیت در فایل top3_results.json ذخیره گردید.")
    print("=" * 75)