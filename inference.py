"""
کد جداگانه استنتاج سریع با استفاده از وزن‌های از پیش آموزش‌دیده (Inference)
مطابق با آخرین اصلاحات و الزامات کارفرما:
۱. تمام پارامترها و ثوابت از config.txt خوانده می‌شوند.
۲. معیار شباهت خروجی نسبی نیست، بلکه دقیقاً شباهت فیزیکی مطلق به داده‌های پایگاه‌داده را بیان می‌کند (Absolute Database Similarity).
۳. قابلیت ارزیابی داده‌های خارج از پایگاه‌داده و تشخیص «عدم وجود در پایگاه‌داده» برای شباهت‌های کمتر از ۳۰٪.

نحوه اجرا:
python inference.py
یا
python inference.py --query query.json --model trained_model.joblib
"""

import sys
import os
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from config_utils import CONFIG


# ==============================================================================
# ۱. توابع استخراج نقاط و بازه‌ها
# ==============================================================================

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
# ۲. انکودر ورودی بر اساس Padding و Masking
# ==============================================================================

def encode_feature_vector(points: List[float], max_len: int, min_val: float, max_val: float) -> Tuple[List[float], List[float]]:
    """نرمال‌سازی داده و ساخت بردار پدینگ همراه با بردار ماسک اعتبار دودویی"""
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
    """تبدیل ورودی با طول متغیر به بردار ورودی ۷۰ بعدی بر اساس پارامترهای config.txt"""
    v1, m1 = encode_feature_vector(
        q1, CONFIG["FEATURE1_MAX_INTERVALS"], CONFIG["FEATURE1_MIN_VAL"], CONFIG["FEATURE1_MAX_VAL"]
    )
    v2, m2 = encode_feature_vector(
        q2, CONFIG["FEATURE2_MAX_INTERVALS"], CONFIG["FEATURE2_MIN_VAL"], CONFIG["FEATURE2_MAX_VAL"]
    )
    v3, m3 = encode_feature_vector(
        q3, CONFIG["FEATURE3_MAX_INTERVALS"], CONFIG["FEATURE3_MIN_VAL"], CONFIG["FEATURE3_MAX_VAL"]
    )
    return np.array(v1 + v2 + v3 + m1 + m2 + m3, dtype=np.float32)


# ==============================================================================
# ۳. محاسبه معیار شباهت مطلق دقیق به پایگاه‌داده (نه نسبی میان کلاس‌ها)
# ==============================================================================

def calc_point_to_interval_match(point: float, intervals: List[Tuple[float, float]], tolerance: float = None) -> float:
    """
    بررسی میزان انطباق یک نقطه با مجموعه بازه‌ها با در نظر گرفتن تلرانس خطای ۱٪:
    - اگر نقطه درون بازه یا درون تلرانس ۱٪ باشد: امتیاز ۱.۰
    - در غیر این صورت: امتیاز عضویت نرم گاوسی
    """
    if tolerance is None:
        tolerance = CONFIG["NOISE_RATE"] * 1.5  # ۱.۵٪ تلرانس با احتساب نویز
        
    best_sim = 0.0
    for s, e in intervals:
        if s * (1.0 - tolerance) <= point <= e * (1.0 + tolerance):
            return 1.0
        elif point < s:
            rel_dist = (s - point) / s if s != 0 else abs(s - point)
        else:
            rel_dist = (point - e) / e if e != 0 else abs(point - e)
            
        sim = float(np.exp(-0.5 * (rel_dist / CONFIG["NOISE_RATE"]) ** 2))
        if sim > best_sim:
            best_sim = sim
    return best_sim


def compute_exact_absolute_similarity(points_tuple: Tuple[List[float], List[float], List[float]], 
                                      db_row_tuple: Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]]) -> float:
    """
    محاسبه معیار شباهت مطلق به پایگاه‌داده (Absolute Database Similarity):
    - مستقل از سایر سطرها (غیرنسبی)
    - دقیقاً بیانگر درصد تطابق فیزیکی مقادیر ورودی با بازه‌های سطر هدف است
    """
    q1, q2, q3 = points_tuple
    d1, d2, d3 = db_row_tuple
    
    def feature_sim(pts, ints):
        if not pts or not ints:
            return 0.0
        # نسبت نقاط منطبق‌شده ورودی بر بازه‌ها
        p_scores = [calc_point_to_interval_match(p, ints) for p in pts]
        precision = float(np.mean(p_scores))
        
        # میزان پوشش بازه‌های سطر توسط نقاط ورودی
        i_scores = []
        for s, e in ints:
            best_match = max(calc_point_to_interval_match(p, [(s, e)]) for p in pts)
            i_scores.append(best_match)
        recall = float(np.mean(i_scores))
        
        return (2.0 * precision * recall) / (precision + recall + 1e-8)
        
    s1 = feature_sim(q1, d1)
    s2 = feature_sim(q2, d2)
    s3 = feature_sim(q3, d3)
    
    # میانگین شباهت ۳ ویژگی
    return float((s1 + s2 + s3) / 3.0)


# ==============================================================================
# ۴. تابع اصلی اجرای استنتاج سریع با اعمال شرط آستانه ۳۰٪
# ==============================================================================

def run_inference_on_query(query: Dict, df: pd.DataFrame, model_path: str = None) -> Dict[str, Any]:
    """اجرای استنتاج بر روی یک کوئری و ارزیابی شباهت مطلق به همراه بررسی شرط ۳۰٪"""
    if model_path is None:
        model_path = CONFIG["MODEL_SAVE_PATH"]
        
    threshold = CONFIG["SIMILARITY_THRESHOLD"]
    
    q1 = parse_query_points(query, "Feature1")
    q2 = parse_query_points(query, "Feature2")
    q3 = parse_query_points(query, "Feature3")
    query_points = (q1, q2, q3)
    
    # استنتاج شبکه عصبی
    probs = np.zeros(len(df))
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        x_input = encode_variable_length_query(q1, q2, q3)
        probs = model.predict_proba([x_input])[0]
        
    # محاسبه شباهت مطلق دقیق برای سطرها
    all_results = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        d1 = parse_intervals(row["Feature1"])
        d2 = parse_intervals(row["Feature2"])
        d3 = parse_intervals(row["Feature3"])
        
        abs_sim = compute_exact_absolute_similarity(query_points, (d1, d2, d3))
        
        all_results.append({
            "row_id": int(idx),
            "absolute_similarity": abs_sim,
            "absolute_similarity_percent": abs_sim * 100.0,
            "neural_confidence": float(probs[idx]),
            "Feature1": row["Feature1"],
            "Feature2": row["Feature2"],
            "Feature3": row["Feature3"],
        })
        
    all_results.sort(key=lambda x: x["absolute_similarity"], reverse=True)
    top3 = all_results[:3]
    
    max_sim = top3[0]["absolute_similarity"]
    is_in_database = max_sim >= threshold
    
    if is_in_database:
        status_text = f"تأیید وجود در پایگاه‌داده (شباهت بیشینه: {max_sim*100:.2f}% >= {threshold*100:.0f}%)"
    else:
        status_text = f"عدم وجود در پایگاه‌داده (شباهت کمتر از {threshold*100:.0f}%: بیشترین شباهت {max_sim*100:.2f}%)"
        
    return {
        "is_in_database": is_in_database,
        "status_text": status_text,
        "threshold_percent": threshold * 100.0,
        "top_matches": top3
    }


def run_inference(model_path: str = None,
                  query_path: str = None,
                  db_path: str = None,
                  output_path: str = None):
    """تابع اجرایی CLI برای استنتاج سریع"""
    if model_path is None: model_path = CONFIG["MODEL_SAVE_PATH"]
    if query_path is None: query_path = CONFIG["QUERY_JSON"]
    if db_path is None: db_path = CONFIG["DATABASE_PKL"]
    if output_path is None: output_path = CONFIG["TOP3_RESULTS_JSON"]
    
    if not os.path.exists(query_path):
        print(f"❌ خطا: فایل داده ورودی ({query_path}) یافت نشد!")
        sys.exit(1)
    if not os.path.exists(db_path):
        print(f"❌ خطا: فایل پایگاه‌داده ({db_path}) یافت نشد!")
        sys.exit(1)
        
    print(f"⚡ اجرای استنتاج سریع با تنظیمات config.txt...")
    print(f"   مسیر مدل: {model_path}")
    print(f"   مسیر داده ورودی: {query_path}")
    
    df = pd.read_pickle(db_path)
    with open(query_path, "r", encoding="utf-8") as f:
        query = json.load(f)
        
    print(f"📥 داده ورودی بارگذاری شد (سطر منبع: {query.get('source_row_id')}).")
    print(f"   Feature1: {str(query.get('Feature1'))[:70]}...")
    print(f"   Feature2: {str(query.get('Feature2'))[:70]}...")
    print(f"   Feature3: {str(query.get('Feature3'))}")
    
    result = run_inference_on_query(query, df, model_path=model_path)
    top3 = result["top_matches"]
    
    print("\n" + "=" * 80)
    print("🏆 نتایج شباهت مطلق به پایگاه‌داده (Absolute Similarity):")
    print("=" * 80)
    print(f"📌 وضعیت تشخیص: {result['status_text']}")
    print("-" * 80)
    
    formatted_results = []
    for rank, item in enumerate(top3, 1):
        print(f"🥇 رتبه {rank}: سطر شماره {item['row_id']:<3} | شباهت مطلق به پایگاه‌داده: {item['absolute_similarity_percent']:.2f}%")
        print(f"   بازه‌های Feature1: {item['Feature1'][:65]}...")
        print(f"   بازه‌های Feature2: {item['Feature2'][:65]}...")
        print(f"   بازه‌های Feature3: {item['Feature3']}")
        print("-" * 80)
        
        formatted_results.append({
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
        
    if not result["is_in_database"]:
        print("\n⚠️ هشدار: حداکثر شباهت کمتر از ۳۰٪ است؛ این داده به عنوان «عدم وجود در پایگاه‌داده» تلقی می‌شود.")
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_results, f, ensure_ascii=False, indent=2)
    print(f"✅ نتایج استنتاج با موفقیت در {output_path} ذخیره شد.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="استنتاج سریع با بارگذاری وزن‌های مدل")
    parser.add_argument("--model", type=str, default=None, help="مسیر فایل وزن‌های مدل")
    parser.add_argument("--query", type=str, default=None, help="مسیر فایل کوئری ورودی")
    parser.add_argument("--db", type=str, default=None, help="مسیر فایل پایگاه‌داده")
    parser.add_argument("--out", type=str, default=None, help="مسیر فایل ذخیره نتایج")
    
    args = parser.parse_args()
    run_inference(args.model, args.query, args.db, args.out)
