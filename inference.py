"""
کد جداگانه استنتاج سریع با استفاده از وزن‌های از پیش آموزش‌دیده (Inference)
مطابق با درخواست کارفرما:
«با توجه به اینکه ترین زمان‌بر است، یک کد جداگانه ارائه شود که وزن‌های مدل ترین‌شده
را به صورت جداگانه دریافت کند و خروجی را بدهد.»

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

# ==============================================================================
# ۱. مشخصات ثابت نرمال‌سازی و ظرفیت تانسور ورودی
# ==============================================================================

MAX_LEN_F1 = 10
MAX_LEN_F2 = 20
MAX_LEN_F3 = 5

MIN_VAL_F1, MAX_VAL_F1 = 1000.0, 10000.0
MIN_VAL_F2, MAX_VAL_F2 = 20000.0, 100000.0
MIN_VAL_F3, MAX_VAL_F3 = 0.1, 1000.0


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
    """تبدیل ورودی با طول متغیر به بردار ورودی ۷۰ بعدی (۳۵ مقدار نرمال‌شده + ۳۵ ماسک)"""
    v1, m1 = encode_feature_vector(q1, MAX_LEN_F1, MIN_VAL_F1, MAX_VAL_F1)
    v2, m2 = encode_feature_vector(q2, MAX_LEN_F2, MIN_VAL_F2, MAX_VAL_F2)
    v3, m3 = encode_feature_vector(q3, MAX_LEN_F3, MIN_VAL_F3, MAX_VAL_F3)
    return np.array(v1 + v2 + v3 + m1 + m2 + m3, dtype=np.float32)


# ==============================================================================
# ۲. تابع اصلی اجرای استنتاج سریع با بارگذاری وزن‌ها
# ==============================================================================

def run_inference(model_path: str = "trained_model.joblib",
                  query_path: str = "query.json",
                  db_path: str = "database.pkl",
                  output_path: str = "top3_results.json") -> List[Dict[str, Any]]:
    """
    بارگذاری وزن‌های از پیش آموزش‌دیده مدل و اجرای فوری استنتاج در کسری از ثانیه
    بدون نیاز به فرآیند آموزش مجدد
    """
    if not os.path.exists(model_path):
        print(f"❌ خطا: فایل وزن‌های مدل ({model_path}) یافت نشد!")
        print("لطفاً ابتدا اسکریپت similarity_nn.py را اجرا کنید تا مدل آموزش داده شده و وزن‌ها ذخیره شوند.")
        sys.exit(1)
        
    if not os.path.exists(query_path):
        print(f"❌ خطا: فایل کوئری ({query_path}) یافت نشد!")
        sys.exit(1)
        
    if not os.path.exists(db_path):
        print(f"❌ خطا: فایل پایگاه‌داده ({db_path}) یافت نشد!")
        sys.exit(1)
        
    # بارگذاری وزن‌های مدل
    print(f"⚡ در حال بارگذاری سریع وزن‌های مدل از {model_path}...")
    model = joblib.load(model_path)
    
    # بارگذاری پایگاه‌داده
    df = pd.read_pickle(db_path)
    
    # بارگذاری داده ورودی
    with open(query_path, "r", encoding="utf-8") as f:
        query = json.load(f)
        
    print(f"📥 داده ورودی از {query_path} بارگذاری شد (سطر منبع: {query.get('source_row_id')}).")
    print(f"   Feature1: {str(query.get('Feature1'))[:70]}...")
    print(f"   Feature2: {str(query.get('Feature2'))[:70]}...")
    print(f"   Feature3: {str(query.get('Feature3'))}")
    
    # انکود داده طول متغیر و استنتاج مستقیم با شبکه
    q1 = parse_query_points(query, "Feature1")
    q2 = parse_query_points(query, "Feature2")
    q3 = parse_query_points(query, "Feature3")
    
    x_input = encode_variable_length_query(q1, q2, q3)
    
    # گذر مستقیم از شبکه (یک گام رو به جلو)
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
        
    # نمایش نتایج
    print("\n" + "=" * 75)
    print("🏆 ۳ سطر برتر پیش‌بینی‌شده توسط شبکه عصبی با استفاده از وزن‌های لودشده:")
    print("=" * 75)
    for item in results:
        print(f"🥇 رتبه {item['rank']}: سطر شماره {item['row_id']:<3} | احتمال شباهت: {item['probability']*100:.2f}%")
        print(f"   بازه‌های Feature1: {item['Feature1'][:65]}...")
        print(f"   بازه‌های Feature2: {item['Feature2'][:65]}...")
        print(f"   بازه‌های Feature3: {item['Feature3']}")
        print("-" * 75)
        
    # ذخیره نتایج
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ نتایج با موفقیت در فایل {output_path} ذخیره گردید.")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="استنتاج سریع با وزن‌های مدل آموزش‌دیده")
    parser.add_argument("--model", type=str, default="trained_model.joblib", help="مسیر فایل وزن‌های مدل")
    parser.add_argument("--query", type=str, default="query.json", help="مسیر فایل کوئری ورودی")
    parser.add_argument("--db", type=str, default="database.pkl", help="مسیر فایل پایگاه‌داده")
    parser.add_argument("--out", type=str, default="top3_results.json", help="مسیر فایل خروجی نتایج")
    
    args = parser.parse_args()
    run_inference(args.model, args.query, args.db, args.out)

