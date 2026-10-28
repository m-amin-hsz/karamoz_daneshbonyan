"""
تولید پایگاه‌داده کارآموزی - قدم ۱
مطابق مشخصات PDF و پارامترهای قابل تنظیم در config.txt

پایگاه‌داده: ۱۰۰ سطر × ۳ ویژگی (Feature1, Feature2, Feature3)
هر ویژگی یک بردار از بازه‌هاست.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
import json
from config_utils import CONFIG

np.random.seed(CONFIG["RANDOM_SEED"])  # برای تکرارپذیری نتایج


def generate_intervals(
    n_min: int,
    n_max: int,
    width_max: float,
    value_min: float,
    value_max: float,
    decimal_places: int = 2
) -> List[Tuple[float, float]]:
    """تولید یک بردار از بازه‌ها با پارامترهای مشخص"""
    n = np.random.randint(n_min, n_max + 1)
    intervals = []
    
    for _ in range(n):
        start = np.random.uniform(value_min, value_max - width_max)
        width = np.random.uniform(0, width_max)
        end = start + width
        
        start = round(start, decimal_places)
        end = round(end, decimal_places)
        
        if start > end:
            start, end = end, start
            
        intervals.append((start, end))
    
    intervals.sort(key=lambda x: x[0])
    return intervals


def intervals_to_string(intervals: List[Tuple[float, float]]) -> str:
    """تبدیل لیست بازه‌ها به رشته با فرمت a-b,c-d,..."""
    return ",".join([f"{start}-{end}" for start, end in intervals])


def calculate_overlap_ratio(intervals1: List[Tuple[float, float]], 
                            intervals2: List[Tuple[float, float]]) -> float:
    """محاسبه نسبت همپوشانی تقریبی بین دو مجموعه بازه"""
    if not intervals1 or not intervals2:
        return 0.0
    
    def total_length(intervals):
        return sum(end - start for start, end in intervals)
    
    overlap_len = 0.0
    for s1, e1 in intervals1:
        for s2, e2 in intervals2:
            overlap_start = max(s1, s2)
            overlap_end = min(e1, e2)
            if overlap_start < overlap_end:
                overlap_len += (overlap_end - overlap_start)
    
    len1 = total_length(intervals1)
    len2 = total_length(intervals2)
    
    if len1 + len2 == 0:
        return 0.0
    
    return overlap_len / ((len1 + len2) / 2)


def generate_database(n_rows: int = None, max_overlap: float = None) -> pd.DataFrame:
    """تولید کامل پایگاه‌داده با استفاده از پارامترهای config.txt"""
    if n_rows is None:
        n_rows = CONFIG["DATABASE_ROWS"]
    if max_overlap is None:
        max_overlap = CONFIG["MAX_OVERLAP_RATIO"]
        
    data = {
        "Feature1": [],
        "Feature2": [],
        "Feature3": []
    }
    
    all_intervals = {
        "Feature1": [],
        "Feature2": [],
        "Feature3": []
    }
    
    for i in range(n_rows):
        # Feature 1
        f1 = generate_intervals(
            n_min=CONFIG["FEATURE1_MIN_INTERVALS"],
            n_max=CONFIG["FEATURE1_MAX_INTERVALS"],
            width_max=CONFIG["FEATURE1_MAX_WIDTH"],
            value_min=CONFIG["FEATURE1_MIN_VAL"],
            value_max=CONFIG["FEATURE1_MAX_VAL"],
            decimal_places=CONFIG["FEATURE1_DECIMALS"]
        )
        
        # Feature 2
        f2 = generate_intervals(
            n_min=CONFIG["FEATURE2_MIN_INTERVALS"],
            n_max=CONFIG["FEATURE2_MAX_INTERVALS"],
            width_max=CONFIG["FEATURE2_MAX_WIDTH"],
            value_min=CONFIG["FEATURE2_MIN_VAL"],
            value_max=CONFIG["FEATURE2_MAX_VAL"],
            decimal_places=CONFIG["FEATURE2_DECIMALS"]
        )
        
        # Feature 3
        f3 = generate_intervals(
            n_min=CONFIG["FEATURE3_MIN_INTERVALS"],
            n_max=CONFIG["FEATURE3_MAX_INTERVALS"],
            width_max=CONFIG["FEATURE3_MAX_WIDTH"],
            value_min=CONFIG["FEATURE3_MIN_VAL"],
            value_max=CONFIG["FEATURE3_MAX_VAL"],
            decimal_places=CONFIG["FEATURE3_DECIMALS"]
        )
        
        # کنترل همپوشانی با سطرهای قبلی
        for feat_name, intervals in [("Feature1", f1), ("Feature2", f2), ("Feature3", f3)]:
            for prev in all_intervals[feat_name]:
                overlap = calculate_overlap_ratio(intervals, prev)
                if overlap > max_overlap:
                    shift = np.random.uniform(-2, 2)
                    intervals = [(round(s + shift, 3), round(e + shift, 3)) for s, e in intervals]
        
        all_intervals["Feature1"].append(f1)
        all_intervals["Feature2"].append(f2)
        all_intervals["Feature3"].append(f3)
        
        data["Feature1"].append(intervals_to_string(f1))
        data["Feature2"].append(intervals_to_string(f2))
        data["Feature3"].append(intervals_to_string(f3))
    
    df = pd.DataFrame(data)
    df.index.name = "row_id"
    return df


def save_database(df: pd.DataFrame, 
                  csv_path: str = None,
                  json_path: str = None,
                  pickle_path: str = None):
    """ذخیره پایگاه‌داده در فرمت‌های مختلف"""
    if csv_path is None: csv_path = CONFIG["DATABASE_CSV"]
    if json_path is None: json_path = CONFIG["DATABASE_JSON"]
    if pickle_path is None: pickle_path = CONFIG["DATABASE_PKL"]
    
    df.to_csv(csv_path, encoding="utf-8-sig")
    print(f"✅ ذخیره شد: {csv_path}")
    
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    print(f"✅ ذخیره شد: {json_path}")
    
    df.to_pickle(pickle_path)
    print(f"✅ ذخیره شد: {pickle_path}")


def preview_database(df: pd.DataFrame, n: int = 5):
    """نمایش چند سطر اول پایگاه‌داده"""
    print("\n" + "="*80)
    print(f"نمونه {n} سطر اول پایگاه‌داده (از مجموع {len(df)} سطر خوانده‌شده از config.txt):")
    print("="*80)
    
    for idx, row in df.head(n).iterrows():
        print(f"\n--- سطر {idx} ---")
        print(f"Feature1: {row['Feature1']}")
        print(f"Feature2: {row['Feature2']}")
        print(f"Feature3: {row['Feature3']}")
    
    print("\n" + "="*80)
    print("آمار تعداد بازه‌ها در هر ویژگی:")
    print("="*80)
    
    for col in ["Feature1", "Feature2", "Feature3"]:
        counts = df[col].apply(lambda x: len(x.split(",")) if x else 0)
        print(f"{col}: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")


if __name__ == "__main__":
    print(f"🚀 شروع تولید پایگاه‌داده {CONFIG['DATABASE_ROWS']} سطری با تنظیمات config.txt...")
    
    db = generate_database()
    preview_database(db, n=5)
    save_database(db)
    
    print("\n✅ پایگاه‌داده با موفقیت ساخته شد!")
