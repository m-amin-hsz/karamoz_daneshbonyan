"""
تولید پایگاه‌داده کارآموزی - قدم ۱
مطابق مشخصات PDF

پایگاه‌داده: ۱۰۰ سطر × ۳ ویژگی (Feature1, Feature2, Feature3)
هر ویژگی یک بردار از بازه‌هاست.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
import json

np.random.seed(42)  # برای تکرارپذیری نتایج


def generate_intervals(
    n_min: int,
    n_max: int,
    width_max: float,
    value_min: float,
    value_max: float,
    decimal_places: int = 2
) -> List[Tuple[float, float]]:
    """
    تولید یک بردار از بازه‌ها
    
    پارامترها:
        n_min, n_max: تعداد اعضا (تصادفی بین این دو)
        width_max: حداکثر عرض بازه
        value_min, value_max: محدوده اعداد
        decimal_places: تعداد رقم اعشار
    """
    n = np.random.randint(n_min, n_max + 1)
    intervals = []
    
    for _ in range(n):
        # انتخاب نقطه شروع در محدوده مجاز
        # اطمینان از اینکه end از value_max بیشتر نشود
        start = np.random.uniform(value_min, value_max - width_max)
        width = np.random.uniform(0, width_max)
        end = start + width
        
        # گرد کردن
        start = round(start, decimal_places)
        end = round(end, decimal_places)
        
        # اطمینان از start <= end
        if start > end:
            start, end = end, start
            
        intervals.append((start, end))
    
    # مرتب‌سازی بر اساس شروع بازه (اختیاری ولی تمیزتر است)
    intervals.sort(key=lambda x: x[0])
    
    return intervals


def intervals_to_string(intervals: List[Tuple[float, float]]) -> str:
    """تبدیل لیست بازه‌ها به رشته با فرمت a-b,c-d,..."""
    return ",".join([f"{start}-{end}" for start, end in intervals])


def calculate_overlap_ratio(intervals1: List[Tuple[float, float]], 
                            intervals2: List[Tuple[float, float]]) -> float:
    """
    محاسبه نسبت همپوشانی تقریبی بین دو مجموعه بازه
    (برای کنترل همپوشانی ۳۰٪)
    """
    if not intervals1 or not intervals2:
        return 0.0
    
    def total_length(intervals):
        return sum(end - start for start, end in intervals)
    
    # محاسبه همپوشانی زوجی
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
    
    # نسبت همپوشانی نسبت به میانگین طول‌ها
    return overlap_len / ((len1 + len2) / 2)


def generate_database(n_rows: int = 100, max_overlap: float = 0.30) -> pd.DataFrame:
    """
    تولید کامل پایگاه‌داده با کنترل همپوشانی تقریبی
    """
    data = {
        "Feature1": [],
        "Feature2": [],
        "Feature3": []
    }
    
    # ذخیره بازه‌ها به صورت لیست برای کنترل همپوشانی
    all_intervals = {
        "Feature1": [],
        "Feature2": [],
        "Feature3": []
    }
    
    for i in range(n_rows):
        # ---------- Feature1 ----------
        # تعداد: ۱–۱۰ | عرض: ۰–۵ | محدوده: ۱۰۰۰–۱۰۰۰۰
        f1 = generate_intervals(
            n_min=1, n_max=10,
            width_max=5.0,
            value_min=1000.0,
            value_max=10000.0,
            decimal_places=2
        )
        
        # ---------- Feature2 ----------
        # تعداد: ۱–۲۰ | عرض: ۰–۱۰ | محدوده: ۲۰۰۰۰–۱۰۰۰۰۰
        f2 = generate_intervals(
            n_min=1, n_max=20,
            width_max=10.0,
            value_min=20000.0,
            value_max=100000.0,
            decimal_places=2
        )
        
        # ---------- Feature3 ----------
        # تعداد: ۱–۵ | عرض: ۰–۱ | محدوده: ۰.۱–۱۰۰۰
        f3 = generate_intervals(
            n_min=1, n_max=5,
            width_max=1.0,
            value_min=0.1,
            value_max=1000.0,
            decimal_places=3
        )
        
        # کنترل همپوشانی تقریبی با سطرهای قبلی (اختیاری ولی مطابق PDF)
        # اگر همپوشانی خیلی زیاد بود، کمی جابه‌جا می‌کنیم
        for feat_name, intervals in [("Feature1", f1), ("Feature2", f2), ("Feature3", f3)]:
            for prev in all_intervals[feat_name]:
                overlap = calculate_overlap_ratio(intervals, prev)
                if overlap > max_overlap:
                    # کمی شیفت تصادفی برای کاهش همپوشانی
                    shift = np.random.uniform(-2, 2)
                    intervals = [(round(s + shift, 3), round(e + shift, 3)) for s, e in intervals]
        
        # ذخیره
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
                  csv_path: str = "database.csv",
                  json_path: str = "database.json",
                  pickle_path: str = "database.pkl"):
    """ذخیره پایگاه‌داده در فرمت‌های مختلف"""
    
    # CSV (خوانا برای انسان)
    df.to_csv(csv_path, encoding="utf-8-sig")
    print(f"✅ ذخیره شد: {csv_path}")
    
    # JSON (برای استفاده در برنامه‌های دیگر)
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    print(f"✅ ذخیره شد: {json_path}")
    
    # Pickle (سریع‌ترین برای پایتون)
    df.to_pickle(pickle_path)
    print(f"✅ ذخیره شد: {pickle_path}")


def preview_database(df: pd.DataFrame, n: int = 5):
    """نمایش چند سطر اول پایگاه‌داده"""
    print("\n" + "="*80)
    print(f"نمونه {n} سطر اول پایگاه‌داده (از مجموع {len(df)} سطر):")
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
    print("🚀 شروع تولید پایگاه‌داده ۱۰۰ سطری...")
    
    db = generate_database(n_rows=100, max_overlap=0.30)
    
    preview_database(db, n=5)
    
    save_database(
        db,
        csv_path="database.csv",
        json_path="database.json",
        pickle_path="database.pkl"
    )
    
    print("\n✅ پایگاه‌داده با موفقیت ساخته شد!")
    print("فایل‌های خروجی در پوشه artifacts ذخیره شدند.")