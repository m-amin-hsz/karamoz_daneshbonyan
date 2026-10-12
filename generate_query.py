"""
تولید داده دریافتی (Query) - قدم ۲
مطابق مشخصات PDF

قوانین:
1. یک سطر تصادفی از پایگاه‌داده انتخاب می‌شود
2. با احتمال ۳۰٪ یکی از اعضای هر بردار حذف می‌شود
3. هر عدد دارای ۱٪ خطا نسبت به مقدار واقعی است
"""

import numpy as np
import pandas as pd
import random
from typing import List, Tuple

np.random.seed(None)  # هر بار نتایج متفاوت


def parse_intervals(interval_str: str) -> List[Tuple[float, float]]:
    """تبدیل رشته بازه‌ها به لیست تاپل (start, end)"""
    if not interval_str or pd.isna(interval_str):
        return []
    
    intervals = []
    for part in interval_str.split(","):
        part = part.strip()
        if "-" in part:
            try:
                start, end = part.split("-")
                intervals.append((float(start), float(end)))
            except ValueError:
                continue
    return intervals


def intervals_to_string(intervals: List[Tuple[float, float]]) -> str:
    """تبدیل لیست بازه‌ها به رشته"""
    return ",".join([f"{start}-{end}" for start, end in intervals])


def apply_1_percent_error(value: float) -> float:
    """اعمال خطای ۱٪ به یک عدد"""
    error = value * 0.01 * np.random.uniform(-1, 1)  # ±۱٪
    return round(value + error, 4)


def remove_random_member(intervals: List[Tuple[float, float]], prob: float = 0.30) -> List[Tuple[float, float]]:
    """
    با احتمال مشخص، یکی از اعضای بردار را حذف می‌کند
    """
    if len(intervals) <= 1:
        return intervals  # اگر فقط یک عضو باشد حذف نمی‌کنیم
    
    if np.random.random() < prob:
        # یک عضو تصادفی را حذف کن
        idx = np.random.randint(0, len(intervals))
        intervals = intervals[:idx] + intervals[idx+1:]
    
    return intervals


def apply_noise_to_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    اعمال نویز روی بازه‌ها:
    - خطای ۱٪ روی هر عدد
    - احتمال ۳۰٪ حذف یک عضو
    """
    # اول خطای ۱٪ را اعمال کن
    noisy = []
    for start, end in intervals:
        new_start = apply_1_percent_error(start)
        new_end = apply_1_percent_error(end)
        # اطمینان از اینکه start <= end باقی بماند
        if new_start > new_end:
            new_start, new_end = new_end, new_start
        noisy.append((new_start, new_end))
    
    # سپس احتمال حذف یک عضو
    noisy = remove_random_member(noisy, prob=0.30)
    
    return noisy


def generate_query(db_path: str = "database.pkl") -> dict:
    """
    تولید یک داده دریافتی (Query) از روی پایگاه‌داده
    """
    # بارگذاری پایگاه‌داده
    df = pd.read_pickle(db_path)
    
    # انتخاب یک سطر تصادفی
    random_idx = np.random.randint(0, len(df))
    original_row = df.iloc[random_idx]
    
    print(f"📌 سطر اصلی انتخاب‌شده از پایگاه‌داده: سطر شماره {random_idx}")
    print("-" * 60)
    print(f"Feature1 اصلی: {original_row['Feature1']}")
    print(f"Feature2 اصلی: {original_row['Feature2']}")
    print(f"Feature3 اصلی: {original_row['Feature3']}")
    print("-" * 60)
    
    # تبدیل به بازه و اعمال نویز
    f1_intervals = parse_intervals(original_row["Feature1"])
    f2_intervals = parse_intervals(original_row["Feature2"])
    f3_intervals = parse_intervals(original_row["Feature3"])
    
    f1_noisy = apply_noise_to_intervals(f1_intervals)
    f2_noisy = apply_noise_to_intervals(f2_intervals)
    f3_noisy = apply_noise_to_intervals(f3_intervals)
    
    query = {
        "source_row_id": int(random_idx),
        "Feature1": intervals_to_string(f1_noisy),
        "Feature2": intervals_to_string(f2_noisy),
        "Feature3": intervals_to_string(f3_noisy),
        "Feature1_intervals": f1_noisy,
        "Feature2_intervals": f2_noisy,
        "Feature3_intervals": f3_noisy,
    }
    
    return query


def save_query(query: dict, 
               json_path: str = "query.json",
               txt_path: str = "query.txt"):
    """ذخیره داده دریافتی"""
    import json
    
    # نسخه ساده برای ذخیره (بدون لیست‌های داخلی)
    simple_query = {
        "source_row_id": query["source_row_id"],
        "Feature1": query["Feature1"],
        "Feature2": query["Feature2"],
        "Feature3": query["Feature3"]
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(simple_query, f, ensure_ascii=False, indent=2)
    print(f"✅ ذخیره شد: {json_path}")
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"source_row_id: {query['source_row_id']}\n")
        f.write(f"Feature1: {query['Feature1']}\n")
        f.write(f"Feature2: {query['Feature2']}\n")
        f.write(f"Feature3: {query['Feature3']}\n")
    print(f"✅ ذخیره شد: {txt_path}")


if __name__ == "__main__":
    print("🚀 شروع تولید داده دریافتی (Query)...\n")
    
    query = generate_query(db_path="database.pkl")
    
    print("\n🎯 داده دریافتی تولیدشده (با نویز):")
    print("=" * 60)
    print(f"Feature1: {query['Feature1']}")
    print(f"Feature2: {query['Feature2']}")
    print(f"Feature3: {query['Feature3']}")
    print("=" * 60)
    
    save_query(query)
    
    print("\n✅ داده دریافتی با موفقیت ساخته شد!")