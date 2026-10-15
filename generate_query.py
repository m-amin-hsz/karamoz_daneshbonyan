"""
تولید داده دریافتی (Query) - قدم ۲
مطابق با مشخصات پروژه کارآموزی و بازخورد کارفرما:
- داده‌های ورودی به شبکه نباید بازه‌ای باشند، بلکه باید به صورت مقادیر عددی (نقاط) از درون بازه‌های پایگاه‌داده انتخاب گردند.
- با احتمال ۳۰٪ ممکن است یکی از اعضای بردار ویژگی دریافت نشود (حذف شود).
- هر عدد دارای ۱٪ خطای تصادفی نسبت به مقدار واقعی است.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import json

np.random.seed(None)  # هر بار اجرای تصادفی


def parse_intervals(interval_str: str) -> List[Tuple[float, float]]:
    """تبدیل رشته بازه‌ها به لیست تاپل‌های (start, end)"""
    if not interval_str or pd.isna(interval_str):
        return []
    
    intervals = []
    for part in str(interval_str).split(","):
        part = part.strip()
        if "-" in part:
            try:
                start, end = part.split("-")
                intervals.append((float(start), float(end)))
            except ValueError:
                continue
    return intervals


def sample_point_from_interval(start: float, end: float, decimal_places: int = 4) -> float:
    """
    انتخاب یک مقدار عددی تصادفی از درون بازه [start, end]
    (پاسخ مستقیم به خواسته کارفرما مبنی بر نقطه‌ای بودن داده‌های ورودی)
    """
    val = np.random.uniform(start, end)
    return round(float(val), decimal_places)


def apply_1_percent_error(value: float, decimal_places: int = 4) -> float:
    """اعمال خطای تصادفی ۱٪ به یک عدد (مطابق بند ۳ PDF)"""
    error = value * 0.01 * np.random.uniform(-1.0, 1.0)  # محدوده ±۱٪
    return round(float(value + error), decimal_places)


def remove_random_member(values: List[float], prob: float = 0.30) -> List[float]:
    """
    با احتمال ۳۰٪، یکی از اعضای بردار ویژگی دریافت نمی‌شود (مطابق بند ۲ PDF)
    """
    if len(values) <= 1:
        return values  # اگر فقط یک عضو داشته باشیم، آن را حذف نمی‌کنیم
    
    if np.random.random() < prob:
        drop_idx = np.random.randint(0, len(values))
        values = values[:drop_idx] + values[drop_idx + 1:]
    
    return values


def generate_feature_points(interval_str: str, decimal_places: int = 4) -> Tuple[List[float], str]:
    """
    تولید بردار مقادیر عددی برای یک ویژگی از روی بازه‌های موجود در پایگاه‌داده:
    ۱. انتخاب یک عدد تصادفی از درون هر بازه
    ۲. احتمال ۳۰٪ حذف یکی از اعضا
    ۳. اعمال خطای ۱٪ روی هر مقدار
    """
    intervals = parse_intervals(interval_str)
    if not intervals:
        return [], ""
    
    # ۱. انتخاب تصادفی عدد از درون بازه
    points = [sample_point_from_interval(s, e, decimal_places) for s, e in intervals]
    
    # ۲. احتمال ۳۰٪ حذف یکی از اعضای بردار
    points = remove_random_member(points, prob=0.30)
    
    # ۳. اعمال ۱٪ نویز به هر عدد
    noisy_points = [apply_1_percent_error(p, decimal_places) for p in points]
    
    # ساخت رشته خوانا از اعداد (جدا شده با کاما، بدون علامت بازه -)
    points_str = ", ".join([f"{p:.{decimal_places}f}" for p in noisy_points])
    
    return noisy_points, points_str


def generate_query(db_path: str = "database.pkl") -> Dict:
    """
    تولید یک نمونه داده دریافتی (Query) بر اساس یک سطر تصادفی از پایگاه‌داده
    """
    df = pd.read_pickle(db_path)
    
    # انتخاب یک سطر تصادفی
    random_idx = int(np.random.randint(0, len(df)))
    original_row = df.iloc[random_idx]
    
    print(f"📌 سطر انتخاب‌شده از پایگاه‌داده برای تولید داده ورودی: سطر شماره {random_idx}")
    print("-" * 75)
    print(f"بازه‌های Feature1 سطر اصلی:\n   {original_row['Feature1']}")
    print(f"بازه‌های Feature2 سطر اصلی:\n   {original_row['Feature2'][:90]}...")
    print(f"بازه‌های Feature3 سطر اصلی:\n   {original_row['Feature3']}")
    print("-" * 75)
    
    # تولید داده‌های نقطه‌ای برای هر ۳ ویژگی
    f1_pts, f1_str = generate_feature_points(original_row["Feature1"], decimal_places=4)
    f2_pts, f2_str = generate_feature_points(original_row["Feature2"], decimal_places=4)
    f3_pts, f3_str = generate_feature_points(original_row["Feature3"], decimal_places=4)
    
    query = {
        "source_row_id": random_idx,
        "Feature1": f1_str,
        "Feature2": f2_str,
        "Feature3": f3_str,
        "Feature1_values": f1_pts,
        "Feature2_values": f2_pts,
        "Feature3_values": f3_pts,
    }
    
    return query


def save_query(query: Dict, 
               json_path: str = "query.json",
               txt_path: str = "query.txt"):
    """ذخیره داده دریافتی به صورت فایل متنی و JSON"""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(query, f, ensure_ascii=False, indent=2)
    print(f"✅ داده دریافتی با موفقیت ذخیره شد: {json_path}")
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"source_row_id: {query['source_row_id']}\n")
        f.write(f"Feature1: {query['Feature1']}\n")
        f.write(f"Feature2: {query['Feature2']}\n")
        f.write(f"Feature3: {query['Feature3']}\n")
    print(f"✅ داده دریافتی با موفقیت ذخیره شد: {txt_path}")


if __name__ == "__main__":
    print("🚀 شروع تولید داده ورودی شبکه (Query) بر اساس بازه‌های پایگاه‌داده...\n")
    
    query = generate_query(db_path="database.pkl")
    
    print("\n🎯 نمونه داده ورودی تولیدشده (مقادیر عددی انتخاب‌شده از بازه‌ها با نویز و خطا):")
    print("=" * 75)
    print(f"شماره سطر مبدأ : {query['source_row_id']}")
    print(f"Feature1 (اعداد): {query['Feature1']}")
    print(f"Feature2 (اعداد): {query['Feature2'][:90]}...")
    print(f"Feature3 (اعداد): {query['Feature3']}")
    print("=" * 75)
    
    save_query(query)
    print("\n✅ داده ورودی با موفقیت تولید و ذخیره شد!")