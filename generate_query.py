"""
تولید داده دریافتی (Query) - قدم ۲
مطابق با مشخصات پروژه کارآموزی، اصلاحیه کارفرما و تنظیمات config.txt:
- داده‌های ورودی به شبکه نباید بازه‌ای باشند، بلکه باید به صورت مقادیر عددی (نقاط) از درون بازه‌های پایگاه‌داده انتخاب گردند.
- هر یک از اعضای بردار ویژگی به صورت مستقل به احتمال ۳۰٪ می‌تواند حضور نداشته باشد (Independent Dropout).
- هر عدد دارای ۱٪ خطای تصادفی نسبت به مقدار واقعی است.
- کلیه ثوابت از فایل config.txt خوانده می‌شوند.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import json
from config_utils import CONFIG

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
    """انتخاب یک مقدار عددی تصادفی از درون بازه [start, end]"""
    val = np.random.uniform(start, end)
    return round(float(val), decimal_places)


def apply_1_percent_error(value: float, noise_rate: float = None, decimal_places: int = 4) -> float:
    """اعمال خطای تصادفی به یک عدد بر اساس NOISE_RATE در config.txt"""
    if noise_rate is None:
        noise_rate = CONFIG["NOISE_RATE"]
    error = value * noise_rate * np.random.uniform(-1.0, 1.0)
    return round(float(value + error), decimal_places)


def apply_independent_dropout(values: List[float], drop_prob: float = None) -> List[float]:
    """
    هر یک از اعضای بردار ویژگی به صورت مستقل به احتمال ۳۰٪ می‌تواند حضور نداشته باشد
    (آزمون برنولی مستقل برای هر عضو مطابق اصلاحیه جدید کارفرما و DROPOUT_PROB در config.txt)
    """
    if drop_prob is None:
        drop_prob = CONFIG["DROPOUT_PROB"]
    kept = [v for v in values if np.random.random() >= drop_prob]
    # در صورتی که تصادفاً همه اعضا حذف شوند، حداقل یک عضو را نگه می‌داریم تا بردار خالی نشود
    if not kept and values:
        kept = [values[np.random.randint(0, len(values))]]
    return kept


def generate_feature_points(interval_str: str, 
                            drop_prob: float = None, 
                            noise_rate: float = None, 
                            decimal_places: int = 4) -> Tuple[List[float], str]:
    """تولید بردار مقادیر عددی برای یک ویژگی از روی بازه‌های موجود در پایگاه‌داده"""
    intervals = parse_intervals(interval_str)
    if not intervals:
        return [], ""
    
    # ۱. انتخاب تصادفی عدد از درون بازه
    points = [sample_point_from_interval(s, e, decimal_places) for s, e in intervals]
    
    # ۲. احتمال مستقل عدم حضور برای هر یک از اعضا
    points = apply_independent_dropout(points, drop_prob=drop_prob)
    
    # ۳. اعمال نویز به هر عدد
    noisy_points = [apply_1_percent_error(p, noise_rate=noise_rate, decimal_places=decimal_places) for p in points]
    
    # ساخت رشته خوانا از اعداد (جدا شده با کاما، بدون علامت بازه -)
    points_str = ", ".join([f"{p:.{decimal_places}f}" for p in noisy_points])
    
    return noisy_points, points_str


def generate_query(db_path: str = None) -> Dict:
    """تولید یک نمونه داده دریافتی (Query) بر اساس یک سطر تصادفی از پایگاه‌داده"""
    if db_path is None:
        db_path = CONFIG["DATABASE_PKL"]
        
    df = pd.read_pickle(db_path)
    
    random_idx = int(np.random.randint(0, len(df)))
    original_row = df.iloc[random_idx]
    
    print(f"📌 سطر انتخاب‌شده از پایگاه‌داده برای تولید داده ورودی: سطر شماره {random_idx}")
    print("-" * 75)
    print(f"بازه‌های Feature1 سطر اصلی:\n   {original_row['Feature1']}")
    print(f"بازه‌های Feature2 سطر اصلی:\n   {original_row['Feature2'][:90]}...")
    print(f"بازه‌های Feature3 سطر اصلی:\n   {original_row['Feature3']}")
    print("-" * 75)
    
    f1_pts, f1_str = generate_feature_points(original_row["Feature1"], decimal_places=CONFIG["FEATURE1_DECIMALS"])
    f2_pts, f2_str = generate_feature_points(original_row["Feature2"], decimal_places=CONFIG["FEATURE2_DECIMALS"])
    f3_pts, f3_str = generate_feature_points(original_row["Feature3"], decimal_places=CONFIG["FEATURE3_DECIMALS"])
    
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
               json_path: str = None,
               txt_path: str = None):
    """ذخیره داده دریافتی به صورت فایل متنی و JSON بر اساس مسیرهای config.txt"""
    if json_path is None: json_path = CONFIG["QUERY_JSON"]
    if txt_path is None: txt_path = CONFIG["QUERY_TXT"]
    
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
    print("🚀 شروع تولید داده ورودی شبکه (Query) با تنظیمات config.txt...\n")
    
    query = generate_query()
    
    print("\n🎯 نمونه داده ورودی تولیدشده (مقادیر عددی انتخاب‌شده از بازه‌ها با نویز و خطای مستقل):")
    print("=" * 75)
    print(f"شماره سطر مبدأ : {query['source_row_id']}")
    print(f"Feature1 (اعداد): {query['Feature1']}")
    print(f"Feature2 (اعداد): {query['Feature2'][:90]}...")
    print(f"Feature3 (اعداد): {query['Feature3']}")
    print("=" * 75)
    
    save_query(query)
    print("\n✅ داده ورودی با موفقیت تولید و ذخیره شد!")
