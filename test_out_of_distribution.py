"""
اسکریپت تست و اعتبارسنجی داده‌های خارج از پایگاه‌داده (Out-of-Distribution Test)
مطابق با نیازمندی شماره ۴ کارفرما:
«یک تست قرار داده شود که داده‌ای خارج از پایگاه‌داده ارائه شود و احتمال شباهت آن بیان شود
و شباهت‌های کمتر از ۳۰ درصد به عنوان عدم وجود در پایگاه‌داده بیان شود.»
"""

import numpy as np
import pandas as pd
import json
from typing import List, Tuple, Dict, Any
from config_utils import CONFIG
from inference import parse_query_points, compute_exact_absolute_similarity, run_inference_on_query


def generate_out_of_distribution_query(mode: str = "completely_outside") -> Dict[str, Any]:
    """
    تولید یک نمونه کوئری که کاملاً خارج از توزیع و بازه‌های پایگاه‌داده است:
    - مقادیر عددی انتخاب‌شده در بازه‌های هیچ سطری از پایگاه‌داده قرار ندارند
    """
    if mode == "completely_outside":
        # اعدادی کاملاً خارج از کرانه‌های تعریف‌شده
        f1_pts = [15250.45, 18340.20]       # مجاز: ۱۰۰۰ تا ۱۰۰۰۰
        f2_pts = [5200.10, 8900.50, 12400.0] # مجاز: ۲۰۰۰۰ تا ۱۰۰۰۰۰
        f3_pts = [2500.80]                  # مجاز: ۰.۱ تا ۱۰۰۰
    else:
        # اعدادی در محدوده کرانه‌ها اما در شکاف‌های خالی بین بازه‌های تمام سطرها
        f1_pts = [1050.25, 1150.80]
        f2_pts = [21050.40, 21950.80]
        f3_pts = [15.50]
        
    f1_str = ", ".join([f"{p:.2f}" for p in f1_pts])
    f2_str = ", ".join([f"{p:.2f}" for p in f2_pts])
    f3_str = ", ".join([f"{p:.2f}" for p in f3_pts])
    
    return {
        "source_row_id": -1,  # خارج از پایگاه داده (OOD)
        "description": "Out-of-Distribution Synthetic Query",
        "Feature1": f1_str,
        "Feature2": f2_str,
        "Feature3": f3_str,
        "Feature1_values": f1_pts,
        "Feature2_values": f2_pts,
        "Feature3_values": f3_pts,
    }


def run_ood_benchmark():
    """اجرای آزمون مقایسه‌ای: داده معتبر پایگاه‌داده در برابر داده خارج از پایگاه‌داده"""
    print("=" * 80)
    print("🧪 اجرای آزمون تشخیص داده خارج از پایگاه‌داده (Out-of-Distribution Validation)")
    print(f"   آستانه تشخیص عدم وجود در پایگاه‌داده: {CONFIG['SIMILARITY_THRESHOLD']*100:.1f}%")
    print("=" * 80)
    
    # بارگذاری پایگاه‌داده
    df = pd.read_pickle(CONFIG["DATABASE_PKL"])
    
    # --------------------------------------------------------------------------
    # آزمون اول: داده معتبر از پایگاه‌داده (In-Distribution Query)
    # --------------------------------------------------------------------------
    print("\n🟢 آزمون اول: ارائه داده معتبر برگرفته از پایگاه‌داده...")
    with open(CONFIG["QUERY_JSON"], "r", encoding="utf-8") as f:
        valid_query = json.load(f)
        
    print(f"   سطر منبع کوئری: سطر {valid_query.get('source_row_id')}")
    print(f"   Feature1: {str(valid_query.get('Feature1'))[:65]}...")
    
    result_valid = run_inference_on_query(valid_query, df)
    
    print("\n   نتیجه بررسی داده معتبر:")
    top1 = result_valid["top_matches"][0]
    print(f"   🥇 شبیه‌ترین سطر: سطر شماره {top1['row_id']} با شباهت مطلق {top1['absolute_similarity_percent']:.2f}%")
    print(f"   وضعیت پذیرش در سیستم: {result_valid['status_text']}")
    
    # --------------------------------------------------------------------------
    # آزمون دوم: داده کاملاً خارج از پایگاه‌داده (Out-of-Distribution Query)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("🔴 آزمون دوم: ارائه داده کاملاً خارج از پایگاه‌داده (OOD Query)...")
    ood_query = generate_out_of_distribution_query(mode="completely_outside")
    
    print(f"   Feature1 (اعداد خارج از محدوده): {ood_query['Feature1']}")
    print(f"   Feature2 (اعداد خارج از محدوده): {ood_query['Feature2']}")
    print(f"   Feature3 (اعداد خارج از محدوده): {ood_query['Feature3']}")
    
    result_ood = run_inference_on_query(ood_query, df)
    
    print("\n   نتیجه بررسی داده خارج از پایگاه‌داده:")
    top1_ood = result_ood["top_matches"][0]
    print(f"   بیشترین شباهت به دست آمده: سطر {top1_ood['row_id']} با شباهت مطلق {top1_ood['absolute_similarity_percent']:.2f}%")
    print(f"   وضعیت نهایی: {result_ood['status_text']}")
    
    if result_ood["is_in_database"] is False:
        print("   ✅ آزمون با موفقیت پاس شد: داده به درستی به عنوان «عدم وجود در پایگاه‌داده» رد گردید.")
    else:
        print("   ❌ خطا در ارزیابی آستانه!")
        
    # ذخیره کوئری OOD برای استفاده‌های بعدی
    with open("query_ood_test.json", "w", encoding="utf-8") as f:
        json.dump(ood_query, f, ensure_ascii=False, indent=2)
    print("   فایل query_ood_test.json ذخیره شد.")
    print("=" * 80)


if __name__ == "__main__":
    run_ood_benchmark()

