# سیستم شباهت‌یابی مبتنی بر شبکه عصبی

پروژه کارآموزی برای ساخت پایگاه‌داده بازه‌ای و یافتن شبیه‌ترین سطرها با استفاده از شبکه عصبی.

## ساختار پروژه

| فایل | توضیح |
|------|-------|
| `generate_database.py` | ساخت پایگاه‌داده ۱۰۰ سطری |
| `generate_query.py` | تولید داده دریافتی (Query) با نویز |
| `similarity_nn.py` | الگوریتم شباهت‌یابی با شبکه عصبی MLP |
| `database.csv` | پایگاه‌داده |
| `query.json` | نمونه داده دریافتی |
| `top3_results.json` | نتایج نهایی |

## نحوه اجرا

```bash
# ساخت محیط مجازی
python -m venv venv
source venv/bin/activate   # در ویندوز: venv\Scripts\activate

# نصب کتابخانه‌ها
pip install pandas numpy scikit-learn

# اجرای مراحل
python generate_database.py
python generate_query.py
python similarity_nn.py