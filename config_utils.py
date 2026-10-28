"""
ماژول لودر فایل پیکربندی (config_utils.py)
وظیفه: خواندن پارامترهای ثابت از فایل config.txt با قابلیت اعمال مقادیر پیش‌فرض
"""

import os
from typing import Dict, Any

DEFAULT_CONFIG: Dict[str, Any] = {
    "DATABASE_ROWS": 100,
    "MAX_OVERLAP_RATIO": 0.30,
    
    "FEATURE1_MIN_VAL": 1000.0,
    "FEATURE1_MAX_VAL": 10000.0,
    "FEATURE1_MIN_INTERVALS": 1,
    "FEATURE1_MAX_INTERVALS": 10,
    "FEATURE1_MAX_WIDTH": 5.0,
    "FEATURE1_DECIMALS": 2,
    
    "FEATURE2_MIN_VAL": 20000.0,
    "FEATURE2_MAX_VAL": 100000.0,
    "FEATURE2_MIN_INTERVALS": 1,
    "FEATURE2_MAX_INTERVALS": 20,
    "FEATURE2_MAX_WIDTH": 10.0,
    "FEATURE2_DECIMALS": 2,
    
    "FEATURE3_MIN_VAL": 0.1,
    "FEATURE3_MAX_VAL": 1000.0,
    "FEATURE3_MIN_INTERVALS": 1,
    "FEATURE3_MAX_INTERVALS": 5,
    "FEATURE3_MAX_WIDTH": 1.0,
    "FEATURE3_DECIMALS": 3,
    
    "DROPOUT_PROB": 0.30,
    "NOISE_RATE": 0.01,
    
    "SAMPLES_PER_CLASS": 80,
    "HIDDEN_LAYER_1": 256,
    "HIDDEN_LAYER_2": 128,
    "LEARNING_RATE": 0.001,
    "MAX_ITER": 300,
    "RANDOM_SEED": 42,
    
    "SIMILARITY_THRESHOLD": 0.30,
    
    "DATABASE_PKL": "database.pkl",
    "DATABASE_CSV": "database.csv",
    "DATABASE_JSON": "database.json",
    "QUERY_JSON": "query.json",
    "QUERY_TXT": "query.txt",
    "MODEL_SAVE_PATH": "trained_model.joblib",
    "TOP3_RESULTS_JSON": "top3_results.json",
    "DATABASE_3D_PLOT": "database_3d_visualization.png",
    "ACCURACY_PLOT": "accuracy_evaluation_plot.png",
    "ARCHITECTURE_DIAGRAM": "network_architecture_diagram.png",
}


def load_config(config_path: str = "config.txt") -> Dict[str, Any]:
    """خواندن پارامترها از فایل متنی و تبدیل به انواع داده مناسب"""
    cfg = dict(DEFAULT_CONFIG)
    if not os.path.exists(config_path):
        return cfg
    
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                # تبدیل نوع
                if key in cfg:
                    orig = cfg[key]
                    if isinstance(orig, int):
                        try: cfg[key] = int(val)
                        except: pass
                    elif isinstance(orig, float):
                        try: cfg[key] = float(val)
                        except: pass
                    elif isinstance(orig, bool):
                        cfg[key] = val.lower() in ("true", "1", "yes")
                    else:
                        cfg[key] = val
                else:
                    cfg[key] = val
    return cfg


# شیء گلوبال تنظیمات
CONFIG = load_config()

