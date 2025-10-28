# =========================================
# modules/evaluation.py
# Forecast Evaluation Metrics
# =========================================

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_forecast(true, pred):
    """Compute standard forecast accuracy metrics."""
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)
    return {
        "MAE": round(mae, 6),
        "RMSE": round(rmse, 6),
        "R²": round(r2, 6)
    }
