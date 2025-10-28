from modules.evaluation import evaluate_forecast
import numpy as np

true = np.array([0.01, 0.02, 0.015, 0.018, 0.017])
pred = np.array([0.011, 0.019, 0.016, 0.020, 0.018])

print(evaluate_forecast(true, pred))
