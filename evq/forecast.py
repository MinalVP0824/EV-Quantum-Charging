"""AI demand forecaster.

Gradient-boosted trees predict next-day hourly charging demand from calendar
features, temperature and lagged demand. Two extra quantile models give a
P10 to P90 band so the optimizer can plan either for the expected load or for
a risk-averse (P90) load.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

FEATURES = ["hour", "dow", "weekend", "temp_c", "lag_24", "lag_168", "roll_24"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    out["hour"] = ts.dt.hour
    out["dow"] = ts.dt.dayofweek
    out["weekend"] = (out["dow"] >= 5).astype(int)
    out["lag_24"] = out["demand_kw"].shift(24)
    out["lag_168"] = out["demand_kw"].shift(168)
    out["roll_24"] = out["demand_kw"].shift(24).rolling(24).mean()
    return out.dropna().reset_index(drop=True)


@dataclass
class ForecastResult:
    metrics: dict
    test: pd.DataFrame      # held-out week: timestamp, actual, pred, naive
    target_day: pd.DataFrame  # day to schedule: timestamp, actual, pred, p10, p90
    history: pd.DataFrame   # recent actuals before the target day


def train_and_forecast(df: pd.DataFrame, test_days: int = 7, seed: int = 0) -> ForecastResult:
    feat = make_features(df)
    n_day = 24
    target = feat.iloc[-n_day:]
    test = feat.iloc[-n_day * (test_days + 1):-n_day]
    train = feat.iloc[:-n_day * (test_days + 1)]

    def fit(loss="squared_error", q=None, data=train):
        kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=seed)
        if loss == "quantile":
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q, **kw)
        else:
            m = HistGradientBoostingRegressor(**kw)
        return m.fit(data[FEATURES], data["demand_kw"])

    # 1) honest evaluation on a held-out week
    model = fit()
    test_pred = model.predict(test[FEATURES])
    naive = test["lag_24"].values
    mae_model = mean_absolute_error(test["demand_kw"], test_pred)
    mae_naive = mean_absolute_error(test["demand_kw"], naive)

    # 2) refit on everything before the target day and forecast it
    full = feat.iloc[:-n_day]
    model = fit(data=full)
    p10 = fit("quantile", 0.1, full)
    p90 = fit("quantile", 0.9, full)
    X = target[FEATURES]
    tgt = pd.DataFrame({
        "timestamp": target["timestamp"].values,
        "actual": target["demand_kw"].values,
        "pred": model.predict(X),
        "p10": p10.predict(X),
        "p90": p90.predict(X),
    })
    tgt["p10"] = np.minimum(tgt["p10"], tgt["pred"])
    tgt["p90"] = np.maximum(tgt["p90"], tgt["pred"])

    metrics = {
        "mae_model_kw": float(mae_model),
        "mae_seasonal_naive_kw": float(mae_naive),
        "improvement_pct": float(100 * (1 - mae_model / mae_naive)),
        "target_day_mae_kw": float(mean_absolute_error(tgt["actual"], tgt["pred"])),
    }
    test_df = pd.DataFrame({
        "timestamp": test["timestamp"].values,
        "actual": test["demand_kw"].values,
        "pred": test_pred,
        "naive": naive,
    })
    hist = feat.iloc[-n_day * 4:-n_day][["timestamp", "demand_kw"]].reset_index(drop=True)
    return ForecastResult(metrics, test_df, tgt, hist)


def slot_loads(fc: ForecastResult, hours, use: str = "pred"):
    """Forecast base load (kW) for the given clock hours of the target day."""
    day = fc.target_day.copy()
    day["hour"] = pd.to_datetime(day["timestamp"]).dt.hour
    return [float(day.loc[day["hour"] == h, use].iloc[0]) for h in hours]
