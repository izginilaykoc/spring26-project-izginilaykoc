"""Per-model train+predict scripts. Each module exposes:

    predict(predict_date: pd.Timestamp,
            as_of_dt: pd.Timestamp | None = None) -> list[str]

so the backtest harness and `classify.py` can call any of them uniformly.
"""
