import numpy as np
API_KEY = '7fa6fe05da148'
import datetime
import os

def get_data_path(state, sig, start_date, end_date):
    assert type(start_date)==datetime.date
    assert type(end_date)==datetime.date
    file_path = os.path.join("data", f"{state}-{sig}-from-{start_date}-to-{end_date}.pkl")
    return file_path

SIGNALS = np.array(["smoothed_whesitancy_reason_sideeffects", 
           "smoothed_whesitancy_reason_allergic", 
           "smoothed_whesitancy_reason_ineffective",
           "smoothed_whesitancy_reason_unnecessary",
           "smoothed_whesitancy_reason_dislike_vaccines",
           "smoothed_whesitancy_reason_not_recommended",
           "smoothed_whesitancy_reason_wait_safety",
           "smoothed_whesitancy_reason_low_priority",
           "smoothed_whesitancy_reason_cost",
           "smoothed_whesitancy_reason_distrust_vaccines",
           "smoothed_whesitancy_reason_distrust_gov",
           "smoothed_whesitancy_reason_health_condition",
           "smoothed_whesitancy_reason_pregnant",
           "smoothed_whesitancy_reason_religious",
           "smoothed_whesitancy_reason_other"])

SHORT_SIGNALS = np.array(["sideeffects", 
           "allergic", 
           "ineffective",
           "unnecessary",
           "dislike_vaccines",
           "not_recommended",
           "wait_safety",
           "low_priority",
           "cost",
           "distrust_vaccines",
           "distrust_gov",
           "health_condition",
           "pregnant",
           "religious",
           "other"])