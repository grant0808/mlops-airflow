def is_model_drift():
    import glob
    import os
    import pandas as pd

    DRAFT_THRESHOLD = 25

    anomalies_files = glob.glob("data/output/*_anomalies.csv")
    anomalies_files.sort(key=os.path.getmtime)
    print(anomalies_files)
    latest_anomalies_file = anomalies_files[-1]

    df = pd.read_csv(latest_anomalies_file)

    return len(df) > DRAFT_THRESHOLD