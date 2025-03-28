def eval():
    from mlops_gcp_client import MLOpsGCPClient
    
    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd
    import numpy as np

    import torch 
    import torch.nn as nn

    from datetime import datetime
    import pytz
    
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst).strftime("%Y-%m-%d_%H%M%S")

    generation = pd.read_csv("data/validation/Plant_1_Generation_Data.csv")
    weather = pd.read_csv("data/validation/Plant_1_Weather_Sensor_Data.csv")

    generation['DATE_TIME'] = pd.to_datetime(generation['DATE_TIME'], dayfirst=True)
    weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'], dayfirst=False)

    generation_source_key = list(generation['SOURCE_KEY'].unique())
    inv = generation[generation['SOURCE_KEY']==generation_source_key[0]]
    mask = ((weather['DATE_TIME'] >= min(inv["DATE_TIME"])) & (weather['DATE_TIME'] <= max(inv["DATE_TIME"])))
    weather_filtered = weather.loc[mask]

    df = inv.merge(weather_filtered, on="DATE_TIME", how='left')

    df_timestamp = df[["DATE_TIME"]]
    df_ = df[["AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]

    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(df_)
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    class LSTMAutoencoder(nn.Module):
        def __init__(self, seq_len, n_features):
            super(LSTMAutoencoder, self).__init__()
            # 인코더
            self.encoder_lstm1 = nn.LSTM(input_size=n_features, hidden_size=16, batch_first=True)
            self.encoder_lstm2 = nn.LSTM(input_size=16, hidden_size=4, batch_first=True)

            # 디코더
            self.decoder_lstm1 = nn.LSTM(input_size=4, hidden_size=4, batch_first=True)
            self.decoder_lstm2 = nn.LSTM(input_size=4, hidden_size=16, batch_first=True)
            self.decoder_output = nn.Linear(16, n_features)
            self.seq_len = seq_len

        def forward(self, x):
            x, _ = self.encoder_lstm1(x)
            x, (h_n, _) = self.encoder_lstm2(x)

            x = h_n.repeat(self.seq_len, 1, 1).permute(1, 0, 2)

            x, _ = self.decoder_lstm1(x)
            x, _ = self.decoder_lstm2(x)
            x = self.decoder_output(x)
            return x
        
    client = MLOpsGCPClient("mlops-models-bucket")
    client.download_model("model.pt", "models/model.pt")
        
    model = LSTMAutoencoder(seq_len=X_test.shape[1], n_features=X_test.shape[2])
    model.load_state_dict(torch.load("models/model.pt", weights_only=True))
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        X_pred = model(X_test_tensor).numpy()
    
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred = scaler.inverse_transform(X_pred)
    X_pred = pd.DataFrame(X_pred, columns=df_.columns)
    X_pred.index = df_.index
    
    scores = X_pred.copy()
    scores['datetime'] = df_timestamp
    scores['real AC'] = df_['AC_POWER'].values
    scores["loss_mae"] = (scores['real AC'] - scores['AC_POWER']).abs()
    scores['Threshold'] = 200
    scores['Anomaly'] = np.where(scores["loss_mae"] > scores["Threshold"], 1, 0)

    anomalies = scores[scores['Anomaly'] == 1][['real AC']]
    anomalies = anomalies.rename(columns={'real AC':'anomalies'})
    scores = scores.merge(anomalies, left_index=True, right_index=True, how='left')

    scores[(scores['Anomaly']==1)&(scores['datetime'].notnull())].to_csv(f"data/output/{now}_anomalies.csv")

if __name__=="__main__":
    eval()