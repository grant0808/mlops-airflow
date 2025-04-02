def train():
    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from mlops_gcp_client import MLOpsGCPClient

    epochs = 10
    batch_size = 20
    learning_rate = 0.01

    print('start training ...')

    generation = pd.read_csv("data/train/Plant_1_Generation_Data.csv")
    weather = pd.read_csv("data/train/Plant_1_Weather_Sensor_Data.csv")

    generation['DATE_TIME'] = pd.to_datetime(generation['DATE_TIME'], dayfirst=True)
    weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'], dayfirst=False)

    generation_source_key = list(generation['SOURCE_KEY'].unique())
    inv = generation[generation['SOURCE_KEY']==generation_source_key[0]]
    mask = ((weather['DATE_TIME'] >= min(inv["DATE_TIME"])) & (weather['DATE_TIME'] <= max(inv["DATE_TIME"])))
    weather_filtered = weather.loc[mask]

    df = inv.merge(weather_filtered, on="DATE_TIME", how='left')
    df = df[['DATE_TIME', 'AC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]

    df_ = df[["AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
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
        
    model = LSTMAutoencoder(seq_len=X_train.shape[1], n_features=X_train.shape[2])
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, _ in dataloader:
            output = model(batch_x)
            loss = criterion(output, batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), f"models/model.pt")

    client = MLOpsGCPClient("mlops-models-bucket")
    client.upload_model("model.pt","models")
    

if __name__=="__main__":
    train()