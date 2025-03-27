import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# In[3]:


generation = pd.read_csv("./data/Plant_1_Generation_Data.csv")
weather = pd.read_csv("./data/Plant_1_Weather_Sensor_Data.csv")

# In[4]:


generation.head()

# In[5]:


weather.head()

# In[6]:


generation.info()

# In[7]:


weather.info()

# In[8]:


generation['DATE_TIME'] = pd.to_datetime(generation['DATE_TIME'], dayfirst=True)
weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'], dayfirst=True)

# In[9]:


generation_source_key = list(generation['SOURCE_KEY'].unique())
print('generation source key 갯수 :', len(generation_source_key))

# In[10]:


inv = generation[generation['SOURCE_KEY']==generation_source_key[0]]
mask = ((weather['DATE_TIME'] >= min(inv["DATE_TIME"])) & (weather['DATE_TIME'] <= max(inv["DATE_TIME"])))
weather_filtered = weather.loc[mask]

# In[11]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=inv["DATE_TIME"], y=inv["AC_POWER"],
                    mode='lines',
                    name='AC Power'))

fig.add_trace(go.Scatter(x=weather_filtered["DATE_TIME"], y=weather_filtered["IRRADIATION"],
                    mode='lines',
                    name='Irradiation', 
                    yaxis='y2'))

fig.update_layout(title_text="Irradiation vs AC POWER",
                  yaxis1=dict(title="AC Power in kW",
                              side='left'),
                  yaxis2=dict(title="Irradiation index",
                              side='right',
                              anchor="x",
                              overlaying="y"
                             ))

fig.show()

# In[12]:


df = inv.merge(weather_filtered, on="DATE_TIME", how='left')
df = df[['DATE_TIME', 'AC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]
df.head()

# In[13]:


df.info()

# In[14]:


df_timestamp = df[["DATE_TIME"]]
df_ = df[["AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]

# In[15]:


train_prp = .6
train = df_.loc[:df_.shape[0]*train_prp]
test = df_.loc[df_.shape[0]*train_prp:]

# In[16]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train)
X_test = scaler.transform(test)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# ## Model design

# In[17]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# In[18]:


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

# In[19]:


def model_traning(epochs, learning_rate, dataloader):
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

    return model

# ## Model Optimization

# 

# In[37]:


import mlflow
from mlflow.data.pandas_dataset import PandasDataset

mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.create_experiment("mlflow-lab")

# In[38]:


mlflow.set_experiment("mlflow-lab")

# In[39]:


def mae_visualization(scores):
    scores['datetime'] = df_timestamp.loc[1893:].values
    scores['real AC'] = test['AC_POWER'].values
    scores["loss_mae"] = (scores['real AC'] - scores['AC_POWER']).abs()
    scores['Threshold'] = 200
    scores['Anomaly'] = np.where(scores["loss_mae"] > scores["Threshold"], 1, 0)

    plt.figure(figsize=(12, 6))
    plt.plot(scores['datetime'], scores['loss_mae'], label='Loss (MAE)')
    plt.plot(scores['datetime'], scores['Threshold'], label='Threshold', linestyle='--')
    plt.title("Error Timeseries and Threshold")
    plt.xlabel("DateTime")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./plot/model_mae_plot.png")

# In[40]:


def anomaly_visualization(scores):
    anomalies = scores[scores['Anomaly'] == 1][['real AC']]
    anomalies = anomalies.rename(columns={'real AC': 'anomalies'})
    scores = scores.merge(anomalies, left_index=True, right_index=True, how='left')

    plt.figure(figsize=(12, 6))
    plt.plot(scores["datetime"], scores["real AC"], label='AC Power')
    plt.scatter(scores["datetime"], scores["anomalies"], color='red', label='Anomaly', s=40)
    plt.title("Anomalies Detected by LSTM Autoencoder")
    plt.xlabel("DateTime")
    plt.ylabel("AC Power")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./plot/model_anomaly_plot.png")

# In[43]:


for batch_size_search in [10,15,20]:
    for learning_rate_search in [0.001,0.01,0.1]:
        with mlflow.start_run(log_system_metrics=True):
            epochs = 10
            batch_size = batch_size_search
            learning_rate = learning_rate_search

            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", learning_rate)

            X_tensor = torch.tensor(X_train, dtype=torch.float32)
            dataset = TensorDataset(X_tensor, X_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # model traning
            model = model_traning(epochs, learning_rate, dataloader)

            # model evaluation
            model.eval()
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            with torch.no_grad():
                X_pred = model(X_test_tensor).numpy()
            
            X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
            X_pred = scaler.inverse_transform(X_pred)
            X_pred = pd.DataFrame(X_pred, columns=train.columns)
            X_pred.index = test.index

            # visualization
            mae_visualization(X_pred)
            anomaly_visualization(X_pred)

            mlflow.log_metric("anomaly cnt", len(X_pred[X_pred['Anomaly'] == 1]))

            mlflow.log_artifact("./plot/model_mae_plot.png")
            mlflow.log_artifact("./plot/model_anomaly_plot.png")

            mlflow.pytorch.log_model(model, "model")

mlflow.end_run()

# In[ ]:



