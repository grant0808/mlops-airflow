def eda():
    
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    import plotly.graph.objects as go
    import pandas as pd

    from datetime import datetime
    import pytz

    kst = pytz.timezone("Asia/Seoul")
    now = datetime.now(kst)

    generation = pd.read_csv("./data/Plant_1_Generation_Data.csv")
    weather = pd.read_csv("./data/Plant_1_Weather_Sensor_Data.csv")

    generation['DATE_TIME'] = pd.to_datetime(generation['DATE_TIME'], dayfirst=True)
    weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'], dayfirst=True)

    generation_source_key = list(generation['SOURCE_KEY'].unique())
    inv = generation[generation['SOURCE_KEY']==generation_source_key[0]]
    mask = ((weather['DATE_TIME'] >= min(inv["DATE_TIME"])) & (weather['DATE_TIME'] <= max(inv["DATE_TIME"])))
    weather_filtered = weather.loc[mask]

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

    fig.write_image(f"./plot/{now}_AC_power.png")

    df = inv.merge(weather_filtered, on="DATE_TIME", how='left')
    df = df[['DATE_TIME', 'AC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]

    df_timestamp = df[["DATE_TIME"]]
    df_ = df[["AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]

    train_prp = .6
    train = df_.loc[:df_.shape[0]*train_prp]
    test = df_.loc[df_.shape[0]*train_prp:]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train)
    X_test = scaler.transform(test)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    return

if __name__ == "__main__":
    eda()