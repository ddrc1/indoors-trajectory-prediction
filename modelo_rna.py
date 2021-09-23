import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras import backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats

file = "market.txt"
env = file.split(".")[0]
dados = pd.read_csv(f"./datasets/{file}", sep=';', usecols=['x', 'y'])

dados['time'] = [i for i in range(len(dados))]

dados = dados[:50000]
size = str(int(len(dados)/1000)) + "k"

def create_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(6,)))
    model.add(layers.Dense(2000, activation='relu'))
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(2, activation='tanh'))
    
    optimizer = optimizers.Adam(learning_rate=0.0001)
    
    model.summary()
    model.compile(loss="mse", optimizer=optimizer)
    return model

def validation(df_test: pd.DataFrame, df_predito: pd.DataFrame):
    x_real = df_test['x2']
    x_previsto = df_predito['x2']
    rmse_x = mean_squared_error(x_real, x_previsto) ** (1 / 2)
    _, _, r_value, _, _ = stats.linregress(x_real, x_previsto)
    r2_x = r_value*r_value
    mae_x = mean_absolute_error(x_real, x_previsto)
    mse_x = mean_squared_error(x_real, x_previsto)

    y_real = df_test['y2']
    y_previsto = df_predito['y2']
    rmse_y = mean_squared_error(y_real, y_previsto) ** (1 / 2)
    _, _, r_value, _, _ = stats.linregress(y_real, y_previsto)
    r2_y = r_value*r_value
    mae_y = mean_absolute_error(y_real, y_previsto)
    mse_y = mean_squared_error(y_real, y_previsto)

    return (r2_x, r2_y), (rmse_x, rmse_y), (mae_x, mae_y), (mse_x, mse_y)

def normalize(row):
    if row['x'] >= 0:
        norm_x = row['x'] / max_x
    else:
        norm_x = row['x'] / -min_x

    if row['y'] >= 0:
        norm_y = row['y'] / max_y
    else:
        norm_y = row['y'] / -min_y

    return {'x': norm_x, 'y': norm_y}

def desnormalize(row):    
    if row['x'] >= 0:
        desnorm_x = row['x'] * max_x
    else:
        desnorm_x = row['x'] * -min_x

    if row['y'] >= 0:
        desnorm_y = row['y'] * max_y
    else:
        desnorm_y = row['y'] * -min_y

    return {'x': desnorm_x, 'y': desnorm_y}
########PARA RODAR EM GPU, COMENTAR ESTE CÓDIGO########
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#######################################################
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


min_x = dados.x.min()
max_x = dados.x.max()
min_y = dados.y.min()
max_y = dados.y.max()

percentage_train = 0.8
percentage_test = 0.2
model_choice = 1
times = 10
epochs=50
callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_delta=1e-4, mode='min')]
callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=0, restore_best_weights=True))

history_list = []

r2_model = []
mse_model = []
rmse_model = []
mae_model = []

def gen_points_to_predict(dados):

    def gen_points(row: dict):
        list_dict = []
        row['t3'] = np.int32(row['t3'])
        for i, t in enumerate(np.arange(1, row['t3'], 1)):
            new_row = row.copy()
            new_row['t2'] = np.int64(t)
            list_dict.append(new_row)
        return list_dict
    list_dicts = list(dados.apply(lambda row: gen_points(dict(row)), axis=1))

    df_to_predict = pd.DataFrame()
    for dict_list in list_dicts:
        df = pd.DataFrame(dict_list)
        df_to_predict = pd.concat([df_to_predict, df])

    df_to_predict.reset_index(inplace=True, drop=True)
    return df_to_predict

def splitData(df: pd.DataFrame, p_train: float, window_size = 3):
    x_start_train = []
    y_start_train = []
    time_start_train = []
    x_end_train = []
    y_end_train = []
    time_end_train = []

    x_target_train = []
    y_target_train = []
    time_target_train = []

    x_start_test = []
    y_start_test = []
    time_start_test = []
    x_end_test = []
    y_end_test = []
    time_end_test = []

    x_real_test = []
    y_real_test = []
    time_real_test = []

    feat_train_data = {}
    feat_test_data = {}
    target_train_data = {}
    real_data = {}
    
    jump = window_size - 1
    for i in range(0, len(df) - jump):
        if (i + jump) <= (len(df) * p_train) - 1:
            for k in range(1, jump):
                x_start_train.append(df.loc[i, "x"])
                y_start_train.append(df.loc[i, "y"])
                time_start_train.append(0)

                x_target_train.append(df.loc[i + k, "x"])
                y_target_train.append(df.loc[i + k, "y"])
                time_target_train.append(k)

                x_end_train.append(df.loc[i + jump, "x"])
                y_end_train.append(df.loc[i + jump, "y"])
                time_end_train.append(jump)
                
        elif i >= len(df) - (len(df) * (1-p_train)):
            for k in range(1, jump):
                x_start_test.append(df.loc[i, "x"])
                y_start_test.append(df.loc[i, "y"])
                time_start_test.append(0)

                x_real_test.append(df.loc[i + k, "x"])
                y_real_test.append(df.loc[i + k, "y"])
                time_real_test.append(k)

                x_end_test.append(df.loc[i + jump, "x"])
                y_end_test.append(df.loc[i + jump, "y"])
                time_end_test.append(jump)

    feat_train_data['x1'] = x_start_train
    feat_train_data['y1'] = y_start_train
    feat_train_data['x3'] = x_end_train
    feat_train_data['y3'] = y_end_train
    feat_train_data['t3'] = time_end_train
    feat_train_data['t2'] = time_target_train

    target_train_data['x2'] = x_target_train
    target_train_data['y2'] = y_target_train

    df_feat_train = pd.DataFrame(data=feat_train_data)
    
    df_target_train = pd.DataFrame(data=target_train_data)

    feat_test_data['x1'] = x_start_test
    feat_test_data['y1'] = y_start_test
    feat_test_data['x3'] = x_end_test
    feat_test_data['y3'] = y_end_test
    feat_test_data['t3'] = time_end_test
    feat_test_data['t2'] = time_real_test

    real_data['x2'] = x_real_test
    real_data['y2'] = y_real_test
    
    df_feat_test = pd.DataFrame(data=feat_test_data)
    
    df_real = pd.DataFrame(data=real_data)
    
    return df_feat_train, df_target_train, df_feat_test, df_real

path = f"./validacao/{env}/rna"
for w in range(3, 33):    
    r2_exec = {'x': [], 'y': []}
    mse_exec = {'x': [], 'y': []}
    rmse_exec = {'x': [], 'y': []}
    mae_exec = {'x': [], 'y': []}
    
    dados_norm = dados.apply(lambda row: normalize(row), axis=1)
    dados_norm = pd.DataFrame(dados_norm.tolist())
    
    print("Separando o dado...")
    df_feat_train_full, df_target_train_full, df_feat_val, df_target_val = splitData(dados_norm, percentage_train, window_size=w)

    size_test = int(len(df_feat_train_full) * percentage_test)

    df_feat_test = df_feat_train_full.iloc[-size_test:]
    df_target_test = df_target_train_full.iloc[-size_test:]

    df_feat_train = df_feat_train_full.iloc[:-size_test]
    df_target_train = df_target_train_full.iloc[:-size_test]
    
    df_target_test.columns = ['x', 'y']
    df_target_test = df_target_test.apply(lambda row: desnormalize(row), axis=1)
    df_target_test = pd.DataFrame(df_target_test.tolist())
    df_target_test.columns = ['x2', 'y2']
    
    df_feat_train.info()
    
    for i in range(times):
        print(w, f"- {i + 1}ª vez")

        print("modelando...")
        model = create_model()
        history = model.fit(df_feat_train, df_target_train, validation_data=(df_feat_val, df_target_val), epochs=epochs, callbacks=callbacks)
        history_list.append(history)
        
        #model.save(f'./modelos/ambiente real/rna/model_d{w-2}')
        
        predicted_points = model.predict(df_feat_test)
        
        df_predito = pd.DataFrame(predicted_points, columns = ['x','y'])
        
        df_predito = df_predito.apply(lambda row: desnormalize(row), axis=1)
        df_predito = pd.DataFrame(df_predito.tolist())
        df_predito.columns = ['x2', 'y2']

        model_indicators = validation(df_target_test, df_predito)
        
        (r2_x, r2_y), (rmse_x, rmse_y), (mae_x, mae_y), (mse_x, mse_y) = model_indicators
        print(f"R2: x={r2_x}, y={r2_y}")
        print(f"RMSE: x={rmse_x}, y={rmse_y}")
        print(f"MAE: x={mae_x}, y={mae_y}")
        print(f"MSE: x={mse_x}, y={mse_y}")
        
        r2_exec['x'].append(r2_x)
        r2_exec['y'].append(r2_y)
        
        mse_exec['x'].append(mse_x)
        mse_exec['y'].append(mse_y)
        
        rmse_exec['x'].append(rmse_x)
        rmse_exec['y'].append(rmse_y)
        
        mae_exec['x'].append(mae_x)
        mae_exec['y'].append(mae_y)
        
        pd.DataFrame(r2_exec).to_csv(f"{path}/r2_{size}_d{w-2}.csv")
        pd.DataFrame(mse_exec).to_csv(f"{path}/mse_{size}_d{w-2}.csv")
        pd.DataFrame(rmse_exec).to_csv(f"{path}/rmse_{size}_d{w-2}.csv")
        pd.DataFrame(mae_exec).to_csv(f"{path}/mae_{size}_d{w-2}.csv")

        del model
        del df_predito
        
    del df_feat_train
    del df_target_train
    del df_feat_test
    del df_target_test
    del df_feat_val
    del df_target_val 
    print("-----------------------------------------------------------------------------------------")
