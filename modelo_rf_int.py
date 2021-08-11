import gc
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import *
import numpy as np
import pandas as pd
from scipy import stats

def getModel(model_choice: int = 0):
    if model_choice == 0:
        return LinearRegression()
    if model_choice == 1:
        return RandomForestRegressor(n_jobs=12)
    if model_choice == 2:
        return GradientBoostingRegressor()
    else:
        return SVR()


def interpol(df, col_x1, col_y1, col_x3, col_y3, col_t3, col_target):

    def get_proporcional_interpolation(row):

        time_diff = np.int64(row[col_t3])

        x_diff = row[col_x3] - row[col_x1]
        y_diff = row[col_y3] - row[col_y1]

        perc_total = (row[col_target]) / time_diff

        x = row[col_x1] + x_diff * perc_total
        y = row[col_y1] + y_diff * perc_total
        return {"x2": x, "y2": y}

    list_dicts = list(df.apply(lambda row: get_proporcional_interpolation(dict(row)), axis=1))
    df_interpoled = pd.DataFrame(list_dicts)

    df_interpoled.columns = ["x2", "y2"]
    return df_interpoled


def validation(df_test: pd.DataFrame, df_predito: pd.DataFrame):
    x_real = df_test['x2']
    x_previsto = df_predito['x2']
    rmse_x = mean_squared_error(x_real, x_previsto) ** 1 / 2
    _, _, r_value, _, _ = stats.linregress(x_real, x_previsto)
    r2_x = r_value*r_value
    mae_x = mean_absolute_error(x_real, x_previsto)
    mse_x = mean_squared_error(x_real, x_previsto)

    y_real = df_test['y2']
    y_previsto = df_predito['y2']
    rmse_y = mean_squared_error(y_real, y_previsto) ** 1 / 2
    _, _, r_value, _, _ = stats.linregress(y_real, y_previsto)
    r2_y = r_value*r_value
    mae_y = mean_absolute_error(y_real, y_previsto)
    mse_y = mean_squared_error(y_real, y_previsto)

    return (r2_x, r2_y), (rmse_x, rmse_y), (mae_x, mae_y), (mse_x, mse_y)


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
            for j in reversed(range(2, window_size)):
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
    df_feat_train = convert_train_to_32bits(df_feat_train)
    
    df_target_train = pd.DataFrame(data=target_train_data)
    df_target_train = convert_test_to_32bits(df_target_train)

    feat_test_data['x1'] = x_start_test
    feat_test_data['y1'] = y_start_test
    feat_test_data['x3'] = x_end_test
    feat_test_data['y3'] = y_end_test
    feat_test_data['t3'] = time_end_test
    feat_test_data['t2'] = time_real_test

    real_data['x2'] = x_real_test
    real_data['y2'] = y_real_test

    df_feat_test = pd.DataFrame(data=feat_test_data)
    df_feat_test = convert_train_to_32bits(df_feat_test)
    
    df_real = pd.DataFrame(data=real_data)
    df_real = convert_test_to_32bits(df_real)

    return df_feat_train, df_target_train, df_feat_test, df_real


def convert_test_to_32bits(df):
    df['x2'] = df['x2'].astype(np.float32)
    df['y2'] = df['y2'].astype(np.float32)
    return df


def convert_train_to_32bits(df):
    df['x1'] = df['x1'].astype(np.float32)
    df['y1'] = df['y1'].astype(np.float32)
    df['x3'] = df['x3'].astype(np.float32)
    df['y3'] = df['y3'].astype(np.float32)
    df['t3'] = df['t3'].astype(np.int32)
    df['t2'] = df['t2'].astype(np.int32)
    return df


file = "real.csv"
env = file.split(".")[0]
dados = pd.read_csv(f"./backend/datasets/{file}", sep=',', usecols=['x', 'y'])

dados['time'] = [i for i in range(len(dados))]

dados['x'] = dados['x'].astype(np.float32)
dados['y'] = dados['y'].astype(np.float32)
dados['time'] = dados['time'].astype(np.int32)

dados = dados[:50000]
size = str(int(len(dados)/1000)) + "k"

percentage_train = 0.8
window_size=17
model_choice = 1
times = 5

r2_interp = []
mse_interp = []
rmse_interp = []
mae_interp = []

r2_model = []
mse_model = []
rmse_model = []
mae_model = []
path_rf = f"./validacao/{env}/random_forest"
path_il = f"./validacao/{env}/interpolacao"
for w in range(3, 33):
    
    r2_exec = {'x': [], 'y': []}
    mse_exec = {'x': [], 'y': []}
    rmse_exec = {'x': [], 'y': []}
    mae_exec = {'x': [], 'y': []}
            
    df_feat_train, df_target_train, df_feat_test, df_real = splitData(dados, percentage_train, window_size=w)
    df_feat_train.info()
    
    for i in range(times):
        print(w, f"- {i + 1}ª vez")

        print("modelando...")
        model = MultiOutputRegressor(getModel(model_choice)).fit(df_feat_train, df_target_train)
        #pickle.dump(model, open(f'./modelos/{env}/rf/model_d{w-2}.sav', 'wb'))

        df_interpolacao = interpol(df_feat_test, "x1", "y1", "x3", "y3", "t3", "t2")
        predicted_points = model.predict(df_feat_test)
        df_predito = pd.DataFrame(predicted_points, columns = ['x2','y2'])

        model_indicators = validation(df_real, df_predito)
        
        (r2_x, r2_y), (rmse_x, rmse_y), (mae_x, mae_y), (mse_x, mse_y) = model_indicators
        
        r2_exec['x'].append(r2_x)
        r2_exec['y'].append(r2_y)
        
        mse_exec['x'].append(mse_x)
        mse_exec['y'].append(mse_y)
        
        rmse_exec['x'].append(rmse_x)
        rmse_exec['y'].append(rmse_y)
        
        mae_exec['x'].append(mae_x)
        mae_exec['y'].append(mae_y)

        base_line_indicators = validation(df_real, df_interpolacao)
        (r2_x, r2_y), (rmse_x, rmse_y), (mae_x, mae_y), (mse_x, mse_y) = base_line_indicators

        del model
        del df_predito
        gc.collect()
    del df_feat_train
    del df_target_train
    del df_feat_test
    del df_real
    
    r2_interp = [{'x': r2_x, 'y': r2_y}]
    mse_interp = [{'x': mse_x, 'y': mse_x}]
    rmse_interp = [{'x': rmse_x, 'y': rmse_y}]
    mae_interp = [{'x': mae_x, 'y': mae_y}]
    
    
    pd.DataFrame(r2_exec).to_csv(f"{path_rf}/r2_{size}_d{w-2}.csv")
    pd.DataFrame(mse_exec).to_csv(f"{path_rf}/mse_{size}_d{w-2}.csv")
    pd.DataFrame(rmse_exec).to_csv(f"{path_rf}/rmse_{size}_d{w-2}.csv")
    pd.DataFrame(mae_exec).to_csv(f"{path_rf}/mae_{size}_d{w-2}.csv")
    
    
    pd.DataFrame(r2_interp).to_csv(f"{path_il}/r2_{size}_d{w-2}.csv")
    pd.DataFrame(mse_interp).to_csv(f"{path_il}/mse_{size}_d{w-2}.csv")
    pd.DataFrame(rmse_interp).to_csv(f"{path_il}/rmse_{size}_d{w-2}.csv")
    pd.DataFrame(mae_interp).to_csv(f"{path_il}/mae_{size}_d{w-2}.csv")
    print("-----------------------------------------------------------------------------------------")