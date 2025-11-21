import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para evitar errores con Tkinter
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings("ignore")

# --- 1. Cargar y preparar datos ---
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['fecha_hora'] = pd.to_datetime(df['fecha'] + ' ' + df['hora'], dayfirst=True)
    df = df.sort_values('fecha_hora')
    df.set_index('fecha_hora', inplace=True)
    df = df.loc['2015-01-01':'2018-12-31 23:59:59']
    df['hora_num'] = df['hora'].str.split(':').str[0].astype(int)
    return df

# --- 2. Crear lags ---
def create_lags(df, targets, n_lags=6):
    for target in targets:
        for lag in range(1, n_lags + 1):
            df[f'{target}_lag_{lag}'] = df[target].shift(lag)
    df.dropna(inplace=True)
    return df

# --- 3. Definir features y objetivos ---
def define_features_targets(df, base_features, targets, n_lags=6):
    feature_cols = base_features.copy()
    for target in targets:
        for lag in range(1, n_lags + 1):
            feature_cols.append(f'{target}_lag_{lag}')
    X = df[feature_cols]
    y = {target: df[target] for target in targets}
    return X, y

# --- 4. Optimizar hiperparámetros con validación temporal ---
def optimize_rf(X, y, n_iter=20, cv_splits=5, random_state=42):
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    param_dist = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [None, 10, 20, 30],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__max_features': ['auto', 'sqrt', 'log2']
    }
    best_models = {}
    for target_name, y_target in y.items():
        print(f"\nOptimizing Random Forest for target: {target_name}")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Escalado para robustez (aunque RF no lo requiere)
            ('rf', RandomForestRegressor(random_state=random_state, n_jobs=-1))
        ])
        search = RandomizedSearchCV(
            pipeline, param_distributions=param_dist,
            n_iter=n_iter, cv=tscv, scoring='neg_root_mean_squared_error',
            random_state=random_state, n_jobs=-1, verbose=1
        )
        search.fit(X, y_target)
        print(f"Best params for {target_name}: {search.best_params_}")
        best_models[target_name] = search.best_estimator_
    return best_models

# --- 5. Evaluar modelos con validación temporal ---
def evaluate_models(models, X, y, cv_splits=5):
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    metrics = {target: {'rmse': [], 'mae': []} for target in y.keys()}
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        for target_name, model in models.items():
            y_train, y_test = y[target_name].iloc[train_idx], y[target_name].iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            metrics[target_name]['rmse'].append(rmse)
            metrics[target_name]['mae'].append(mae)
    for target_name in metrics:
        print(f"\n{target_name} - RMSE CV: {np.mean(metrics[target_name]['rmse']):.3f} ± {np.std(metrics[target_name]['rmse']):.3f}")
        print(f"{target_name} - MAE CV: {np.mean(metrics[target_name]['mae']):.3f} ± {np.std(metrics[target_name]['mae']):.3f}")
    return metrics

# --- 6. Análisis de residuos ---
def plot_residuals(y_true, y_pred, target_name):
    residuos = y_true - y_pred
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(residuos, bins=30, color='blue', alpha=0.7)
    plt.title(f'Histograma de residuos - {target_name}')
    plt.xlabel('Error')
    plt.ylabel('Frecuencia')

    plt.subplot(1,2,2)
    plt.scatter(y_pred, residuos, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Residuos vs Predicción - {target_name}')
    plt.xlabel('Predicción')
    plt.ylabel('Residuo')

    plt.tight_layout()
    plt.savefig(f'residuals_{target_name}.png')  # Guarda la figura en archivo
    plt.close()

# --- 7. Importancia de variables ---
def plot_feature_importance(model, feature_names, target_name, top_n=15):
    importances = model.named_steps['rf'].feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10,6))
    plt.title(f'Importancia de variables - {target_name}')
    plt.barh(range(top_n), importances[indices][::-1], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
    plt.xlabel('Importancia')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{target_name}.png')  # Guarda la figura en archivo
    plt.close()

# --- 8. Entrenar modelos finales ---
def train_final_models(models, X, y):
    for target_name, model in models.items():
        model.fit(X, y[target_name])
    return models

# --- 9. Predicción iterativa para el año siguiente ---
def iterative_forecast(models, df, feature_cols, targets, n_lags=6, periods=8760):
    last_date = df.index[-1]
    future_index = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=periods, freq='H')
    df_pred = pd.DataFrame(index=future_index)
    last_row = df.iloc[-1].copy()

    pred_results = {target: [] for target in targets}

    for current_time in future_index:
        # Actualizar variables temporales
        last_row['mes_num'] = current_time.month
        last_row['dia_semana_num'] = current_time.weekday() + 1
        last_row['hora_num'] = current_time.hour

        X_pred = last_row[feature_cols].values.reshape(1, -1)

        for target in targets:
            pred = models[target].predict(X_pred)[0]
            pred_results[target].append(pred)

        # Actualizar lags para cada target
        for target in targets:
            for lag in range(n_lags, 1, -1):
                last_row[f'{target}_lag_{lag}'] = last_row[f'{target}_lag_{lag-1}']
            last_row[f'{target}_lag_1'] = pred_results[target][-1]

        # Actualizar valores actuales para la siguiente iteración
        for target in targets:
            last_row[target] = pred_results[target][-1]

    for target in targets:
        df_pred[f'{target}_pred'] = pred_results[target]

    return df_pred

# --- 10. Visualización resultados históricos + predicción ---
def plot_combined_series(df_hist, df_pred, target, title, color_hist, color_pred):
    plt.figure(figsize=(14,5))
    plt.plot(df_hist.index, df_hist[target], label='Histórico', color=color_hist)
    plt.plot(df_pred.index, df_pred[f'{target}_pred'], label='Predicción', color=color_pred, alpha=0.7)
    plt.title(title)
    plt.xlabel('Fecha y Hora')
    plt.ylabel(target)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'combined_series_{target}.png')  # Guarda la figura en archivo
    plt.close()

# --- MAIN ---

if __name__ == "__main__":
    filepath = "C:/Users/User/OneDrive/Desktop/Prediccion de series temporales/df_limpio.csv"
    df = load_and_prepare_data(filepath)

    base_features = ['mes_num', 'dia_semana_num', 'generacion_con_biomasa', 'generacion_con_carbon',
                     'generacion_con_gas_natural', 'generacion_con_carbon_duro', 'generacion_con_petroleo',
                     'energia_consumida_bateria', 'generacion_hidro_corriente', 'generacion_hidro_embalse',
                     'generacion_nuclear', 'otras', 'renovables_otras', 'generacion_solar',
                     'generacion_con_residuos', 'generacion_eolica_onshore', 'hora_num']

    targets = ['carga_total_real', 'precio_real']
    n_lags = 6

    df = create_lags(df, targets, n_lags=n_lags)
    X, y = define_features_targets(df, base_features, targets, n_lags=n_lags)

    # Optimización hiperparámetros con validación temporal
    best_models = optimize_rf(X, y, n_iter=20, cv_splits=5, random_state=42)

    # Evaluación con validación temporal
    metrics = evaluate_models(best_models, X, y, cv_splits=5)

    # Análisis residuos en último split (usar último fold)
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_test = {target: y[target].iloc[test_idx] for target in targets}

    for target in targets:
        model = best_models[target]
        model.fit(X_train, y[target].iloc[train_idx])
        y_pred = model.predict(X_test)
        plot_residuals(y_test[target], y_pred, target)
        plot_feature_importance(model, X.columns, target)

    # Entrenar modelos finales con todo el dataset
    final_models = train_final_models(best_models, X, y)

    # Guardar modelos para uso futuro
    for target in targets:
        joblib.dump(final_models[target], f"rf_model_{target}.joblib")
        print(f"Modelo guardado: rf_model_{target}.joblib")

    # Predicción iterativa para el año siguiente
    df_pred = iterative_forecast(final_models, df, X.columns, targets, n_lags=n_lags, periods=8760)

    # Visualizar resultados históricos + predicción
    for target, color_hist, color_pred in zip(targets, ['blue', 'green'], ['cyan', 'lime']):
        plot_combined_series(df, df_pred, target, f'{target} - Histórico y Predicción', color_hist, color_pred)























