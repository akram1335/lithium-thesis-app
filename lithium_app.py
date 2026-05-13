"""
Lithium thesis dashboard + reproducible ARMA/ARIMA/LSTM benchmark (single file).
Dashboard: streamlit run lithium_app.py
CLI:       python lithium_app.py --benchmark
"""
import json
import os
import random
import sys
import warnings
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

SPOT_PRICE_FILE = "Lithium Carbonate 99.5_Min China Spot Historical Data.csv"
USGS_FILE = "ds140-lithium-2021.xlsx"
WB_FILE = "CMO-Historical-Data-Monthly.xlsx"

try:
    import torch
    import torch.nn as nn

    LOOKBACK = 12
    HIDDEN = 48
    EPOCHS = 250
    LR = 1e-3

    class LSTMRegressor(nn.Module):
        def __init__(self, hidden: int = HIDDEN):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, batch_first=True, num_layers=1)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    def _make_sequences(series: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for i in range(lookback, len(series)):
            xs.append(series[i - lookback : i])
            ys.append(series[i])
        return np.stack(xs), np.array(ys)

    def _lstm_iterative_forecast(model, scaler, train_scaled, test_len, lookback, device):
        model.eval()
        window = train_scaled[-lookback:].copy()
        preds_scaled = []
        with torch.no_grad():
            for _ in range(test_len):
                x = torch.tensor(window[-lookback:], dtype=torch.float32, device=device).view(1, lookback, 1)
                y = model(x).cpu().numpy().ravel()[0]
                preds_scaled.append(y)
                window = np.append(window, y)
        preds_scaled = np.array(preds_scaled).reshape(-1, 1)
        return scaler.inverse_transform(preds_scaled).ravel()

    def run_lstm_forecast(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42) -> np.ndarray:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device("cpu")
        scaler = MinMaxScaler()
        train_s = scaler.fit_transform(train[["Price"]]).ravel()
        X, y = _make_sequences(train_s, LOOKBACK)
        X_t = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(-1)
        y_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(-1)
        model = LSTMRegressor(HIDDEN).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(EPOCHS):
            opt.zero_grad()
            pred = model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()
        return _lstm_iterative_forecast(model, scaler, train_s, len(test), LOOKBACK, device)

    _LSTM_AVAILABLE = True
except ImportError:
    LOOKBACK = HIDDEN = EPOCHS = None
    run_lstm_forecast = None  # type: ignore
    _LSTM_AVAILABLE = False


def _invoked_by_streamlit() -> bool:
    """Detect Streamlit execution.

    ``streamlit run app.py`` rewrites ``sys.argv`` to ``[app.py, ...]`` (no ``streamlit``
    token). Hosted health checks also rely on the runtime, not argv. Use ScriptRunContext.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None:
            return True
    except Exception:
        pass
    return any("streamlit" in (a or "").lower() for a in sys.argv)


def load_monthly_spot_cli(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = [c.strip().replace('"', "") for c in df.columns]
    if not pd.api.types.is_numeric_dtype(df["Price"]):
        df["Price"] = (
            df["Price"].astype(str).str.replace(",", "").str.replace('"', "").astype(float)
        )
    df["Date"] = pd.to_datetime(df["Date"])
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"])
    df = df.sort_values("Date").set_index("Date")
    monthly = df["Price"].resample("MS").mean().to_frame()
    monthly.dropna(inplace=True)
    return monthly


def run_benchmark_cli() -> None:
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    root = Path(__file__).resolve().parent
    csv_path = root / SPOT_PRICE_FILE
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}")
    df = load_monthly_spot_cli(str(csv_path))
    raw_n = len(pd.read_csv(csv_path))
    if len(df) < 24:
        raise SystemExit(f"Need at least 24 monthly points, got {len(df)}")
    train = df.iloc[:-12]
    test = df.iloc[-12:]
    y_test = np.asarray(test["Price"], dtype=np.float64)
    endog_train = np.asarray(train["Price"].squeeze(), dtype=np.float64)
    fit_arma = ARIMA(endog_train, order=(2, 0, 2)).fit()
    fit_arima = ARIMA(endog_train, order=(2, 1, 2)).fit()
    steps = len(test)
    pred_arma = np.asarray(fit_arma.get_forecast(steps=steps).predicted_mean, dtype=np.float64)
    pred_arima = np.asarray(fit_arima.get_forecast(steps=steps).predicted_mean, dtype=np.float64)

    def metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return {
            "model": name,
            "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
        }

    rows = [
        metrics("ARMA (2,0,2)", y_test, pred_arma),
        metrics("ARIMA (2,1,2)", y_test, pred_arima),
    ]
    if _LSTM_AVAILABLE and run_lstm_forecast is not None:
        pred_lstm = run_lstm_forecast(train, test, seed=SEED)
        rows.append(metrics("LSTM (univariate, iterative 12-step)", y_test, pred_lstm))
    else:
        print("Note: PyTorch not available — LSTM skipped.", file=sys.stderr)

    out = {
        "csv": str(csv_path),
        "daily_rows": int(raw_n),
        "monthly_rows": int(len(df)),
        "monthly_date_start": str(df.index.min().date()),
        "monthly_date_end": str(df.index.max().date()),
        "train_months": int(len(train)),
        "test_months": int(len(test)),
        "train_period": f"{train.index[0].strftime('%Y-%m')} -- {train.index[-1].strftime('%Y-%m')}",
        "test_period": f"{test.index[0].strftime('%Y-%m')} -- {test.index[-1].strftime('%Y-%m')}",
        "lookback": LOOKBACK,
        "hidden": HIDDEN,
        "epochs": EPOCHS,
        "metrics": rows,
    }
    print(json.dumps(out, indent=2))
    out_path = root / "experiment_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__" and "--benchmark" in sys.argv:
    run_benchmark_cli()
    raise SystemExit(0)
if __name__ == "__main__" and not _invoked_by_streamlit():
    print(
        "Use:  streamlit run lithium_app.py\n   or:  python lithium_app.py --benchmark",
        file=sys.stderr,
    )
    raise SystemExit(2)


st.set_page_config(
    page_title="Lithium Price Prediction Thesis",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 Lithium Price Prediction Thesis")
st.markdown("""
**Topic:** Lithium price prediction using ARMA, ARIMA, and an LSTM baseline (PyTorch).
**Data Sources:**
1.  **Forecasting:** China Spot Price (Daily aggregated to Monthly)
2.  **Correlations:** World Bank Commodity Data (Monthly)
3.  **History:** USGS Statistics (Annual Context)
""")



def parse_daily_csv(file_path):
    """Parses the Lithium Spot Price CSV (Forecast Target)."""
    try:
        df = pd.read_csv(file_path)

        df.columns = [c.strip().replace('"', '') for c in df.columns]

        if not pd.api.types.is_numeric_dtype(df['Price']):
            df['Price'] = df['Price'].astype(str).str.replace(',', '').str.replace('"', '').astype(float)

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')

        df_monthly = df['Price'].resample('MS').mean().to_frame()
        df_monthly.dropna(inplace=True)
        return df_monthly
    except Exception as e:
        st.error(f"Error reading Spot CSV: {e}")
        return None

def parse_usgs(file_path):
    """Parses USGS Annual Excel."""
    try:

        df_raw = pd.read_excel(file_path)
        header_row = 0
        for i, row in df_raw.iterrows():
            row_str = str(row.values).lower()
            if 'year' in row_str and 'unit value' in row_str:
                header_row = i + 1
                break
        
        df = pd.read_excel(file_path, header=header_row)
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        year_col = next(c for c in df.columns if 'year' in c)
        price_col = next(c for c in df.columns if 'unit value' in c and '98' not in c)
        
        df = df[[year_col, price_col]].copy()
        df.columns = ['Year', 'Price']
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error reading USGS Excel: {e}")
        return None

def parse_world_bank(file_path):
    """Parses World Bank Pink Sheet (Monthly Commodities)."""
    try:

        header_row = 4
        df = pd.read_excel(file_path, sheet_name='Monthly Prices', header=header_row)
        

        df.rename(columns={df.columns[0]: 'Date_Raw'}, inplace=True)
        

        df = df.iloc[2:]
        

        cols_to_numeric = df.columns.drop('Date_Raw')
        df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric, errors='coerce')
        

        def parse_wb_date(val):
            if isinstance(val, str) and 'M' in val:
                parts = val.split('M')
                return f"{parts[0]}-{parts[1]}-01"
            return np.nan

        df['Date'] = df['Date_Raw'].apply(parse_wb_date)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        return df[numeric_cols]
        
    except Exception as e:
        st.error(f"Error reading World Bank Excel: {e}")
        return None




df_spot = None
df_usgs = None
df_wb = None


if os.path.exists(SPOT_PRICE_FILE):
    df_spot = parse_daily_csv(SPOT_PRICE_FILE)
if os.path.exists(USGS_FILE):
    df_usgs = parse_usgs(USGS_FILE)
if os.path.exists(WB_FILE):
    df_wb = parse_world_bank(WB_FILE)


tab1, tab2, tab3, tab4 = st.tabs(["📈 ARIMA Forecasting", "📊 Market Correlations", "📚 Historical Context", "⚖️ Model Benchmarking"])


with tab1:
    if df_spot is not None:
        col_graph, col_params = st.columns([3, 1])
        
        with col_params:
            st.subheader("Model Parameters")
            p = st.number_input("p (AR)", 0, 10, 2)
            d = st.number_input("d (I)", 0, 2, 1)
            q = st.number_input("q (MA)", 0, 10, 2)
            forecast_horizon = st.slider("Horizon (Months)", 6, 24, 12)
            run_forecast = st.button("Run Forecast")

        with col_graph:
            st.subheader("Lithium Spot Price (Monthly Aggregated)")
            st.line_chart(df_spot)

        if run_forecast:
            with st.spinner("Calculating..."):
                try:
                    if len(df_spot) < 15:
                        st.error(f"Insufficient data for ARIMA. Need at least 15 months, got {len(df_spot)}.")
                        st.stop()


                    train = df_spot.iloc[:-12]
                    test = df_spot.iloc[-12:]
                    

                    model = ARIMA(train['Price'], order=(p, d, q))
                    model_fit = model.fit()
                    

                    steps = len(test) + forecast_horizon
                    forecast_res = model_fit.get_forecast(steps=steps)
                    pred_mean = forecast_res.predicted_mean
                    conf_int = forecast_res.conf_int()
                    

                    pred_test = pred_mean[:len(test)]
                    rmse = sqrt(mean_squared_error(test['Price'], pred_test))
                    mape = mean_absolute_percentage_error(test['Price'], pred_test)
                    mae = mean_absolute_error(test['Price'], pred_test)
                    

                    st.success("Analysis Complete")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("RMSE", f"{rmse:,.0f}")
                    c2.metric("MAPE", f"{mape:.2%}")
                    c3.metric("MAE", f"{mae:,.0f}")
                    c4.metric("AIC", f"{model_fit.aic:.0f}")
                    

                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(train.index, train['Price'], label='History', color='gray')
                    ax.plot(test.index, test['Price'], label='Actual', color='green')
                    ax.plot(pred_mean.index, pred_mean, label='Forecast', color='red', linestyle='--')
                    ax.fill_between(pred_mean.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='red', alpha=0.1)
                    ax.set_title("ARIMA Forecast Results")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Model Error: {e}")
    else:
        if not os.path.exists(SPOT_PRICE_FILE):
            st.warning(f"Missing File: {SPOT_PRICE_FILE}")
        else:
            st.warning(f"Could not load {SPOT_PRICE_FILE}. See error above.")


with tab2:
    if df_wb is not None and df_spot is not None:
        st.subheader("Macroeconomic Correlation Analysis")
        st.info("Compare Lithium against other commodities (from World Bank Pink Sheet) to find market drivers.")
        

        cols_available = df_wb.columns.tolist()

        default_idx = 0
        for i, c in enumerate(cols_available):
            if "copper" in str(c).lower(): default_idx = i; break
            
        target_commodity = st.selectbox("Compare with:", cols_available, index=default_idx)
        

        df_combined = df_spot.join(df_wb[[target_commodity]], how='inner')
        
        if not df_combined.empty:
            df_combined.columns = ['Lithium (CNY)', f'{target_commodity} (USD)']
            

            fig2, ax1 = plt.subplots(figsize=(10, 5))
            
            color = 'tab:red'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Lithium Price (CNY)', color=color)
            ax1.plot(df_combined.index, df_combined['Lithium (CNY)'], color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel(f'{target_commodity} (USD)', color=color)
            ax2.plot(df_combined.index, df_combined.iloc[:, 1], color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title(f"Correlation: Lithium vs {target_commodity}")
            fig2.tight_layout()
            st.pyplot(fig2)
            

            corr = df_combined.corr().iloc[0, 1]
            st.metric(f"Correlation Coefficient", f"{corr:.4f}")
        else:
            st.warning("No overlapping dates found between Lithium data and World Bank data.")
            
    else:
        if df_wb is None:
            if not os.path.exists(WB_FILE):
                st.warning(f"Missing File: {WB_FILE}")
            else:
                st.warning(f"Could not load {WB_FILE}. See error above.")
        
        if df_spot is None:
            if not os.path.exists(SPOT_PRICE_FILE):
                st.warning(f"Missing File: {SPOT_PRICE_FILE}")
            else:
                st.warning(f"Could not load {SPOT_PRICE_FILE}. See error above.")


with tab3:
    if df_usgs is not None:
        st.subheader("Long-Term History (1900-2021)")
        fig3 = px.bar(df_usgs, x='Year', y='Price', title="USGS Historical Unit Value ($/t)")
        st.plotly_chart(fig3, width="stretch")
    else:
        if not os.path.exists(USGS_FILE):
            st.warning(f"Missing File: {USGS_FILE}")
        else:
            st.warning(f"Could not load {USGS_FILE}. See error above.")


# --- Tab 4: Model Benchmarking ---
with tab4:
    st.subheader("⚖️ Automated model comparison (ARMA / ARIMA / LSTM)")
    st.markdown("""
    Same benchmark as the thesis: **ARMA (2,0,2)**, **ARIMA (2,1,2)**, and a **univariate LSTM** (12-month lookback,
    iterative 12-step test forecast). Figures save as **300 DPI PNG** under `thesis_figures/`; metrics also go to **`experiment_results.json`**.
    """)
    if not _LSTM_AVAILABLE:
        st.info(
            "PyTorch is not installed — only **ARMA** and **ARIMA** run. "
            "Install: `pip install torch`, then restart the app."
        )

    if df_spot is not None:
        generate_figures = st.button("Run full benchmark (ARMA / ARIMA / LSTM)")

        if generate_figures:
            FIGURES_DIR = "thesis_figures"
            os.makedirs(FIGURES_DIR, exist_ok=True)

            train = df_spot.iloc[:-12]
            test = df_spot.iloc[-12:]
            forecast_horizon = 12

            metrics = {}
            pred_lstm = None

            with st.spinner("Running ARMA model (d=0)..."):
                try:
                    model_arma = ARIMA(train['Price'], order=(2, 0, 2))
                    fit_arma = model_arma.fit()
                    steps = len(test) + forecast_horizon
                    fc_arma = fit_arma.get_forecast(steps=steps)
                    pred_arma = fc_arma.predicted_mean
                    ci_arma = fc_arma.conf_int()
                    pred_test_arma = pred_arma[:len(test)]

                    metrics['ARMA'] = {
                        'MAPE': mean_absolute_percentage_error(test['Price'], pred_test_arma),
                        'RMSE': sqrt(mean_squared_error(test['Price'], pred_test_arma)),
                        'MAE': mean_absolute_error(test['Price'], pred_test_arma),
                    }
                except Exception as e:
                    st.error(f"ARMA model error: {e}")
                    st.stop()

            with st.spinner("Running ARIMA model (d=1)..."):
                try:
                    model_arima = ARIMA(train['Price'], order=(2, 1, 2))
                    fit_arima = model_arima.fit()
                    steps = len(test) + forecast_horizon
                    fc_arima = fit_arima.get_forecast(steps=steps)
                    pred_arima = fc_arima.predicted_mean
                    ci_arima = fc_arima.conf_int()
                    pred_test_arima = pred_arima[:len(test)]

                    metrics['ARIMA'] = {
                        'MAPE': mean_absolute_percentage_error(test['Price'], pred_test_arima),
                        'RMSE': sqrt(mean_squared_error(test['Price'], pred_test_arima)),
                        'MAE': mean_absolute_error(test['Price'], pred_test_arima),
                    }
                except Exception as e:
                    st.error(f"ARIMA model error: {e}")
                    st.stop()

            if _LSTM_AVAILABLE:
                with st.spinner("Training LSTM (PyTorch, ~250 epochs)..."):
                    try:
                        pred_lstm = run_lstm_forecast(train, test, seed=42)
                        pred_lstm = np.asarray(pred_lstm, dtype=np.float64).ravel()
                        metrics['LSTM'] = {
                            'MAPE': mean_absolute_percentage_error(test['Price'], pred_lstm),
                            'RMSE': sqrt(mean_squared_error(test['Price'], pred_lstm)),
                            'MAE': mean_absolute_error(test['Price'], pred_lstm),
                        }
                    except Exception as e:
                        st.error(f"LSTM error: {e}")
                        pred_lstm = None

            # --- ARMA Forecast ---
            st.markdown("---")
            st.markdown("### ARMA Model — Actual vs Forecast")
            fig8, ax8 = plt.subplots(figsize=(12, 6))
            ax8.plot(train.index, train['Price'], label='Training Data', color='#555555', linewidth=1)
            ax8.plot(test.index, test['Price'], label='Actual Price', color='#2ca02c', linewidth=2)
            ax8.plot(pred_arma.index, pred_arma, label='ARMA Forecast', color='#d62728', linewidth=2, linestyle='--')
            ax8.fill_between(pred_arma.index, ci_arma.iloc[:, 0], ci_arma.iloc[:, 1], color='#d62728', alpha=0.1, label='95% Confidence Interval')
            ax8.axvline(x=test.index[0], color='black', linestyle=':', alpha=0.5, label='Forecast Start')
            ax8.set_title('Comparison of Actual Lithium Prices vs ARMA Model Forecast', fontsize=14, fontweight='bold')
            ax8.set_xlabel('Date', fontsize=12)
            ax8.set_ylabel('Price (CNY/tonne)', fontsize=12)
            ax8.legend(loc='upper left', fontsize=10)
            ax8.grid(True, alpha=0.3)
            fig8.tight_layout()
            fig8_path = os.path.join(FIGURES_DIR, "figure_8_arma_forecast.png")
            fig8.savefig(fig8_path, dpi=300, bbox_inches='tight')
            st.pyplot(fig8)
            st.success(f"✅ Saved: `{fig8_path}`")

            # --- ARIMA Forecast ---
            st.markdown("---")
            st.markdown("### ARIMA Model — Actual vs Forecast")
            fig9, ax9 = plt.subplots(figsize=(12, 6))
            ax9.plot(train.index, train['Price'], label='Training Data', color='#555555', linewidth=1)
            ax9.plot(test.index, test['Price'], label='Actual Price', color='#2ca02c', linewidth=2)
            ax9.plot(pred_arima.index, pred_arima, label='ARIMA Forecast', color='#1f77b4', linewidth=2, linestyle='--')
            ax9.fill_between(pred_arima.index, ci_arima.iloc[:, 0], ci_arima.iloc[:, 1], color='#1f77b4', alpha=0.1, label='95% Confidence Interval')
            ax9.axvline(x=test.index[0], color='black', linestyle=':', alpha=0.5, label='Forecast Start')
            ax9.set_title('Comparison of Actual Lithium Prices vs ARIMA Model Forecast', fontsize=14, fontweight='bold')
            ax9.set_xlabel('Date', fontsize=12)
            ax9.set_ylabel('Price (CNY/tonne)', fontsize=12)
            ax9.legend(loc='upper left', fontsize=10)
            ax9.grid(True, alpha=0.3)
            fig9.tight_layout()
            fig9_path = os.path.join(FIGURES_DIR, "figure_9_arima_forecast.png")
            fig9.savefig(fig9_path, dpi=300, bbox_inches='tight')
            st.pyplot(fig9)
            st.success(f"✅ Saved: `{fig9_path}`")

            if pred_lstm is not None:
                st.markdown("---")
                st.markdown("### LSTM — Actual vs Forecast (test window)")
                fig11, ax11 = plt.subplots(figsize=(12, 6))
                ax11.plot(train.index, train['Price'], label='Training Data', color='#555555', linewidth=1)
                ax11.plot(test.index, test['Price'], label='Actual Price', color='#2ca02c', linewidth=2)
                ax11.plot(test.index, pred_lstm, label='LSTM forecast', color='#9467bd', linewidth=2, linestyle='--')
                ax11.axvline(x=test.index[0], color='black', linestyle=':', alpha=0.5, label='Forecast start')
                ax11.set_title('Lithium prices vs univariate LSTM (iterative 12-step)', fontsize=14, fontweight='bold')
                ax11.set_xlabel('Date', fontsize=12)
                ax11.set_ylabel('Price (CNY/tonne)', fontsize=12)
                ax11.legend(loc='upper left', fontsize=10)
                ax11.grid(True, alpha=0.3)
                fig11.tight_layout()
                fig11_path = os.path.join(FIGURES_DIR, "figure_11_lstm_forecast.png")
                fig11.savefig(fig11_path, dpi=300, bbox_inches='tight')
                st.pyplot(fig11)
                st.success(f"✅ Saved: `{fig11_path}`")

            # --- Error Metrics Comparison ---
            st.markdown("---")
            st.markdown("### Forecasting error metrics comparison")

            metric_names = ['MAPE (%)', 'RMSE', 'MAE']
            arma_vals = [
                metrics['ARMA']['MAPE'] * 100,
                metrics['ARMA']['RMSE'],
                metrics['ARMA']['MAE'],
            ]
            arima_vals = [
                metrics['ARIMA']['MAPE'] * 100,
                metrics['ARIMA']['RMSE'],
                metrics['ARIMA']['MAE'],
            ]
            if pred_lstm is not None:
                lstm_vals = [
                    metrics['LSTM']['MAPE'] * 100,
                    metrics['LSTM']['RMSE'],
                    metrics['LSTM']['MAE'],
                ]
                bar_labels = ['ARMA (d=0)', 'ARIMA (d=1)', 'LSTM']
                colors = ['#d62728', '#1f77b4', '#9467bd']
                fig_w = 18
            else:
                lstm_vals = None
                bar_labels = ['ARMA (d=0)', 'ARIMA (d=1)']
                colors = ['#d62728', '#1f77b4']
                fig_w = 15

            fig10, axes10 = plt.subplots(1, 3, figsize=(fig_w, 5))
            fig10.suptitle('Visual comparison of forecasting errors', fontsize=14, fontweight='bold', y=1.02)

            for idx, ax in enumerate(axes10):
                name = metric_names[idx]
                if pred_lstm is not None:
                    vals = [arma_vals[idx], arima_vals[idx], lstm_vals[idx]]
                    ymax = max(vals) * 1.15
                else:
                    vals = [arma_vals[idx], arima_vals[idx]]
                    ymax = max(vals) * 1.2
                bars = ax.bar(bar_labels, vals, color=colors, width=0.55, edgecolor='white', linewidth=1.5)
                ax.set_title(name, fontsize=13, fontweight='bold')
                ax.set_ylabel(name, fontsize=11)
                ax.grid(axis='y', alpha=0.3)
                for bar, val in zip(bars, vals):
                    fmt = f"{val:.2f}%" if idx == 0 else f"{val:,.0f}"
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., h + max(ymax, 1e-9) * 0.02,
                            fmt, ha='center', va='bottom', fontsize=10, fontweight='bold')
                ax.set_ylim(0, ymax)

            fig10.tight_layout()
            fig10_path = os.path.join(FIGURES_DIR, "figure_10_error_comparison.png")
            fig10.savefig(fig10_path, dpi=300, bbox_inches='tight')
            st.pyplot(fig10)
            st.success(f"✅ Saved: `{fig10_path}`")

            # Summary table
            st.markdown("---")
            st.markdown("### Summary table")
            if pred_lstm is not None:
                summary_df = pd.DataFrame({
                    'Metric': ['MAPE', 'RMSE', 'MAE'],
                    'ARMA (d=0)': [f"{metrics['ARMA']['MAPE']:.2%}", f"{metrics['ARMA']['RMSE']:,.0f}", f"{metrics['ARMA']['MAE']:,.0f}"],
                    'ARIMA (d=1)': [f"{metrics['ARIMA']['MAPE']:.2%}", f"{metrics['ARIMA']['RMSE']:,.0f}", f"{metrics['ARIMA']['MAE']:,.0f}"],
                    'LSTM': [f"{metrics['LSTM']['MAPE']:.2%}", f"{metrics['LSTM']['RMSE']:,.0f}", f"{metrics['LSTM']['MAE']:,.0f}"],
                })
            else:
                summary_df = pd.DataFrame({
                    'Metric': ['MAPE', 'RMSE', 'MAE'],
                    'ARMA (d=0)': [f"{metrics['ARMA']['MAPE']:.2%}", f"{metrics['ARMA']['RMSE']:,.0f}", f"{metrics['ARMA']['MAE']:,.0f}"],
                    'ARIMA (d=1)': [f"{metrics['ARIMA']['MAPE']:.2%}", f"{metrics['ARIMA']['RMSE']:,.0f}", f"{metrics['ARIMA']['MAE']:,.0f}"],
                })
            st.table(summary_df)

            rows_json = [
                {"model": "ARMA (2,0,2)", "MAPE": float(metrics['ARMA']['MAPE']), "RMSE": float(metrics['ARMA']['RMSE']), "MAE": float(metrics['ARMA']['MAE'])},
                {"model": "ARIMA (2,1,2)", "MAPE": float(metrics['ARIMA']['MAPE']), "RMSE": float(metrics['ARIMA']['RMSE']), "MAE": float(metrics['ARIMA']['MAE'])},
            ]
            if pred_lstm is not None:
                rows_json.append({
                    "model": "LSTM (univariate, iterative 12-step)",
                    "MAPE": float(metrics['LSTM']['MAPE']),
                    "RMSE": float(metrics['LSTM']['RMSE']),
                    "MAE": float(metrics['LSTM']['MAE']),
                })
            exp = {
                "csv": os.path.abspath(SPOT_PRICE_FILE),
                "daily_rows": int(len(pd.read_csv(SPOT_PRICE_FILE))),
                "monthly_rows": int(len(df_spot)),
                "monthly_date_start": str(df_spot.index.min().date()),
                "monthly_date_end": str(df_spot.index.max().date()),
                "train_months": int(len(train)),
                "test_months": int(len(test)),
                "train_period": f"{train.index[0].strftime('%Y-%m')} -- {train.index[-1].strftime('%Y-%m')}",
                "test_period": f"{test.index[0].strftime('%Y-%m')} -- {test.index[-1].strftime('%Y-%m')}",
                "lookback": LOOKBACK,
                "hidden": HIDDEN,
                "epochs": EPOCHS,
                "metrics": rows_json,
            }
            json_path = os.path.abspath("experiment_results.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(exp, f, indent=2)
            st.success(f"✅ Wrote `{json_path}`")
    else:
        st.warning("Cannot generate figures: Spot price data not loaded.")