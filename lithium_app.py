import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from math import sqrt
import os
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


st.set_page_config(
    page_title="Lithium Price Prediction Thesis",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 Lithium Price Prediction Thesis")
st.markdown("""
**Topic:** Lithium price prediction using ARMA and ARIMA analytical methods.
**Data Sources:**
1.  **Forecasting:** China Spot Price (Daily aggregated to Monthly)
2.  **Correlations:** World Bank Commodity Data (Monthly)
3.  **History:** USGS Statistics (Annual Context)
""")



SPOT_PRICE_FILE = "Lithium Carbonate 99.5_Min China Spot Historical Data.csv"
USGS_FILE = "ds140-lithium-2021.xlsx"
WB_FILE = "CMO-Historical-Data-Monthly.xlsx"



def parse_daily_csv(file_path):
    """Parses the Lithium Spot Price CSV (Forecast Target)."""
    try:
        df = pd.read_csv(file_path)

        df.columns = [c.strip().replace('"', '') for c in df.columns]
        

        if df['Price'].dtype == object:
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
    st.subheader("⚖️ Automated Model Comparison (ARMA vs ARIMA)")
    st.markdown("""
    Run a side-by-side benchmark of the ARMA and ARIMA models on the lithium spot price data.
    All output charts are saved as **300 DPI PNG** files in the `thesis_figures/` directory.
    - **ARMA Model** (p=2, d=0, q=2) — Actual vs Forecast
    - **ARIMA Model** (p=2, d=1, q=2) — Actual vs Forecast
    - **Error Metrics Comparison** — MAPE, RMSE, MAE
    """)

    if df_spot is not None:
        generate_figures = st.button("� Run Model Benchmark")

        if generate_figures:
            FIGURES_DIR = "thesis_figures"
            os.makedirs(FIGURES_DIR, exist_ok=True)

            train = df_spot.iloc[:-12]
            test = df_spot.iloc[-12:]
            forecast_horizon = 12

            metrics = {}

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

            # --- Error Metrics Comparison ---
            st.markdown("---")
            st.markdown("### Forecasting Error Metrics Comparison")

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

            fig10, axes10 = plt.subplots(1, 3, figsize=(15, 5))
            fig10.suptitle('Visual Comparison of Forecasting Errors', fontsize=14, fontweight='bold', y=1.02)

            colors = ['#d62728', '#1f77b4']
            bar_labels = ['ARMA (d=0)', 'ARIMA (d=1)']

            for idx, (ax, name, arma_v, arima_v) in enumerate(zip(axes10, metric_names, arma_vals, arima_vals)):
                bars = ax.bar(bar_labels, [arma_v, arima_v], color=colors, width=0.5, edgecolor='white', linewidth=1.5)
                ax.set_title(name, fontsize=13, fontweight='bold')
                ax.set_ylabel(name, fontsize=11)
                ax.grid(axis='y', alpha=0.3)
                # Add value labels on top of bars
                for bar, val in zip(bars, [arma_v, arima_v]):
                    fmt = f"{val:.2f}%" if idx == 0 else f"{val:,.0f}"
                    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + bar.get_height() * 0.02,
                            fmt, ha='center', va='bottom', fontsize=11, fontweight='bold')
                ax.set_ylim(0, max(arma_v, arima_v) * 1.2)

            fig10.tight_layout()
            fig10_path = os.path.join(FIGURES_DIR, "figure_10_error_comparison.png")
            fig10.savefig(fig10_path, dpi=300, bbox_inches='tight')
            st.pyplot(fig10)
            st.success(f"✅ Saved: `{fig10_path}`")

            # Summary table
            st.markdown("---")
            st.markdown("### Summary Table")
            summary_df = pd.DataFrame({
                'Metric': ['MAPE', 'RMSE', 'MAE'],
                'ARMA (d=0)': [f"{metrics['ARMA']['MAPE']:.2%}", f"{metrics['ARMA']['RMSE']:,.0f}", f"{metrics['ARMA']['MAE']:,.0f}"],
                'ARIMA (d=1)': [f"{metrics['ARIMA']['MAPE']:.2%}", f"{metrics['ARIMA']['RMSE']:,.0f}", f"{metrics['ARIMA']['MAE']:,.0f}"],
            })
            st.table(summary_df)

    else:
        st.warning("Cannot generate figures: Spot price data not loaded.")