import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import cvxpy as cp
import streamlit as st

st.set_page_config(page_title="CVaR Portfolio Optimizer", layout="centered")

st.title("Portfolio Optimizer: Minimizing CVaR Risk")

with st.expander("What is CVaR (Conditional Value at Risk)?"):
    st.markdown("""
    **Value at Risk (VaR)** estimates the **maximum loss** a portfolio might suffer over a given time period with a certain confidence level. 

    **Conditional Value at Risk (CVaR)** goes further — it measures the **average loss** if things go worse than the VaR threshold. 
    """)

st.write("### Fill out your preferences:")

st.markdown("""
**Enter at least two stock tickers separated by commas (e.g.: AAPL, MSFT, TSLA)**
- No final dots.
- Ensure tickers are in Yahoo Finance format.
""")

tickers_input_raw = st.text_input(label="Tickers", placeholder='TSLA, AMS.MC, T, CSCO')
tickers_input = [ticker.strip().upper() for ticker in tickers_input_raw.split(',') if ticker.strip()]

start_input = st.text_input('Start Date (YYYY-MM-DD format)')
end_input = st.text_input('End Date (YYYY-MM-DD format)')
confidence_level = st.number_input("Confidence Level", min_value=0.89, max_value=0.99, value=0.95)

with st.expander("What is the Confidence Level?"):
    st.markdown("""
    At a **95% confidence level**, CVaR tells you the average loss you’d suffer in the **worst 5% of cases**.
    """)

if tickers_input and start_input and end_input:
    with st.spinner("⏳ Optimizing portfolio, please wait..."):
        try:
            raw_data = yf.download(
                tickers_input, 
                start=start_input, 
                end=end_input, 
                auto_adjust=False, 
                progress=False
            )['Adj Close']

            # Limpieza de datos
            raw_data.dropna(axis=1, how='all', inplace=True)
            raw_data.dropna(axis=0, how='any', inplace=True)

            if raw_data.empty:
                st.error("❌ No data found. Please check your tickers and date range.")
                st.stop()

            valid_tickers = list(raw_data.columns)
            invalid_tickers = [ticker for ticker in tickers_input if ticker not in valid_tickers]

            if invalid_tickers:
                st.error(f"❌ The following tickers do not exist or have no data: {', '.join(invalid_tickers)}")
                st.stop()

            if len(valid_tickers) < 2:
                st.error("❌ You need at least 2 valid tickers to run the optimization.")
                st.stop()

            price_data = raw_data
            daily_returns = np.log(price_data).diff().dropna()
            daily_mean_returns = daily_returns.mean()

            alpha = 1 - confidence_level
            num_data = daily_returns.shape[0]
            num_assets = daily_returns.shape[1]

            # Variables para optimización
            weights = cp.Variable(num_assets)
            t = cp.Variable()
            ui = cp.Variable(num_data)

            risk = t + cp.sum(ui) / (alpha * num_data)
            returns = daily_mean_returns.to_numpy() @ weights

            constraints = [
                -daily_returns.to_numpy() @ weights - t - ui <= 0,
                ui >= 0,
                cp.sum(weights) == 1,
                weights >= 0,
            ]

            objective = cp.Minimize(risk)
            prob = cp.Problem(objective, constraints)
            prob.solve()

            portfolio_weights = np.array([np.round(x, 3) if x > 1e-4 else 0 for x in weights.value])
            portfolio_return = (portfolio_weights @ daily_mean_returns) * 252
            portfolio_cvar = prob.value

            filtered_weights = []
            filtered_labels = []

            for w, label in zip(portfolio_weights, price_data.columns):
                if w > 0:
                    filtered_weights.append(w)
                    filtered_labels.append(label)

            fig, ax = plt.subplots(figsize=(6, 6))
            colors = plt.cm.tab20.colors

            wedges, _ = ax.pie(
                filtered_weights,
                labels=None,
                startangle=180,
                colors=colors[:len(filtered_weights)],
                wedgeprops={'width': 0.3, 'edgecolor': 'black'}
            )

            ax.set_title('Optimal weights of your portfolio')

            legend_labels = [f"{label}: {weight * 100:.2f}%" for label, weight in zip(filtered_labels, filtered_weights)]
            ax.legend(
                wedges,
                legend_labels,
                loc='lower center',
                bbox_to_anchor=(0.5, -0.2),
                ncol=2,
                frameon=False
            )

            st.write("## Optimal weights and returns of your portfolio")
            col1, col2 = st.columns(2)
            col1.pyplot(fig)
            col2.metric(label="Total return of your portfolio (annual)", value=f"{portfolio_return * 100:.2f}%")
            col2.metric(label=f"Portfolio's daily CVaR", value=f"{portfolio_cvar * 100:.2f}%")

            st.info(
                """
                Short positions are not allowed. The assets and their corresponding weights shown above represent the optimal allocation. 
                If a ticker you entered is not shown, the optimizer assigned it a weight of 0%.
                """
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.write("Please enter tickers, start date and end date to see results.")

footer_html = """
<div style="position: fixed; bottom: 10px; right: 10px; font-size: 12px; color: grey;">
    © 2025 Julian Andres Thomas | Economics & Finance Student  <br>
    <a href="mailto:julianandres.thomas@gmail.com" style="color: grey; text-decoration: none;">Email</a> | 
    <a href="https://linkedin.com/in/julianandresthomas" target="_blank" style="color: grey; text-decoration: none;">LinkedIn</a>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
