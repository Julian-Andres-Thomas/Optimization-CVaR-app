import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import cvxpy as cp
import streamlit as st

st.title("Portfolio Optimizer: Minimizing CVaR Risk")

with st.expander("What is CVaR (Conditional Value at Risk)?"):
    st.markdown("""
    **Value at Risk (VaR)** estimates the **maximum loss** a portfolio might suffer over a given time period with a certain confidence level. For example:  
    - **VaR (for a 95% confidence level)**: "There’s a 95% chance we won’t lose more than X $."

    **Conditional Value at Risk (CVaR)**, or *Expected Shortfall*, goes further — it measures the **average loss** if things go worse than the VaR threshold:  
    - **CVaR**: "If we fall into the worst 5%, our average loss will be Y euros."

    The optimizer's objective is to reduce the portfolio CVaR at a given confidence level.
    """)

st.write("### Fill out your preferences:")

st.markdown("""
**Enter at least two stock tickers separated by commas (e.g.: AAPL, MSFT, TSLA)**  
- **No final dots.**  
- **Ensure tickers are in Yahoo Finance format:**  
[Click here to verify tickers on Yahoo Finance](https://finance.yahoo.com/lookup)
""")

tickers_input = st.text_input(label="", placeholder='TSLA, AMS.MC, T, CSCO')
st.markdown("**Select the date range for the analysis:**")
start_input = st.text_input('Start Date (YYYY-MM-DD format)')
end_input = st.text_input('End Date (YYYY-MM-DD format)')
confidence_level = st.number_input("Confidence Level", min_value=0.89, max_value=0.99, value=0.95)

with st.expander("What is the Confidence Level?"):
    st.markdown("""
    The confidence level tells you how sure you want to be that your losses won’t exceed a certain amount.  
    At a **95% confidence level**, CVaR tells you the average loss you'd suffer in the **worst 5% of cases**.
    """)

if tickers_input and start_input and end_input:
    try:
        tickers = [ticker.strip() for ticker in tickers_input.split(',') if ticker.strip()]
        price_data = yf.download(
            tickers, start=start_input, end=end_input,
            multi_level_index=False, auto_adjust=False
        )['Adj Close']

        # Filter valid and invalid tickers
        valid_tickers = [ticker for ticker in tickers if ticker in price_data.columns]
        invalid_tickers = [ticker for ticker in tickers if ticker not in price_data.columns]

        if not valid_tickers:
            st.error("None of the tickers are valid or have data in the selected date range.")
        else:
            if invalid_tickers:
                st.warning(f"The following tickers were ignored due to missing data: {', '.join(invalid_tickers)}")

            price_data = price_data[valid_tickers]
            daily_returns = np.log(price_data).diff().dropna()

            if daily_returns.empty:
                st.error("Not enough data to compute returns. Try a different date range.")
            else:
                daily_mean_returns = daily_returns.mean()
                alpha = 1 - confidence_level
                num_data = daily_returns.shape[0]
                num_assets = daily_returns.shape[1]

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

                with st.expander("What is Portfolio's daily CVaR?"):
                    st.markdown(f"""
                    Example for a {int(confidence_level*100)}% confidence level:  
                    Below the {int((1-confidence_level)*100)}% worst cases, the **average loss is CVaR** = {portfolio_cvar*100:.2f}% of your capital.
                    """)
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please enter tickers, start date and end date to see results.")

footer_html = """
<div style="position: fixed; bottom: 10px; right: 10px; font-size: 12px; color: grey;">
    © 2025 Julian Andres Thomas | Economics & Finance Student  
    <br>
    <a href="mailto:julianandres.thomas@gmail.com" style="color: grey; text-decoration: none;">Email</a> | 
    <a href="https://linkedin.com/in/julianandresthomas" target="_blank" style="color: grey; text-decoration: none;">LinkedIn</a>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
