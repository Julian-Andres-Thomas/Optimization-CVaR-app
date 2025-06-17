
# 📊 Portfolio Optimizer (through CVaR Minimization)

🌐 **Try the app here:** [https://optimizercvarjulianandresthomas.streamlit.app](https://optimizercvarjulianandresthomas.streamlit.app)

This Streamlit app allows users to construct an **optimal portfolio** based on the **Conditional Value at Risk (CVaR)** approach.  
It uses **historical data from Yahoo Finance**, applies **convex optimization** (via `cvxpy`), and outputs **optimal weights**, **CVaR**, and **expected annual return**.

---

## 🚀 Features
* ⚠️ Minimizes **Conditional Value at Risk (CVaR)** at a chosen confidence level.
* 📊 Visualizes optimal asset allocation with a clean donut chart.
* 📆 Customizable date range and list of tickers.
* 🧠 Built with a user-friendly Streamlit interface.

---

## 🔧 Tools

* **Python**  
* **Streamlit**  
* **NumPy**, **Pandas**, **Matplotlib**  
* **cvxpy** (convex optimization)  
* **yfinance** (data acquisition)  

---

## 📘 What is CVaR?

**CVaR** (Conditional Value at Risk), also known as **Expected Shortfall**, is a risk metric that estimates the **average loss** in the **worst-case scenarios** (e.g., the worst 5% of cases for a 95% confidence level).

This optimizer minimizes that expected shortfall to build a safer portfolio under stress.

---

## 🧠 Author

**Julian Andres Thomas**  
🎓 BSc in Economics & Finance  
📧 [julianandres.thomas@gmail.com](mailto:julianandres.thomas@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/julianandresthomas)  
