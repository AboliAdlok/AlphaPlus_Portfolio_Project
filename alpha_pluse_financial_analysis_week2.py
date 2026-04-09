import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------------------------
# Step 1: Load Clean Dataset
# ---------------------------------
price_data = pd.read_csv("clean_portfolio_data.csv")

print("First 5 rows of clean dataset:")
print(price_data.head())

# Convert Date column to datetime
price_data['Date'] = pd.to_datetime(price_data['Date'])

# Set Date as index
price_data.set_index('Date', inplace=True)

print("\nDataset Info:")
print(price_data.info())

# ---------------------------------
# Step 2: Calculate Daily Returns
# ---------------------------------
#daily_returns = price_data.pct_change()
daily_returns = np.log(price_data / price_data.shift(1))

print("\nDaily Returns Sample:")
print(daily_returns.head())

# ---------------------------------
# Step 3: Remove NA Values
# ---------------------------------
daily_returns = daily_returns.dropna()

# ---------------------------------
# Step 4: Calculate Volatility
# ---------------------------------
volatility = daily_returns.std()

print("\nVolatility of Each Stock:")
print(volatility)

# ---------------------------------
# Step 5: Calculate Correlation Matrix
# ---------------------------------
correlation_matrix = daily_returns.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# ---------------------------------
# Step 6: Plot Correlation Heatmap
# ---------------------------------
plt.figure(figsize=(8,6))
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

plt.title("Stock Correlation Heatmap")
plt.show()

# ---------------------------------
# Step 7: Plot Daily Returns
# ---------------------------------
daily_returns.plot(figsize=(10,6))

plt.title("Daily Returns of Portfolio Stocks")
plt.xlabel("Date")
plt.ylabel("Daily Return")

plt.show()

# ---------------------------------
# Step 9: Rolling 30 Day Volatility
# ---------------------------------

rolling_volatility = daily_returns.rolling(window=30).std()

rolling_volatility.plot(figsize=(10,6))

plt.title("30 Day Rolling Volatility")
plt.xlabel("Date")
plt.ylabel("Volatility")

plt.show()

# ---------------------------------
# Step 10: Define Portfolio Weights
# ---------------------------------
#weights = np.array([0.2,0.2,0.2,0.2,0.2])
num_assets = len(price_data.columns)
weights = np.array([1/num_assets] * num_assets)

# Calculate Portfolio Returns
portfolio_returns = daily_returns.dot(weights)

print("\nPortfolio Return Sample:")
print(portfolio_returns.head())

#-------------------------------------------------------------------------------
# ---------------------------------
# Sharpe Ratio Calculation
# ---------------------------------

risk_free_rate = 0.01 / 252   # assuming 1% yearly return → daily

sharpe_ratio = (portfolio_returns.mean() - risk_free_rate) / portfolio_returns.std()

print("\nSharpe Ratio:", sharpe_ratio)

#-------------------------------------------------------------------------------

# ---------------------------------
# Step 11: Monte Carlo Simulation
# ---------------------------------

num_simulations = 10000
num_days = 252

mean_returns = daily_returns.mean()
cov_matrix = daily_returns.cov()

#--------------------------------------------------------------------------------

# ---------------------------------
# Portfolio Volatility (Matrix Method)
# ---------------------------------

portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

print("\nPortfolio Volatility:", portfolio_volatility)

#---------------------------------------------------------------------------------

initial_portfolio = 100000

simulation_results = np.zeros((num_days, num_simulations))

for i in range(num_simulations):

    random_returns = np.random.multivariate_normal(mean_returns, cov_matrix, num_days)

    # portfolio_path = initial_portfolio * np.cumprod(1 + random_returns.dot(weights))
    portfolio_path = initial_portfolio * np.exp(np.cumsum(random_returns.dot(weights)))

    simulation_results[:, i] = portfolio_path

#--------------------------------------------------------------------------------------
# ---------------------------------
# Save Monte Carlo Simulation Output
# ---------------------------------

simulation_df = pd.DataFrame(simulation_results)

simulation_df.to_csv("monte_carlo_simulation.csv", index=False)

print("\nMonte Carlo simulation data saved!")

#--------------------------------------------------------------------------------------


# ---------------------------------
# Step 12 — Plot Monte Carlo Simulation
# ---------------------------------
plt.figure(figsize=(10,6))
#plt.plot(simulation_results[:, :100], alpha=0.3)
#plt.plot(simulation_results, alpha=0.1)
plt.plot(simulation_results[:, :100], alpha=0.2)
plt.title("Monte Carlo Simulation of Portfolio Value")
plt.xlabel("Days")
plt.ylabel("Portfolio Value")
plt.show()

# ---------------------------------
# Step 13: Value at Risk (95%)
# ---------------------------------

final_values = simulation_results[-1]

percentile_5 = np.percentile(final_values, 5)

VaR_95 = initial_portfolio - percentile_5

print("\nValue at Risk (95%):", VaR_95)
print(f"VaR at 95% confidence level means there is a 5% chance of losing more than {VaR_95:.2f}")
# ---------------------------------
# Step 14: Save Results
# ---------------------------------
daily_returns.to_csv("daily_returns.csv")
correlation_matrix.to_csv("correlation_matrix.csv")
volatility.to_csv("volatility.csv")
rolling_volatility.to_csv("rolling_volatility.csv")
portfolio_returns.to_csv("portfolio_returns.csv")
var_df = pd.DataFrame({"VaR_95":[VaR_95]})
var_df.to_csv("value_at_risk.csv", index=False)

print("\nWeek 2 analysis files saved successfully!")