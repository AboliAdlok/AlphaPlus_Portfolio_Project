import pandas as pd

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
df = pd.read_csv("StockPriceDataset.csv")

print("First 5 rows of dataset:")
print(df.head())

# -------------------------------
# Step 2: Check Dataset Information
# -------------------------------
print("\nDataset Information:")
print(df.info())

print("\nDataset Description:")
print(df.describe())

# -------------------------------
# Step 3: Convert Date Column to Datetime
# -------------------------------
df['Date'] = pd.to_datetime(df['Date'])

# -------------------------------
# Step 4: Select Portfolio Assets
# -------------------------------
# Choose stocks for portfolio
assets = ['AAPL', 'MSFT', 'AMZN', 'JPM', 'XOM']

portfolio = df[df['Ticker'].isin(assets)]

print("\nSelected Portfolio Assets:")
print(portfolio['Ticker'].unique())

# -------------------------------
# Step 5: Check Missing Data
# -------------------------------
print("\nMissing Values in Dataset:")
print(portfolio.isnull().sum())

# Fill missing values using Forward Fill
portfolio = portfolio.ffill()

# -------------------------------
# Step 6: Sort Data by Date
# -------------------------------
portfolio = portfolio.sort_values(by="Date")
print("\nPortfolio Information:")
print(portfolio.info())


# -------------------------------
# Step 7: Data Integrity Check
# -------------------------------
print("\nDuplicate Rows:")
print(portfolio.duplicated().sum())

# Remove duplicates if any
portfolio = portfolio.drop_duplicates()

# -------------------------------
# Step 8: Select Adjusted Close Price
# -------------------------------
price_data = portfolio[['Date', 'Ticker', 'Adj Close']]

print("\nPrice Data Sample:")
print(price_data.head())


# -------------------------------
# Step 9: Create Pivot Table
# -------------------------------
price_pivot = price_data.pivot(index='Date', columns='Ticker', values='Adj Close')

print("\nPivot Table Sample:")
print(price_pivot.head())

# -------------------------------
# Step 10: Save Clean Dataset
# -------------------------------
price_pivot.to_csv("clean_portfolio_data.csv")

print("\nClean dataset saved as clean_portfolio_data.csv")