import pandas as pd
import matplotlib.pyplot as plt

#Load csv file
df = pd.read_csv("sales_data.csv")

#Clean missing values
df = df.dropna()

#Cleaned dataset
df.to_csv("cleaned_sales_data.csv")

#Summary Statistics
print("Summary Statistics")
print(df.describe())

#Generate Total Revenue
print("\nTotal Revenue by Category:")
df["revenue"] = df["price"] * df["quantity"]
total_revenue = df.groupby("category")["revenue"].sum()
print(total_revenue)

#Top 10 Customers
print("\nTop 10 Customers:")
top10 = df.groupby("customer_id")["revenue"].sum().sort_values(ascending=False).head(10).reset_index()
top10.index = range(1,11)
top10 = top10["customer_id"]
print(top10)

#Monthly Sales - Plot 1
df["invoice_date"] = pd.to_datetime(df["invoice_date"],dayfirst=True)
df["month"] = df["invoice_date"].dt.month
monthly_sales = df.groupby("month")["revenue"].sum()
monthly_sales.plot(kind="line",marker="o")
plt.title("Monthly Sales Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.show()

#Revenue by Category - Plot 2
total_revenue.plot(kind="bar")
plt.title("Revenue by Category")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.show()
