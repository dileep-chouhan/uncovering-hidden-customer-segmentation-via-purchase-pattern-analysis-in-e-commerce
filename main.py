import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_customers = 500
num_products = 10
# Generate random customer data
customers = pd.DataFrame({
    'CustomerID': range(1, num_customers + 1),
    'Age': np.random.randint(18, 65, num_customers),
    'Income': np.random.randint(20000, 150000, num_customers)
})
# Generate random product data
products = pd.DataFrame({
    'ProductID': range(1, num_products + 1),
    'Category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home Goods'], num_products),
    'Price': np.random.uniform(10, 1000, num_products)
})
# Generate random transaction data
num_transactions = 2000
transactions = pd.DataFrame({
    'CustomerID': np.random.choice(customers['CustomerID'], num_transactions),
    'ProductID': np.random.choice(products['ProductID'], num_transactions),
    'PurchaseDate': pd.to_datetime(np.random.choice(pd.date_range('2022-01-01', '2023-12-31'), num_transactions)),
    'Quantity': np.random.randint(1, 5, num_transactions)
})
# --- 2. Data Cleaning and Preparation ---
# Merge dataframes to get a comprehensive view of transactions
transaction_data = pd.merge(transactions, customers, on='CustomerID')
transaction_data = pd.merge(transaction_data, products, on='ProductID')
transaction_data['TotalAmount'] = transaction_data['Quantity'] * transaction_data['Price']
# --- 3. Analysis and Feature Engineering ---
# Calculate total spending per customer
customer_spending = transaction_data.groupby('CustomerID')['TotalAmount'].sum().reset_index()
customer_spending.rename(columns={'TotalAmount': 'TotalSpending'}, inplace=True)
# Calculate purchase frequency per customer
customer_frequency = transaction_data.groupby('CustomerID')['PurchaseDate'].count().reset_index()
customer_frequency.rename(columns={'PurchaseDate': 'PurchaseFrequency'}, inplace=True)
# Merge spending and frequency data
customer_behavior = pd.merge(customer_spending, customer_frequency, on='CustomerID')
# Calculate average purchase value per customer
customer_behavior['AveragePurchaseValue'] = customer_behavior['TotalSpending'] / customer_behavior['PurchaseFrequency']
# --- 4. Visualization ---
# Customer Segmentation based on spending and frequency
plt.figure(figsize=(10, 6))
sns.scatterplot(x='TotalSpending', y='PurchaseFrequency', data=customer_behavior, hue='Age')
plt.title('Customer Segmentation based on Spending and Frequency')
plt.xlabel('Total Spending')
plt.ylabel('Purchase Frequency')
plt.savefig('customer_segmentation.png')
print("Plot saved to customer_segmentation.png")
# Distribution of average purchase value
plt.figure(figsize=(10,6))
sns.histplot(customer_behavior['AveragePurchaseValue'], kde=True)
plt.title('Distribution of Average Purchase Value')
plt.xlabel('Average Purchase Value')
plt.ylabel('Frequency')
plt.savefig('avg_purchase_value.png')
print("Plot saved to avg_purchase_value.png")
#Most Popular Product Categories
plt.figure(figsize=(10,6))
category_counts = transaction_data['Category'].value_counts()
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.title('Most Popular Product Categories')
plt.xlabel('Product Category')
plt.ylabel('Number of Purchases')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('popular_categories.png')
print("Plot saved to popular_categories.png")