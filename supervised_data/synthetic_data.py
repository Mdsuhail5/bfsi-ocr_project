import random
import pandas as pd
from faker import Faker

# Initialize Faker for synthetic data generation
fake = Faker()

# Define spending categories and common transactions
transaction_categories = [
    {"MCC": "5411", "Description": "Grocery Stores, Supermarkets",
        "Min": 50, "Max": 200, "Frequency": 8},
    {"MCC": "5541",
        "Description": "Service Stations (Fuel)", "Min": 40, "Max": 150, "Frequency": 4},
    {"MCC": "4900",
        "Description": "Utilities (Electric, Gas, Water)", "Min": 80, "Max": 300, "Frequency": 1},
    {"MCC": "5812", "Description": "Eating Places and Restaurants",
        "Min": 20, "Max": 100, "Frequency": 6},
    {"MCC": "4814",
        "Description": "Telecommunication Services (Internet, Phone)", "Min": 50, "Max": 120, "Frequency": 1},
    {"MCC": "6011",
        "Description": "Automated Cash Disbursements (ATM Withdrawals)", "Min": 100, "Max": 500, "Frequency": 3},
    {"MCC": "7311",
        "Description": "Streaming Subscriptions (Netflix, Spotify)", "Min": 10, "Max": 30, "Frequency": 1},
    {"MCC": "4111",
        "Description": "Transportation (Public, Ride-Sharing)", "Min": 10, "Max": 50, "Frequency": 4},
    {"MCC": "8099",
        "Description": "Healthcare (Medical, Pharmacy)", "Min": 50, "Max": 300, "Frequency": 2},
    {"MCC": "5533", "Description": "Automotive Parts and Accessories",
        "Min": 100, "Max": 500, "Frequency": 1},
    {"MCC": "8699", "Description": "Gym Membership",
        "Min": 30, "Max": 100, "Frequency": 1},
    {"MCC": "5942",
        "Description": "Book Stores (Educational, Hobby)", "Min": 15, "Max": 80, "Frequency": 2},
    {"MCC": "6513", "Description": "Rent Payments",
        "Min": 800, "Max": 1500, "Frequency": 1},
    {"MCC": "5300",
        "Description": "Wholesale Clubs (Bulk Shopping)", "Min": 150, "Max": 400, "Frequency": 2},
]

payment_methods = ["Credit Card", "Debit Card",
                   "Cash", "Bank Transfer", "PayPal"]

# Generate synthetic transactions for 6 months


def generate_personal_transactions(num_months=6):
    transactions = []

    for month in range(num_months):
        month_date = fake.date_between(
            start_date="-6M", end_date="today").strftime("%Y-%m")

        for category in transaction_categories:
            for _ in range(category["Frequency"]):
                transaction = {
                    "Date": fake.date_between_dates(
                        date_start=pd.to_datetime(month_date + "-01"),
                        date_end=pd.to_datetime(month_date + "-28")
                    ).strftime("%Y-%m-%d"),
                    "Description": category["Description"],
                    "Amount": round(random.uniform(category["Min"], category["Max"]), 2),
                    "MCC": category["MCC"],
                    "Payment_Method": random.choice(payment_methods),
                }
                transactions.append(transaction)

    return pd.DataFrame(transactions)


# Generate transactions
synthetic_data_df = generate_personal_transactions(num_months=6)

# Save to CSV file
synthetic_data_df.to_csv("synthetic_transactions.csv", index=False)

print("Synthetic personal transaction data generated and saved to 'synthetic_transactions.csv'")
print(synthetic_data_df.head())
