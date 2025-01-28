import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Load the synthetic data
df = pd.read_csv("synthetic_transactions.csv")

# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Aggregate spending by category (MCC Description)
category_spending = df.groupby('Description')['Amount'].sum().reset_index()

# 1. Category-wise Spending (Pie Chart)
fig_pie = px.pie(category_spending, values='Amount', names='Description',
                 title='Category-wise Spending Distribution',
                 color_discrete_sequence=px.colors.sequential.Plasma)

fig_pie.update_traces(textinfo='percent+label',
                      pull=[0.1] + [0] * (len(category_spending)-1))
fig_pie.show()

# 2. Monthly Spending Trends (Line Chart)
df['Month_Year'] = df['Date'].dt.to_period(
    'M').astype(str)  # Convert to string

monthly_spending = df.groupby('Month_Year')['Amount'].sum().reset_index()

fig_line = px.line(monthly_spending, x='Month_Year', y='Amount',
                   title='Monthly Spending Trends',
                   markers=True, line_shape='spline')

fig_line.update_layout(xaxis_title='Month', yaxis_title='Total Spending',
                       template='plotly_dark', xaxis=dict(type='category'))

fig_line.show()

# 3. Payment Method Distribution (Bar Chart)
payment_distribution = df.groupby('Payment_Method')[
    'Amount'].sum().reset_index()

fig_bar = px.bar(payment_distribution, x='Payment_Method', y='Amount',
                 title='Payment Method Distribution', color='Payment_Method',
                 text_auto=True, template='plotly_white')

fig_bar.update_layout(xaxis_title='Payment Method',
                      yaxis_title='Total Spending')
fig_bar.show()

# 4. Top Spending Categories (Bar Chart)
top_categories = category_spending.sort_values(
    by='Amount', ascending=False).head(10)

fig_top_bar = px.bar(top_categories, x='Amount', y='Description',
                     title='Top 10 Spending Categories',
                     orientation='h', color='Amount',
                     template='plotly_dark')

fig_top_bar.update_layout(xaxis_title='Total Spending', yaxis_title='Category')
fig_top_bar.show()

# 5. Daily Spending Heatmap
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month

# Create pivot table for heatmap
daily_spending = df.pivot_table(
    index='Day', columns='Month', values='Amount', aggfunc='sum').fillna(0)

plt.figure(figsize=(10, 6))
sns.heatmap(daily_spending, cmap='coolwarm',
            annot=True, fmt='.0f', linewidths=0.5)

plt.title('Daily Spending Heatmap')
plt.xlabel('Month')
plt.ylabel('Day of Month')
plt.show()
