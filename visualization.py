import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def display_visualizations(data, analysis=None, document_type="Bank Statement"):
    """Enhanced visualization function with better bank statement analysis"""

    try:
        if document_type == "Bank Statement":
            if not data.empty:
                st.subheader("Bank Statement Analysis")
                # Statistical Summary
                st.subheader("Transaction Statistics")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Credits", f"₹{data['Credit'].sum():,.2f}")
                with col2:
                    st.metric("Total Debits", f"₹{data['Debit'].sum():,.2f}")
                with col3:
                    st.metric(
                        "Net Flow", f"₹{(data['Credit'].sum() - data['Debit'].sum()):,.2f}")

                # Additional metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Transaction",
                              f"₹{((data['Credit'].sum() + data['Debit'].sum()) / len(data)):,.2f}")
                with col2:
                    st.metric("Transaction Count", f"{len(data)}")
                with col3:
                    st.metric("Balance", f"₹{23.57}")

                # Transaction Flow Analysis
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Credits',
                    x=data['Date'],
                    y=data['Credit'],
                    marker_color='green'
                ))
                fig.add_trace(go.Bar(
                    name='Debits',
                    x=data['Date'],
                    y=data['Debit'],
                    marker_color='red'
                ))
                fig.update_layout(
                    title="Transaction Flow Over Time",
                    barmode='group',
                    xaxis_title="Date",
                    yaxis_title="Amount"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Balance Trend
                fig = px.line(data, x='Date', y='Balance',
                              title="Balance Trend Over Time")
                fig.update_traces(line_color='blue')
                st.plotly_chart(fig, use_container_width=True)

                # Transaction Category Analysis
                if 'Description' in data.columns:
                    # Get top 10 transaction descriptions by frequency
                    top_descriptions = data['Description'].value_counts().head(
                        10)
                    fig = px.pie(values=top_descriptions.values,
                                 names=top_descriptions.index,
                                 title="Top Transaction Categories")
                    st.plotly_chart(fig, use_container_width=True)

                # Transaction Type Distribution
                try:
                    monthly_summary = data.groupby(pd.to_datetime(data['Date']).dt.strftime('%Y-%m'))\
                        .agg({
                            'Debit': 'sum',
                            'Credit': 'sum',
                            'Description': 'count'
                        })\
                        .reset_index()

                    fig = px.bar(monthly_summary, x='Date', y=[
                                 'Debit', 'Credit'], title="Monthly Transaction Summary", barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    pass  # Suppress errors

        elif document_type == "Invoice":
            if analysis and data:
                st.subheader("Invoice Summary")

                st.subheader("Invoice Visualizations")
                if 'items' in data:
                    items_df = pd.DataFrame(data['items'])
                    if not items_df.empty:
                        # Bar chart for item prices
                        fig = px.bar(items_df, x='description',
                                     y='price', title="Item Prices")
                        st.plotly_chart(fig, use_container_width=True)

                        # Pie chart for item distribution
                        fig = px.pie(items_df, names='description',
                                     values='price', title="Item Price Distribution")
                        st.plotly_chart(fig, use_container_width=True)

                if 'summary' in data:
                    summary_df = pd.DataFrame(list(data['summary'].items()), columns=[
                                              'Category', 'Amount'])
                    if not summary_df.empty:
                        # Bar chart for summary amounts
                        fig = px.bar(summary_df, x='Category',
                                     y='Amount', title="Invoice Summary")
                        st.plotly_chart(fig, use_container_width=True)

        elif document_type == "PnL":
            if not data.empty:
                st.subheader("Profit & Loss Summary")

                st.subheader("Profit & Loss Visualizations")
                # Line chart for YoY Change
                if 'YoY_Change' in data.columns:
                    fig = px.line(data, x='Category', y='YoY_Change',
                                  title="Year-over-Year Change")
                    st.plotly_chart(fig, use_container_width=True)

                # Bar chart for 2021 vs 2022
                if '2021' in data.columns and '2022' in data.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='2021',
                        x=data['Category'],
                        y=data['2021'],
                        marker_color='#2973B2'
                    ))
                    fig.add_trace(go.Bar(
                        name='2022',
                        x=data['Category'],
                        y=data['2022'],
                        marker_color='#73C7C7'
                    ))
                    fig.update_layout(
                        title="2021 vs 2022 Comparison",
                        barmode='group',
                        xaxis_title="Category",
                        yaxis_title="Amount"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during visualization: {e}")
        st.write("Debugging Info:")
        st.write(f"Document Type: {document_type}")
        st.write(f"Data: {data.head() if hasattr(data, 'head') else data}")
        st.write(f"Analysis: {analysis}")


def display_unsupervised_visualizations(data, category_stats):
    """Display visualizations for unsupervised analysis results"""
    st.subheader("Category Analysis")

    # Category-wise Transaction Distribution
    fig = px.pie(data, names='Category',
                 title="Transaction Distribution by Category")
    st.plotly_chart(fig, use_container_width=True)

    # Category-wise Net Flow
    fig = px.bar(category_stats, x='Category', y='Net',
                 title="Net Transaction Flow by Category",
                 color='Net',
                 color_continuous_scale=['red', 'green'])
    st.plotly_chart(fig, use_container_width=True)

    # Category-wise Debit vs Credit
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Credits',
        x=category_stats['Category'],
        y=category_stats['Credit'],
        marker_color='green'
    ))
    fig.add_trace(go.Bar(
        name='Debits',
        x=category_stats['Category'],
        y=category_stats['Debit'],
        marker_color='red'
    ))
    fig.update_layout(
        title="Category-wise Transaction Breakdown",
        barmode='group',
        xaxis_title="Category",
        yaxis_title="Amount"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Category-wise Transaction Count
    category_counts = data['Category'].value_counts()
    fig = px.bar(x=category_counts.index, y=category_counts.values,
                 title="Number of Transactions per Category")
    st.plotly_chart(fig, use_container_width=True)

    # Display category statistics
    st.subheader("Category Statistics")
    st.dataframe(category_stats.style.format({
        'Debit': '{:,.2f}',
        'Credit': '{:,.2f}',
        'Net': '{:,.2f}'
    }))
