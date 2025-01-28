import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


def visualize_data(data, chart_type="bar"):
    if data.empty:
        st.write("No data to visualize.")
        return

    if chart_type == "bar":
        visualize_bar_chart(data)
    elif chart_type == "pie":
        visualize_pie_chart(data)
    elif chart_type == "line":
        visualize_line_chart(data)
    else:
        st.write("Invalid chart type.")


def visualize_bar_chart(data):
    if "category" in data.columns and "amount" in data.columns:
        grouped_data = data.groupby("category")["amount"].sum()
        fig, ax = plt.subplots()
        grouped_data.plot(kind="bar", ax=ax,
                          color="skyblue", edgecolor="black")
        ax.set_title("Category-wise Distribution")
        ax.set_xlabel("Category")
        ax.set_ylabel("Amount")
        st.pyplot(fig)
    else:
        st.write("Bar chart requires 'category' and 'amount' columns.")


def visualize_pie_chart(data):
    if "category" in data.columns and "amount" in data.columns:
        grouped_data = data.groupby("category")["amount"].sum()
        fig,
