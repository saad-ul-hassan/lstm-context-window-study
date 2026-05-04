import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 LSTM Context Window Study")

# Load data
df = pd.read_csv("results.csv")

# Show table
st.subheader("Results Table")
st.dataframe(df)

# Plot 1: Loss
st.subheader("Loss vs Sequence Length")
fig1, ax1 = plt.subplots()
ax1.plot(df["seq_length"], df["loss"], marker='o')
ax1.set_xlabel("Sequence Length")
ax1.set_ylabel("Loss")
st.pyplot(fig1)

# Plot 2: Time
st.subheader("Training Time vs Sequence Length")
fig2, ax2 = plt.subplots()
ax2.plot(df["seq_length"], df["time_sec"], marker='o')
ax2.set_xlabel("Sequence Length")
ax2.set_ylabel("Time (sec)")
st.pyplot(fig2)

# Plot 3: Memory
st.subheader("Memory Usage vs Sequence Length")
fig3, ax3 = plt.subplots()
ax3.plot(df["seq_length"], df["memory_MB"], marker='o')
ax3.set_xlabel("Sequence Length")
ax3.set_ylabel("Memory (MB)")
st.pyplot(fig3)