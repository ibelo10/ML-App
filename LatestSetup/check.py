import streamlit as st
import logging
import os

# Ensure the directory exists
os.makedirs('data', exist_ok=True)

# Set up logging
log_file = 'data/training_log_test.log'
logging.basicConfig(
    level=logging.DEBUG,  # Capture all levels of logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # Write logs to a file
        logging.StreamHandler()  # Also output logs to the console
    ]
)

# Test logging entry
logging.info("Streamlit-based logging setup test.")

st.write("Check the log file after running.")
