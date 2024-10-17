#!/bin/bash

# Set the path to your Python executable
PYTHON_PATH="/home/data_science/project_files/santhosh/mainenv/bin/python"

# Set the path to your project directory
PROJECT_DIR="/home/ntlpt59/Documents/checkbox"

# Change to the project directory
cd "$PROJECT_DIR"

# Run the main checkbox detection script
echo "Running checkbox detection..."
$PYTHON_PATH -m client_code.trade_finance_structure_document.src.main.checkbox_detection.main
