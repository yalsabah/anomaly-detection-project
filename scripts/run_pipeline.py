# scripts/run_pipeline.py
# Runs the complete tuned pipeline for credit card fraud detection

import os
import sys

print("Running Anomaly Detection Pipeline...")

# Assume data exists in /data and scripts in /models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from train_and_tune import main as run_main

run_main()
