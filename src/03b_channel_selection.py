"""
================================================================================
STEP 3b: NON-STATIONARY SIGNAL ANALYSIS & CHANNEL SELECTION (LAYMAN SCRIPT)
================================================================================
Applies the 19-channel EEG Inter/Intra class variance method to 12-channel ECG
to select the 3 to 4 most unique and informative channels.
Usage:
  PYTHONPATH=. python3 src/03b_channel_selection.py
================================================================================
"""

from src.channel_selection import run_channel_selection

if __name__ == "__main__":
    run_channel_selection(top_k=4)
