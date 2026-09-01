"""
================================================================================
STEP 2: VERIFY NOISE REMOVAL QUALITY (LAYMAN SCRIPT)
================================================================================
Runs the controlled noise validation experiment on 100 Hz and 500 Hz ECG signals.
Proves that noise is eliminated while 100% of P-QRS-T complex shape is preserved.
Usage:
  PYTHONPATH=. python3 src/02_verify_denoising_quality.py
================================================================================
"""

from src.validate_denoising import run_denoising_validation

if __name__ == "__main__":
    print("Running Controlled Noise Removal Verification for 100 Hz...")
    run_denoising_validation(sampling_rate=100, target_snr=10.0)

    print("\nRunning Controlled Noise Removal Verification for 500 Hz...")
    run_denoising_validation(sampling_rate=500, target_snr=10.0)
