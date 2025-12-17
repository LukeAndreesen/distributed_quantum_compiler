#!/usr/bin/env python3
"""
Convenience wrapper for tuning hyperparameters.

Delegates to `tuning/tune_params.py` so you can run:
  python tune_params.py --trials 50
"""
from tuning.tune_params import main

if __name__ == "__main__":
    main()
