"""Pytest bootstrap: make the repo root importable so ``btate``, the vendored
``bayes_tda`` (Maroulas posterior), and ``tcda_uq`` (frequentist bands) resolve
without an editable install. All three are top-level packages in this repo.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
