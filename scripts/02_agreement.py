"""Computes inter-annotator agreement (Cohen's kappa) between two label files
sharing a global_id column, as reported in the paper (kappa = 0.615 on the
100-post overlap)."""
import sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score

a = pd.read_csv(sys.argv[1]).set_index('global_id')['Category'].str.strip().str.upper()
b = pd.read_csv(sys.argv[2]).set_index('global_id')['Category'].str.strip().str.upper()
ids = a.index.intersection(b.index)
kappa = cohen_kappa_score(a[ids], b[ids])
raw = (a[ids] == b[ids]).mean()
print(f"n={len(ids)}  kappa={kappa:.3f}  raw agreement={raw:.1%}")
