# Beyond Fake or Real: A Journalism-Informed Taxonomy for Fine-Grained Misinformation Detection in Hindi Social Media

Code and results for the HM-800 project: a journalism-informed re-annotation of Hindi misinformation by *mechanism* (fabricated / manipulated / recirculated / missing-context / misrepresented satire) rather than binary fake-vs-real labels.

## Artifacts

- **Dataset (HM-800):** https://huggingface.co/datasets/tanmayvasvani/hindi-misinfo-taxonomy-800 (CC BY 4.0, with per-post rationales and adjudication flags)
- **Paper:** arXiv link forthcoming
- **Base corpus:** [CONSTRAINT 2021 Hindi Hostility Detection](https://github.com/mohit19014/Hindi-Hostility-Detection-CONSTRAINT-2021) (Bhardwaj et al., 2020)

## Repository structure

| Path | Contents |
|---|---|
| `notebooks/exp1_binary_baseline.ipynb` | Experiment 1: MuRIL binary fake detection on full CONSTRAINT (F1 = 0.816) |
| `notebooks/exp2b_taxonomy_classifier.ipynb` | Experiment 2b: class-weighted 3-class mechanism classifier (macro-F1 = 0.682) |
| `scripts/01_data_prep.py` | Reproduces the seed-42 sampling and splits |
| `scripts/02_agreement.py` | Cohen's kappa computation (κ = 0.615 on the 100-post overlap) |
| `results/` | Frozen result JSONs and the 21-error analysis file |

## Reproducing

Both notebooks are self-contained Colab notebooks (free T4 GPU): upload, set Runtime → T4 GPU, Run all. Seeds are fixed (42); reported numbers reproduce exactly.

Key headline: naive fine-tuning on the 3-class task collapses to the majority baseline (macro-F1 0.2202); class-weighted training recovers all classes (0.6819). Remaining errors concentrate entirely on verification-dependent category boundaries — see the paper's error analysis.

## Citation

Citation entry will be added upon arXiv posting. Please also cite the CONSTRAINT 2021 dataset (Bhardwaj et al., 2021, arXiv:2011.03588).
