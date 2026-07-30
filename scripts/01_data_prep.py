"""Reproduces the HM-800 sampling from the CONSTRAINT 2021 Hindi dataset.
Requires: git clone https://github.com/mohit19014/Hindi-Hostility-Detection-CONSTRAINT-2021
Outputs the exploration/annotation/pilot splits exactly as used in the paper (seed 42)."""
import pandas as pd

dfs = [pd.read_csv(f'Hindi-Hostility-Detection-CONSTRAINT-2021/Dataset/{f}')
       for f in ['train.csv','valid.csv','test.csv']]
full = pd.concat(dfs, ignore_index=True)
fake = (full[full['Labels Set'].str.contains('fake', na=False)]
        .drop_duplicates(subset='Post').reset_index(drop=True))
fake['global_id'] = ['HM' + str(i).zfill(4) for i in range(len(fake))]
assert len(fake) == 1632

shuffled = fake.sample(frac=1, random_state=42).reset_index(drop=True)
shuffled.iloc[:150].to_csv('exploration_150.csv', index=False)
shuffled.iloc[150:950].to_csv('annotation_800.csv', index=False)
shuffled.iloc[950:1000].to_csv('pilot_50.csv', index=False)
print("Splits written: 150 / 800 / 50 (seed 42, deduplicated pool of 1632)")
