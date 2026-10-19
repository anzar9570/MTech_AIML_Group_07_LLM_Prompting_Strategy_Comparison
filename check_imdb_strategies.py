from pathlib import Path
import pandas as pd

path = Path('results/latest_details.csv')
if not path.exists():
    raise SystemExit('Missing results/latest_details.csv. Run and save a new evaluation first.')
df = pd.read_csv(path).fillna('')
df = df[(df['dataset'] == 'IMDb') & df['strategy'].isin(['good_few_shot', 'poor_few_shot'])]
if df.empty:
    raise SystemExit('No IMDb good/poor few-shot records in latest_details.csv')
good = df[df['strategy'] == 'good_few_shot'].set_index('sample_number')
poor = df[df['strategy'] == 'poor_few_shot'].set_index('sample_number')
ids = good.index.intersection(poor.index)
if len(ids) == 0:
    raise SystemExit('No matching sample_number values found.')
a, b = good.loc[ids], poor.loc[ids]
print('Matched IMDb samples:', len(ids))
for col in ('text', 'demonstration_examples', 'prompt', 'predicted_label', 'raw_response'):
    if col in a and col in b:
        matches = (a[col].astype(str) == b[col].astype(str)).sum()
        print(f'{col}: {matches}/{len(ids)} identical')
if 'valid_prediction' in a:
    print('Good invalid predictions:', (~a['valid_prediction'].astype(str).str.lower().eq('true')).sum())
    print('Poor invalid predictions:', (~b['valid_prediction'].astype(str).str.lower().eq('true')).sum())
print('\nFirst comparison:')
for i in ids[:1]:
    print('Sample:', i)
    print('Good demonstrations:\n', a.loc[i, 'demonstration_examples'])
    print('Poor demonstrations:\n', b.loc[i, 'demonstration_examples'])
    print('Good prediction:', a.loc[i, 'predicted_label'])
    print('Poor prediction:', b.loc[i, 'predicted_label'])
