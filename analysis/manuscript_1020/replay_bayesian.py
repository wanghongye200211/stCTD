"""Replay the saved S34 series using the unchanged historical Bayesian module."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from score_statistics import analyze

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
out = args.output_dir.resolve()
if out.exists() and any(out.iterdir()):
    raise SystemExit('Choose a new or empty output directory.')
out.mkdir(parents=True, exist_ok=True)
scores = pd.read_csv(root / 'inputs/scores.csv')
expected = pd.read_csv(root / 'expected/change_time_posterior.csv')
keys = ['mouse', 'zebrafish_B04', 'zebrafish_B07',
        'axolotl_reaEGC', 'axolotl_VLMC', 'axolotl_CMPN']
records, checks = [], []
for key in keys:
    d = scores[scores.series.eq(key)].sort_values('time')
    times = d.time.to_numpy()
    candidates, prob, conditional, _ = analyze(times, d.score.to_numpy())
    assert np.isfinite(prob).all() and np.isfinite(conditional).all()
    assert np.isclose(prob.sum(), 1) and np.isclose(conditional.sum(), 1)
    previous = expected[expected.series.eq(key)]
    np.testing.assert_allclose(prob, previous.posterior_probability, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(times[candidates], previous.candidate_time.iloc[1:], rtol=0, atol=1e-12)
    np.testing.assert_allclose(conditional, previous.conditional_time_probability.iloc[1:], rtol=1e-9, atol=1e-12)
    checks.append(dict(series=key, nodes=len(d), max_probability_error=float(np.max(np.abs(prob-previous.posterior_probability.to_numpy())))))
    for i, p in enumerate(prob):
        records.append(dict(series=key, model='no_slope_change' if i == 0 else 'one_slope_change',
                            candidate_time=None if i == 0 else float(times[candidates[i-1]]),
                            posterior_probability=float(p),
                            conditional_time_probability=None if i == 0 else float(conditional[i-1])))
pd.DataFrame(records).to_csv(out/'change_time_posterior.csv', index=False)
(out/'verification.json').write_text(json.dumps(dict(status='PASS', checks=checks), indent=2)+'\n')
print('PASS: all six biological S34 posterior series match the saved source values.')
