"""Reproduce S4-S6 from the accepted main-figure score series."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, t
from statsmodels.stats.multitest import multipletests

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parent
OUT = args.output_dir.resolve()
if OUT.exists() and any(OUT.iterdir()):
    raise SystemExit('Choose a new or empty output directory.')
for folder in ['audit', 'source_data']:
    (OUT / folder).mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source = ROOT / 'inputs/scores.csv'
reference = ROOT / 'expected/local_statistics.csv'
scores = pd.read_csv(source)
saved = pd.read_csv(reference)
keys = ["mouse", "zebrafish_B04", "zebrafish_B07",
        "axolotl_reaEGC", "axolotl_VLMC", "axolotl_CMPN"]
records, checks = [], []
for key in keys:
    d = scores[scores.series.eq(key)].sort_values("time")
    x, y = d.time.to_numpy(), d.score.to_numpy()
    assert np.isclose(y[0], 1)
    expected_spacing = .1 if key == "mouse" else .2 if key.startswith("zebrafish") else .5
    assert np.allclose(np.diff(x), expected_spacing)
    for n in ([4, 2] if key in ["zebrafish_B04", "axolotl_reaEGC"] else [4]):
        batch = []
        for i in range(2*n-1, len(d)):
            a, b = y[i-2*n+1:i-n+1], y[i-n+1:i+1]
            v1, v2 = a.var(ddof=1)/n, b.var(ddof=1)/n
            stat = (b.mean()-a.mean()) / np.sqrt(v1+v2)
            df = (v1+v2)**2/(v1*v1/(n-1)+v2*v2/(n-1))
            p = float(t.sf(stat, df))
            assert np.isclose(p, ttest_ind(b, a, equal_var=False, alternative="greater").pvalue,
                              atol=1e-14, rtol=1e-12)
            batch.append(dict(series=key, n_per_window=n, time=float(x[i]),
                P=p, earlier_times=x[i-2*n+1:i-n+1].tolist(),
                recent_times=x[i-n+1:i+1].tolist(), earlier_scores=a.tolist(),
                recent_scores=b.tolist(), t=float(stat), df=float(df),
                earlier_mean=float(a.mean()), recent_mean=float(b.mean())))
        pp = np.array([r["P"] for r in batch])
        order = np.argsort(pp)
        qq = np.empty(len(pp))
        qq[order] = np.minimum(1, np.minimum.accumulate(
            (pp[order]*len(pp)/np.arange(1,len(pp)+1))[::-1])[::-1])
        assert np.allclose(qq, multipletests(pp, method="fdr_bh")[1], atol=1e-14)
        for r, q in zip(batch, qq):
            r["BH_adjusted_P"] = float(q)
            r["family_size"] = len(batch)
            assert max(r["earlier_times"]) < min(r["recent_times"])
            assert max(r["recent_times"]) == r["time"]
        if n == 4:
            old = saved[saved.series.eq(key) & saved.n_per_window.eq(4)].sort_values("time")
            assert len(old) == len(batch)
            assert np.allclose(old.time, [r["time"] for r in batch])
            assert np.allclose(old.p_increase, pp, atol=1e-13)
            assert np.allclose(old.q_bh, qq, atol=1e-13)
        else:
            oldpath = (ROOT / "expected/fig4b_n2.csv"
                       if key == "zebrafish_B04" else
                       ROOT / "expected/fig5_n2.csv")
            old = pd.read_csv(oldpath).dropna(subset=["P_value"])
            assert np.allclose(old.P_value, pp, atol=1e-13)
            assert np.allclose(old.BH_adjusted_P, qq, atol=1e-13)
        checks.append(dict(series=key, n_per_window=n, tests=len(batch),
            first_test=batch[0]["time"], last_test=batch[-1]["time"], time_spacing=expected_spacing,
            raw_below_005=int(sum(pp < .05)), BH_below_005=int(sum(qq < .05)),
            reproduced_saved_results=True))
        records.extend(batch)

labels = {"mouse":"RGC origin", "zebrafish_B04":"NC/OT", "zebrafish_B07":"SP/TB",
          "axolotl_reaEGC":"reaRGC", "axolotl_VLMC":"VLMC", "axolotl_CMPN":"CMPN"}
tables = []
for num, table_keys, unit in [(4, keys[:1], "Embryonic day"),
                              (5, keys[1:3], "hpf"), (6, keys[3:], "DPI")]:
    rows = []
    # Four-point primary sections first, then the requested two-point sensitivity.
    for n in [4, 2]:
        for key in table_keys:
            for r in records:
                if r["series"] == key and r["n_per_window"] == n:
                    rows.append([f"{labels[key]} (n = {n})", round(r["time"], 1), r["P"], r["BH_adjusted_P"]])
    tables.append(dict(number=num, headers=["Group (points per window)", f"Time ({unit})", "P", "BH-adjusted P"],
                       rows=rows))
    pd.DataFrame(rows, columns=["Group (points per window)",f"Time ({unit})","P","BH-adjusted P"]).to_csv(
        OUT / f"source_data/Table_S{num}.csv", index=False)

(OUT / "source_data/tables_payload.json").write_text(json.dumps(tables,indent=2))
(OUT / "source_data/window_audit.json").write_text(json.dumps(records,indent=2))
scores[scores.series.isin(keys)].to_csv(OUT / "source_data/unchanged_scores.csv",index=False)
audit = dict(source=str(source), source_sha256=sha(source), primary="One-sided Welch, recent 4 versus preceding 4",
    sensitivity="Recent 2 versus preceding 2, only NC/OT and reaRGC",
    scan_stride=1, correction="BH within each series and window size, all eligible endpoints",
    checks=checks, table_rows={f"S{x['number']}":len(x["rows"]) for x in tables})
(OUT / "audit/statistical_verification.json").write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))

for number in [4, 5, 6]:
    actual = pd.read_csv(OUT / f'source_data/Table_S{number}.csv')
    expected = pd.read_csv(ROOT / f'expected/Table_S{number}.csv')
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-10, atol=1e-13)
print('PASS: 256 rows of Tables S4-S6 match the frozen manuscript inputs.')
