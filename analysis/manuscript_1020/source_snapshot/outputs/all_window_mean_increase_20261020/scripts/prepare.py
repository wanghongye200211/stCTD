"""Recompute saved-score diagnostics; direction is the difference in window means."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from score_statistics import posterior, analyze

O=Path(__file__).resolve().parents[1]; I=O/'inputs'; D=O/'source_data'
base=pd.read_csv(I/'all_scores.csv'); fish=pd.read_csv(I/'current_fish_scores.csv')
keys=fish.series.unique()
fish['time_unit']='hpf'
c=pd.concat([base[~base.series.isin(keys)],fish[base.columns]],ignore_index=True)
meta=pd.read_csv(I/'series_manifest.csv')
for key in keys:
 meta.loc[meta.series.eq(key),'score_definition']='Raw growth-weighted absolute potential increments and raw spatial transport; each divided by its first value; weights 0.9/0.1; no min-max or clipping'
 meta.loc[meta.series.eq(key),'source_file']='inputs/current_fish_scores.csv'
order={k:i for i,k in enumerate(meta.series)}
c['order']=c.series.map(order);c=c.sort_values(['order','time']).drop(columns='order').reset_index(drop=True)
c.to_csv(D/'scores.csv',index=False);meta.to_csv(D/'series_manifest.csv',index=False)
for name in ['mouse_HVG.csv','zebrafish_HVG.csv']:
 (D/name).write_bytes((I/name).read_bytes())
oldall=pd.read_csv(I/'previous_n4_statistics.csv')
oldfish=pd.read_csv(I/'current_fish_statistics.csv')
oldpost=pd.read_csv(I/'previous_change_time.csv');oldfishpost=pd.read_csv(I/'current_fish_change_time.csv')
local=[];windows=[];summaries=[];timepost=[];components=[];sensitivity=[];checks=[];manual=[]

for key in meta.series:
 d=c[c.series.eq(key)].sort_values('time');t=d.time.to_numpy();y=d.score.to_numpy()
 assert np.isfinite(y).all() and np.all(np.diff(t)>0)
 if len(y)<6:
  for n in [4,3]:
   summaries.append({'series':key,'n_per_window':n,'status':'not_estimable','n_score_nodes':len(y),
    'reason':f'Only {len(y)} saved nodes; {2*n} required for local test and at least 9 for current change-time model.'})
  continue
 sd=y.std(ddof=1);z=(y-y.mean())/sd
 for n in [4,3]:
  rows=[]
  for e in range(2*n-1,len(y)):
   a=y[e-2*n+1:e-n+1];b=y[e-n+1:e+1]
   tt=stats.ttest_ind(b,a,equal_var=False,alternative='greater')
   va=a.var(ddof=1)/n;vb=b.var(ddof=1)/n
   df=(va+vb)**2/(va**2/(n-1)+vb**2/(n-1))
   manual.append(abs(stats.t.sf((b.mean()-a.mean())/np.sqrt(va+vb),df)-tt.pvalue))
   pp,cs,weights=posterior(z[e-2*n+1:e+1]);variants=[]
   for v in [1.,4.,16.]:
    prob=posterior(z[e-2*n+1:e+1],v,ci=False)[0]['posterior_prob_increase'];variants.append(prob)
    sensitivity.append({'series':key,'n_per_window':n,'time':t[e],'analysis':'local_increase','prior_variance':v,'probability':prob})
   rows.append({'series':key,'n_per_window':n,'time':t[e],
    'earlier_start':t[e-2*n+1],'earlier_end':t[e-n],'recent_start':t[e-n+1],'recent_end':t[e],
    'earlier_mean':a.mean(),'recent_mean':b.mean(),'window_delta':b.mean()-a.mean(),
    'adjacent_delta':y[e]-y[e-1],'t_statistic':tt.statistic,'df':df,'p_increase':tt.pvalue,**pp,
    'posterior_mean_delta':pp['posterior_mean_delta_standardized']*sd,
    'credible_low':pp['credible_low_standardized']*sd,'credible_high':pp['credible_high_standardized']*sd,
    'prior_sensitivity_low':min(variants),'prior_sensitivity_high':max(variants)})
   for comp,weight in zip(cs,weights):components.append({'series':key,'n_per_window':n,'time':t[e],**comp,'posterior_weight':weight})
   for j in range(e-2*n+1,e+1):
    assert j<=e
    windows.append({'series':key,'n_per_window':n,'tested_endpoint':t[e],'time':t[j],'score':y[j],
     'group':'earlier' if j<=e-n else 'recent'})
  w=pd.DataFrame(rows);w['q_bh']=stats.false_discovery_control(w.p_increase.to_numpy())
  w['positive_window']=w.window_delta.gt(0)
  w['nominal_rising_pass']=w.positive_window&w.p_increase.lt(.05)
  w['BH_rising_pass']=w.positive_window&w.q_bh.lt(.05)
  w['previous_adjacent_gate_nominal_pass']=w.nominal_rising_pass&w.adjacent_delta.gt(0)
  w['previous_adjacent_gate_BH_pass']=w.BH_rising_pass&w.adjacent_delta.gt(0)
  w['newly_retained_nominal']=w.nominal_rising_pass&~w.previous_adjacent_gate_nominal_pass
  w['newly_retained_BH']=w.BH_rising_pass&~w.previous_adjacent_gate_BH_pass
  for field in ['nominal_rising_pass','BH_rising_pass']:
   w[field.replace('_pass','_run_start')]=w[field]&~w[field].shift(fill_value=False)
  # Preserve existing series-specific display thresholds; only remove adjacent-point gating.
  if key in ['mouse','zebrafish_B04','zebrafish_B07']:
   w['display_marker']=False
   if w.nominal_rising_pass.any():w.loc[w.index[w.nominal_rising_pass][0],'display_marker']=True
   w['display_rule']='First one-sided raw P < 0.05 with recent mean > earlier mean'
  else:
   w['display_marker']=w.BH_rising_run_start
   w['display_rule']='First endpoint of each BH q < 0.05 positive-window run'
  first=w[w.nominal_rising_pass].head(1);oldfirst=w[w.previous_adjacent_gate_nominal_pass].head(1)
  fq=w[w.BH_rising_pass].head(1)
  summaries.append({'series':key,'n_per_window':n,'status':'available','n_score_nodes':len(y),'n_tests':len(w),
   'first_nominal_time':first.time.iloc[0] if len(first) else np.nan,
   'first_nominal_P':first.p_increase.iloc[0] if len(first) else np.nan,
   'q_at_first_nominal':first.q_bh.iloc[0] if len(first) else np.nan,
   'first_BH_time':fq.time.iloc[0] if len(fq) else np.nan,
   'previous_adjacent_gate_first_time':oldfirst.time.iloc[0] if len(oldfirst) else np.nan,
   'newly_retained_nominal_count':int(w.newly_retained_nominal.sum()),
   'newly_retained_BH_count':int(w.newly_retained_BH.sum()),
   'displayed_times':'; '.join(f'{x:g}' for x in w.loc[w.display_marker,'time'])})
  if key in keys:
   prev=oldfish[oldfish.series.eq(key)&oldfish.n_per_window.eq(n)].sort_values('time')
   aliases={'p_increase':'p_increase','q_bh':'q_bh','posterior_prob_increase':'posterior_prob_increase'}
  elif n==4:
   prev=oldall[oldall.series.eq(key)].sort_values('window_end')
   aliases={'p_increase':'p_increase','q_bh':'q_increase_within_curve','posterior_prob_increase':'posterior_prob_increase'}
  else:prev=None
  if prev is not None:
   err={field:float(np.max(np.abs(w[field].to_numpy()-prev[other].to_numpy()))) for field,other in aliases.items()}
   assert max(err.values())<1e-10,(key,n,err)
   checks.append({'series':key,'n_per_window':n,**err})
  assert not w.loc[~w.positive_window,'nominal_rising_pass'].any()
  local.extend(w.to_dict('records'))
  d.merge(w.drop(columns='series'),on='time',how='left',validate='one_to_one').to_csv(D/f'{key}_n{n}_all_points.csv',index=False)
 j,prob,cond,_=analyze(t,y);cdf=cond.cumsum()
 variants={v:analyze(t,y,v)[1] for v in [1.,4.,16.]}
 for k,v in enumerate(prob):
  timepost.append({'series':key,'model':'no_slope_change' if k==0 else 'one_slope_change',
   'candidate_time':np.nan if k==0 else t[j[k-1]],'posterior_probability':v,
   'conditional_time_probability':np.nan if k==0 else cond[k-1],
   'prior_sensitivity_low':min(x[k] for x in variants.values()),'prior_sensitivity_high':max(x[k] for x in variants.values())})
 before=oldfishpost if key in keys else oldpost
 err=float(np.max(np.abs(prob-before[before.series.eq(key)].posterior_probability.to_numpy())))
 assert err<1e-10,(key,err)
 checks.append({'series':key,'change_time_posterior_error':err})
 for row in summaries:
  if row['series']==key:
   row.update({'posterior_no_slope_change':prob[0],'conditional_mode':t[j[np.argmax(cond)]],
    'conditional_CI_low':t[j[np.searchsorted(cdf,.025)]],'conditional_CI_high':t[j[np.searchsorted(cdf,.975)]]})
 print(key,'complete',flush=True)

w=pd.DataFrame(local);p=pd.DataFrame(timepost)
for name,data in [('local_statistics.csv',w),('summary.csv',pd.DataFrame(summaries)),('change_time_posterior.csv',p),
 ('window_values.csv',pd.DataFrame(windows)),('local_rho_components.csv',pd.DataFrame(components)),
 ('prior_sensitivity.csv',pd.DataFrame(sensitivity)),('displayed_points.csv',w[w.display_marker]),
 ('newly_retained_points.csv',w[w.newly_retained_nominal]),('nominal_run_starts.csv',w[w.nominal_rising_run_start])]:
 data.to_csv(D/name,index=False)
assert len(c)==313 and len(w)==520 and max(manual)<1e-12
assert w[w.series.eq('zebrafish_B04')&w.n_per_window.eq(3)&np.isclose(w.time,13.1)].nominal_rising_pass.all()
contract={'date':'2026-10-20','scope':'All 9 saved score series; 8 analyzable plus 5-node smooth control',
 'direction':'recent window mean > earlier window mean; no adjacent-point sign filter',
 'score_policy':'All accepted source STE, cohorts, weights, smoothing, normalization and HVG curves retained; Fig4 uses the adopted raw-component first-reference version.',
 'test':'One-sided Welch alternative=greater; n=4 retained main comparison, n=3 sensitivity; two disjoint consecutive windows ending at current node',
 'BH':'Correct within each curve and n over every eligible window, not only positive windows',
 'display':'Retain first nominal P<0.05 for mouse/Fig4, first BH-passing run endpoints for simulations and Fig5. All nominal and BH results exported for all series.',
 'no_adjacent_gate':True,'Bayesian':'Same AR1 mixture models and priors; refitted, with previous matched outputs verified unchanged',
 'interpretation':'Window-level increase, not necessarily a positive instantaneous slope or start of a rise. Test endpoint is not window start.',
 'time_limits':'No later score nodes enter Welch windows. Precomputed trajectories and some score preprocessing use all times; Bayesian standardization and full-curve change-time analysis are retrospective. Fish x coordinates remain interval midpoints.',
 'unavailable':'Smooth control has five nodes; no invented P values or Bayesian results.',
 'Word_modified':False}
(O/'analysis_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
qa={'status':'PASS','score_nodes':len(c),'series':9,'analyzable_series':8,'windows_by_n':w.groupby('n_per_window').size().to_dict(),
 'manual_Welch_max_error':max(manual),'previous_matched_results':checks,
 'newly_retained_nominal':int(w.newly_retained_nominal.sum()),'newly_retained_BH':int(w.newly_retained_BH.sum()),
 'nonpositive_windows_marked':int(w.loc[w.window_delta.le(0),'display_marker'].sum()),
 'posterior_mass_sums':p.groupby('series').posterior_probability.sum().to_dict()}
(O/'audit/numerical_qa.json').write_text(json.dumps(qa,indent=2)+'\n')
print(pd.DataFrame(summaries)[['series','n_per_window','first_nominal_time','first_nominal_P','q_at_first_nominal','previous_adjacent_gate_first_time','displayed_times']].to_string(index=False))
