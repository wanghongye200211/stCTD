"""Recover raw Fig4 components, normalize each to its first interval, and refit diagnostics."""
from pathlib import Path
import hashlib, json, shutil
import numpy as np
import pandas as pd
from scipy import stats
from score_statistics import posterior, analyze

O=Path(__file__).resolve().parents[1]; R=O.parents[1]
sources={
 'raw_components.csv':'result_figures/zesta/lineage_critical_compact/figure5c_same_score_nstep_welch_2_3_4_dt0p2/zesta_same_score_nstep_welch_2_3_4_unchanged_score_source.csv',
 'previous_scores.csv':'outputs/all_transition_statistics_fig2_0703_20261018/source_data/scores.csv',
 'hvg.csv':'outputs/all_transition_statistics_fig2_0703_20261018/source_data/zebrafish_HVG_source.csv',
}
manifest=[]
for name,rel in sources.items():
 p=O/'inputs'/name
 if not p.exists():shutil.copy2(R/rel,p)
 manifest.append({'file':name,'original_source':rel,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
raw=pd.read_csv(O/'inputs/raw_components.csv')
previous=pd.read_csv(O/'inputs/previous_scores.csv')
shutil.copy2(O/'inputs/hvg.csv',O/'source_data/hvg.csv')
frames=[];checks=[]
for code,ncells in [('B04',41),('B07',339)]:
 g=raw[raw.branch_id.eq('zebrafish_branch'+code[1:])].sort_values('time').reset_index(drop=True)
 key='zebrafish_'+code
 old=previous[previous.series.eq(key)].sort_values('time').reset_index(drop=True)
 assert len(g)==40 and g.n_subset.eq(ncells).all() and g.stepU_aggregation.eq('weighted_mean').all()
 np.testing.assert_allclose(g.time,old.time,rtol=0,atol=1e-12)
 np.testing.assert_allclose(g.score/g.score.iloc[0],old.score,rtol=0,atol=1e-12)
 p=g.weighted_mean_abs_step_delta_U.to_numpy();s=g.spatial_ot_balanced.to_numpy()
 assert np.isfinite(p).all() and np.isfinite(s).all() and (p>0).all() and (s>0).all()
 score=.9*p/p[0]+.1*s/s[0]
 assert score[0]==1 and (score>0).all()
 d=pd.DataFrame({'series':key,'time':g.time,'interval_start':g.start_time,'interval_end':g.end_time,
                 'n_cells':ncells,'potential_raw':p,'spatial_raw':s,'potential_first':p[0],
                 'spatial_first':s[0],'potential_ratio':p/p[0],'spatial_ratio':s/s[0],
                 'score':score,'previous_minmax_score_first_one':old.score,
                 'potential_weight':.9,'spatial_weight':.1})
 frames.append(d)
 checks.append({'series':key,'first':float(score[0]),'minimum':float(score.min()),
                'minimum_time':float(g.time.iloc[np.argmin(score)]),'maximum':float(score.max()),
                'last':float(score[-1]),'previous_zero_nodes':int(old.score.eq(0).sum()),
                'new_zero_nodes':int(np.count_nonzero(score==0)),'potential_first':float(p[0]),'spatial_first':float(s[0])})
c=pd.concat(frames,ignore_index=True)
c.to_csv(O/'source_data/scores_and_raw_components.csv',index=False)
locals=[];timepost=[];summaries=[];components=[];sensitivity=[];windows=[];manual=[]
for key,d in c.groupby('series',sort=False):
 t=d.time.to_numpy();y=d.score.to_numpy();sd=y.std(ddof=1);z=(y-y.mean())/sd
 for n in [3,4]:
  rr=[]
  for e in range(2*n-1,len(y)):
   a=y[e-2*n+1:e-n+1];b=y[e-n+1:e+1]
   test=stats.ttest_ind(b,a,equal_var=False,alternative='greater')
   va=a.var(ddof=1)/n;vb=b.var(ddof=1)/n;df=(va+vb)**2/(va**2/(n-1)+vb**2/(n-1))
   manual.append(abs(stats.t.sf((b.mean()-a.mean())/np.sqrt(va+vb),df)-test.pvalue))
   pp,cs,weights=posterior(z[e-2*n+1:e+1])
   probs=[]
   for prior in [1.,4.,16.]:
    v=posterior(z[e-2*n+1:e+1],prior,ci=False)[0]['posterior_prob_increase'];probs.append(v)
    sensitivity.append({'series':key,'n_per_window':n,'analysis':'local_increase','time':t[e],'prior_variance':prior,'probability':v})
   rr.append({'series':key,'n_per_window':n,'time':t[e],
              'earlier_start':t[e-2*n+1],'earlier_end':t[e-n],
              'recent_start':t[e-n+1],'recent_end':t[e],
              'earlier_mean':a.mean(),'recent_mean':b.mean(),'window_delta':b.mean()-a.mean(),
              'adjacent_delta':y[e]-y[e-1],'t_statistic':test.statistic,'df':df,'p_increase':test.pvalue,
              **pp,'posterior_mean_delta':pp['posterior_mean_delta_standardized']*sd,
              'credible_low':pp['credible_low_standardized']*sd,'credible_high':pp['credible_high_standardized']*sd,
              'prior_sensitivity_low':min(probs),'prior_sensitivity_high':max(probs)})
   for comp,weight in zip(cs,weights):components.append({'series':key,'n_per_window':n,'time':t[e],**comp,'posterior_weight':weight})
   for j in range(e-2*n+1,e+1):
    assert t[j]<=t[e]
    windows.append({'series':key,'n_per_window':n,'tested_endpoint':t[e],'time':t[j],'score':y[j],'group':'earlier' if j<=e-n else 'recent'})
  w=pd.DataFrame(rr);w['q_bh']=stats.false_discovery_control(w.p_increase.to_numpy(),method='bh')
  w['nominal_rising_pass']=w.p_increase.lt(.05)&w.window_delta.gt(0)&w.adjacent_delta.gt(0)
  w['BH_rising_pass']=w.q_bh.lt(.05)&w.window_delta.gt(0)&w.adjacent_delta.gt(0)
  assert not w.loc[w.adjacent_delta.le(0),'nominal_rising_pass'].any()
  first=w[w.nominal_rising_pass].head(1);firstq=w[w.BH_rising_pass].head(1)
  summaries.append({'series':key,'n_per_window':n,'n_tests':len(w),
                    'first_nominal_rising_time':float(first.time.iloc[0]) if len(first) else np.nan,
                    'first_nominal_P':float(first.p_increase.iloc[0]) if len(first) else np.nan,
                    'q_at_first_nominal':float(first.q_bh.iloc[0]) if len(first) else np.nan,
                    'first_BH_rising_time':float(firstq.time.iloc[0]) if len(firstq) else np.nan})
  locals.extend(w.to_dict('records'))
 j,prob,cond,evidence=analyze(t,y);cdf=cond.cumsum()
 variants={prior:analyze(t,y,prior)[1] for prior in [1.,4.,16.]}
 for k,v in enumerate(prob):
  timepost.append({'series':key,'model':'no_slope_change' if k==0 else 'one_slope_change',
                   'candidate_time':np.nan if k==0 else t[j[k-1]],'posterior_probability':v,
                   'conditional_time_probability':np.nan if k==0 else cond[k-1],
                   'prior_sensitivity_low':min(x[k] for x in variants.values()),
                   'prior_sensitivity_high':max(x[k] for x in variants.values())})
 for r in summaries:
  if r['series']==key:
   r.update({'posterior_no_slope_change':float(prob[0]),'conditional_mode':float(t[j[np.argmax(cond)]]),
             'conditional_CI_low':float(t[j[np.searchsorted(cdf,.025)]]),
             'conditional_CI_high':float(t[j[np.searchsorted(cdf,.975)]])})
 print(key,'complete',flush=True)
w=pd.DataFrame(locals)
for name,data in [('local_statistics.csv',w),('change_time_posterior.csv',pd.DataFrame(timepost)),
                  ('summary.csv',pd.DataFrame(summaries)),('score_summary.csv',pd.DataFrame(checks)),
                  ('local_rho_components.csv',pd.DataFrame(components)),('prior_sensitivity.csv',pd.DataFrame(sensitivity)),
                  ('window_values.csv',pd.DataFrame(windows))]:data.to_csv(O/'source_data'/name,index=False)
for key,d in c.groupby('series',sort=False):
 for n in [3,4]:
  q=w[w.series.eq(key)&w.n_per_window.eq(n)].drop(columns='series')
  a=d.merge(q,on='time',how='left',validate='one_to_one');a['adjacent_delta']=a.score.diff()
  a.to_csv(O/'source_data'/f'{key}_n{n}_all_points.csv',index=False)
assert len(w)==136 and max(manual)<1e-12
contract={'scope':'Fig4 B04 and B07 only; remove component and final min-max normalization.',
          'score_formula':'STE(t)=0.9*P_raw(t)/P_raw(first)+0.1*S_raw(t)/S_raw(first)',
          'P_raw':'growth-mass-weighted mean of absolute uncentered paired potential increments',
          'S_raw':'saved raw balanced spatial transport cost for the same fixed branch population',
          'cohorts':{'B04':41,'B07':339},'time_axis':'Retained original 0.2-hpf interval midpoints, 10.1 through 17.9 hpf. Raw intervals start/end are exported.',
          'component_minmax':False,'final_minmax':False,'epsilon_added':False,'curve_smoothing_changed':False,
          'HVG_changed':False,'Welch':'one-sided greater; n=4 retained primary comparison, n=3 sensitivity; total 2n nodes ending at current score node',
          'annotations':'raw P<0.05 plus positive window mean contrast and positive adjacent score change; q values exported separately',
          'Bayesian':'Refitted on the new scores: local positive-contrast posterior for both n=3 and n=4, and full-curve zero/one continuous-slope-change posterior. Same priors and AR1 grid as previous package.',
          'limits':'P windows contain no later score node. Curve nodes are correlated model outputs; local Bayesian scaling and full-curve time posterior are retrospective. Midpoint labels do not make interval summaries available before their interval ends.',
          'other_figures_and_Word_modified':False}
(O/'analysis_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2))
(O/'audit/numerical_qa.json').write_text(json.dumps({'status':'PASS','curves':checks,'local_tests':len(w),
 'manual_Welch_max_error':max(manual),'future_score_nodes_in_Welch':False,
 'posterior_mass_sums':pd.DataFrame(timepost).groupby('series').posterior_probability.sum().to_dict(),
 'source_manifest':manifest},indent=2))
print(pd.DataFrame(summaries).to_string(index=False))
