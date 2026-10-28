"""Matched frequentist and Bayesian local contrasts of saved reconstruction scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats, special, optimize

OUT=Path(__file__).resolve().parents[1]
RHOS=np.array([0.,.25,.5,.75,.9])
W=4

def bh(p):
    p=np.asarray(p,float); order=np.argsort(p); q=np.empty(len(p))
    q[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1])
    return q

def component(y, rho, delta_prior_variance=4.):
    n=len(y); X=np.column_stack([np.ones(n),np.r_[np.zeros(n//2),np.ones(n//2)]])
    R=rho**np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])
    Ri=np.linalg.inv(R)
    V0=np.diag([25.,delta_prior_variance]); P0=np.linalg.inv(V0)
    P=P0+X.T@Ri@X; V=np.linalg.inv(P); m=V@X.T@Ri@y
    a0,b0=2.,1.;a=a0+n/2;b=b0+.5*(y@Ri@y-m@P@m)
    assert b>0
    logz=-n/2*np.log(2*np.pi)-.5*np.linalg.slogdet(R)[1]+.5*(np.linalg.slogdet(V)[1]-np.linalg.slogdet(V0)[1])+a0*np.log(b0)-a*np.log(b)+special.gammaln(a)-special.gammaln(a0)
    scale=np.sqrt(b/a*V[1,1]);df=2*a
    return {'rho':float(rho),'mean':float(m[1]),'scale':float(scale),'df':float(df),'logz':float(logz),'prob':float(stats.t.sf(0,df,loc=m[1],scale=scale))}

def posterior(y, prior_var=4., ci=True):
    cs=[component(y,rho,prior_var) for rho in RHOS]
    weights=special.softmax([c['logz'] for c in cs])
    p=float(sum(w*c['prob'] for w,c in zip(weights,cs)))
    result={'posterior_prob_increase':p,'posterior_prob_increase_independent':cs[0]['prob'],'posterior_mean_delta_standardized':sum(w*c['mean'] for w,c in zip(weights,cs)), 'posterior_mean_rho':float(weights@RHOS)}
    if ci:
        def cdf(v):return sum(w*stats.t.cdf(v,c['df'],loc=c['mean'],scale=c['scale']) for w,c in zip(weights,cs))
        result['credible_low_standardized']=optimize.brentq(lambda x:cdf(x)-.025,-100,100)
        result['credible_high_standardized']=optimize.brentq(lambda x:cdf(x)-.975,-100,100)
    return result,cs,weights


"""Full-curve Bayesian evidence for zero or one continuous slope change."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import gammaln,logsumexp

O=Path(__file__).resolve().parents[1]
RHOS=np.array([0.,.25,.5,.75,.9])

def log_evidence(y,X,Ri,logdetR,prior_variance):
    n=len(y);v=np.asarray(prior_variance,float)
    precision=np.diag(1/v)+X.T@Ri@X
    m=np.linalg.solve(precision,X.T@Ri@y)
    a=2+n/2;b=1+.5*(y@Ri@y-m@precision@m)
    assert b>0
    return -n/2*np.log(2*np.pi)-.5*logdetR-.5*np.linalg.slogdet(precision)[1]-.5*np.log(v).sum()-a*np.log(b)+gammaln(a)-gammaln(2.)

def analyze(t,s,hinge_variance=4.):
    n=len(s);u=(t-(t.min()+t.max())/2)/((t.max()-t.min())/2)
    y=(s-s.mean())/s.std(ddof=1)
    candidates=np.arange(4,n-4)
    X0=np.column_stack([np.ones(n),u])
    Xs=[X0]+[np.column_stack([X0,np.maximum(0,u-u[j])]) for j in candidates]
    z=[]
    for rho in RHOS:
        R=rho**np.abs(np.arange(n)[:,None]-np.arange(n)[None,:]);Ri=np.linalg.inv(R)
        ld=np.linalg.slogdet(R)[1]
        z.append([log_evidence(y,X,Ri,ld,[25,4] if k==0 else [25,4,hinge_variance]) for k,X in enumerate(Xs)])
    logz=logsumexp(z,axis=0)-np.log(len(RHOS))
    prior=np.r_[.5,np.full(len(candidates),.5/len(candidates))]
    logpost=logz+np.log(prior);post=np.exp(logpost-logsumexp(logpost))
    conditional=post[1:]/post[1:].sum()
    assert abs(post.sum()-1)<1e-12
    return candidates,post,conditional,logz

