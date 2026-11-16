"""Shared, tested statistical primitives for the final validation analysis."""
from pathlib import Path
import re, hashlib
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = Path.home() / 'metal_data' / 'features'
UUID = re.compile(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$')

def valid_artists(value):
    if not isinstance(value, str): return []
    return sorted({s.lower() for s in re.split(r'[|/,;\s]+', value.strip()) if UUID.fullmatch(s)})

def clusters(meta):
    """Connected components linked by album or ANY valid credited artist; stable IDs."""
    parent = list(range(len(meta)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(meta.itertuples(index=False)):
        ids = ['artist:'+s for s in valid_artists(row.artist_mbids)]
        album = str(row.release_group).lower()
        if UUID.fullmatch(album): ids.append('album:'+album)
        for key in ids:
            if key in seen: parent[find(i)] = find(seen[key])
            else: seen[key] = i
    groups = {}
    for i, rid in enumerate(meta.recordingmbid): groups.setdefault(find(i), []).append(str(rid))
    labels = {k: 'cc_'+min(v) for k,v in groups.items()}
    return np.array([labels[find(i)] for i in range(len(meta))])

def retained(meta):
    """Deterministic original quality rule, without changing the original split."""
    duplicate = meta.sort_values('row').duplicated('audio_md5', keep='first')
    drop_rows = meta.sort_values('row').loc[duplicate, 'row']
    return meta.essentia_version.isin(['2.1-beta1','2.1-beta2']) & ~meta.row.isin(drop_rows)

def fit_int(X):
    mappings = []
    for x in np.asarray(X).T:
        values, counts = np.unique(x, return_counts=True)
        ranks = np.cumsum(counts) - (counts-1)/2
        z = stats.norm.ppf((ranks-.375)/(len(x)+.25))
        mappings.append((values, z))
    return mappings

def apply_int(X, mappings):
    X=np.asarray(X)
    return np.column_stack([np.interp(X[:,j], v, z, left=z[0], right=z[-1]) for j,(v,z) in enumerate(mappings)])

def weights(meta):
    cells=meta.groupby(['subgenre_short','cluster']).size()
    m=meta.groupby('subgenre_short').cluster.nunique()
    return np.array([1/(cells.loc[(s,c)]*m.loc[s]) for s,c in zip(meta.subgenre_short,meta.cluster)])

def contrasts(k):
    return np.column_stack([np.eye(k-1), -np.ones(k-1)])

def wald(estimate, covariance, A):
    C=A@covariance@A.T; C=(C+C.T)/2
    rank=int(np.linalg.matrix_rank(C)); condition=float(np.linalg.cond(C))
    info={'rank':rank,'condition':condition,'df':len(A)}
    if not np.isfinite(C).all() or rank != len(A) or condition > 1e10 or np.linalg.eigvalsh(C).min() <= 0:
        return {'statistic':None,'p':None,**info,'status':'descriptive: unstable contrast covariance'}
    d=A@estimate; W=float(d@np.linalg.solve(C,d))
    return {'statistic':W,'p':float(stats.chi2.sf(W,len(A))),**info,'status':'inferential (asymptotic)'}

def cr1(X,y,w,cluster):
    X=np.asarray(X,float); y=np.asarray(y,float); w=np.asarray(w,float)
    n,k=X.shape; rank=int(np.linalg.matrix_rank(X*np.sqrt(w[:,None])))
    if rank != k: raise ValueError('Regression design is not full rank')
    gram=X.T@(w[:,None]*X)
    beta=np.linalg.solve(gram,X.T@(w*y)); residual=y-X@beta
    ids,inv=np.unique(cluster,return_inverse=True); M=len(ids)
    if M<=k or n<=k: raise ValueError('Insufficient clusters for regression covariance')
    score=np.zeros((M,k)); np.add.at(score,inv,X*(w*residual)[:,None])
    bread=np.linalg.inv(gram); correction=M/(M-1)*(n-1)/(n-k)
    cov=correction*bread@(score.T@score)@bread; cov=(cov+cov.T)/2
    assert np.isfinite(cov).all()
    return beta,cov,{'design_rank':rank,'parameters':k,'clusters':M,'CR1_factor':float(correction),'design_condition':float(np.linalg.cond(X*np.sqrt(w[:,None])))}

def holm(p):
    p=np.asarray(p,float); order=np.argsort(p); adjusted=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    out=np.empty_like(p); out[order]=adjusted; return out

class Dispersion:
    """Multiplicity-weight implementation of literal whole-cluster resampling.

    A cluster selected k times contributes k distinct occurrences to every cell.
    Cell weights are k/(n_cg * sum_c k), exactly as in expanded sampled data.
    Precomputed within-group Gram matrices avoid repeating high-dimensional distances.
    """
    def __init__(self,X,meta):
        self.ids,self.inv=np.unique(meta.cluster,return_inverse=True)
        self.groups=sorted(meta.subgenre_short.unique()); self.cells=[]
        for sg in self.groups:
            rows=np.flatnonzero(meta.subgenre_short.eq(sg)); Xi=np.asarray(X[rows],float)
            ci=self.inv[rows]; present,local=np.unique(ci,return_inverse=True)
            counts=np.bincount(local); means=np.zeros((len(present),Xi.shape[1])); np.add.at(means,local,Xi); means/=counts[:,None]
            self.cells.append((present,local,counts,np.einsum('ij,ij->i',Xi,Xi),Xi@means.T,means@means.T))
    def compute(self,multiplicity=None):
        if multiplicity is None:multiplicity=np.ones(len(self.ids))
        out=[]
        for present,local,counts,norm,cross,gram in self.cells:
            mult=np.asarray(multiplicity)[present]; total=mult.sum()
            if total==0:return None
            q=mult/total; rw=q[local]/counts[local]
            d2=norm-2*(cross@q)+q@gram@q
            if d2.min() < -1e-7:raise ValueError('Negative squared distance beyond roundoff')
            out.append(float(rw@np.sqrt(np.maximum(d2,0))))
        return np.array(out)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
