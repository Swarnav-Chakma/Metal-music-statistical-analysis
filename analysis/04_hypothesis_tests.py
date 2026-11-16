"""Final validation H1-H3; replaces invalid inference in the preserved Gemini version.
Run from any directory. No raw inputs are modified. See dated plan amendment.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from statistical_core import ROOT, DATA, weights, contrasts, wald, cr1, holm, Dispersion, sha

A=ROOT/'analysis'; SEED=20261027; B=1999

def main(reuse_bootstrap=False):
    tr=pd.read_csv(A/'03_meta_train.csv'); va=pd.read_csv(A/'03_meta_val.csv')
    rawmeta=pd.read_csv(DATA/'metadata.csv'); raw=np.load(DATA/'X.npy',mmap_mode='r')
    names=pd.read_csv(DATA/'feature_names.csv').name.tolist(); X=np.load(A/'03_X_val_R1.npy').astype(float)
    for m in [tr,va]:
        assert np.array_equal(rawmeta.iloc[m.row].recordingmbid,m.recordingmbid)
        assert np.array_equal(rawmeta.iloc[m.row].split,m.split)
    assert X.shape==(len(va),2234) and np.isfinite(X).all()
    groups=sorted(va.subgenre_short.unique()); k=len(groups); C=contrasts(k)
    w=weights(va); indicators=np.column_stack([va.subgenre_short.eq(g).to_numpy(float) for g in groups])
    support=va.groupby('subgenre_short').cluster.nunique().reindex(groups)
    sufficient=bool(support.min()>=20)
    counts={'retained':len(tr)+len(va),'train':len(tr),'validation':len(va),
            'clusters_retained':pd.concat([tr,va]).cluster.nunique(),'clusters_train':tr.cluster.nunique(),
            'clusters_validation':va.cluster.nunique(),'minimum_clusters_per_subgenre':int(support.min()),
            'validation_clusters_crossing_subgenres':int((va.groupby('cluster').subgenre_short.nunique()>1).sum()),
            'clusters_crossing_splits':len(set(tr.cluster)&set(va.cluster))}
    print('Population checks:',counts,flush=True)
    # H1: whole-cluster bootstrap. Multiplicities are equivalent to explicit occurrences.
    disp=Dispersion(X,va); D=disp.compute(); rng=np.random.default_rng(SEED)
    if reuse_bootstrap:
        # Output-only refresh: reuse saved draws only with identical numerical inputs/core.
        previous=json.loads((A/'04_summary.json').read_text())
        assert previous['H1']['B']==B and previous['H1']['seed']==SEED
        for path,digest in previous['inputs'].items():
            if path!='analysis/04_hypothesis_tests.py': assert sha(ROOT/path)==digest, 'Bootstrap input changed'
        with np.load(A/'04_inference_arrays.npz') as saved: boot=saved['H1_boot'].copy()
        attempts=previous['H1']['attempts']
    else:
        boot=[]; attempts=0
        while len(boot)<B and attempts<2*B:
            mult=np.bincount(rng.integers(0,len(disp.ids),size=len(disp.ids)),minlength=len(disp.ids))
            d=disp.compute(mult); attempts+=1
            if d is not None and np.isfinite(d).all() and (d>0).all():boot.append(d)
            if len(boot) and len(boot)%250==0:print('H1 bootstrap',len(boot),flush=True)
        boot=np.asarray(boot)
    if len(boot)!=B:raise RuntimeError('Insufficient successful bootstrap draws; no inference written')
    covlog=np.cov(np.log(boot),rowvar=False,ddof=1); h1=wald(np.log(D),covlog,C)
    if not sufficient:h1.update(statistic=None,p=None,status='descriptive: fewer than 20 clusters in a subgenre')
    relative=D/np.exp(np.log(D).mean()); rb=boot/np.exp(np.log(boot).mean(1))[:,None]
    ci=np.percentile(boot,[2.5,97.5],axis=0); rci=np.percentile(rb,[2.5,97.5],axis=0)
    ratio=float(D.max()/D.min()); ratio_ci=np.percentile(boot.max(1)/boot.min(1),[2.5,97.5])
    h1.update(ratio=ratio,ratio_ci=ratio_ci.tolist(),B=B,attempts=attempts,seed=SEED,
              beyond_SESOI=int((np.abs(relative-1)>.1).sum()),most=groups[int(D.argmax())],least=groups[int(D.argmin())])
    f1=pd.DataFrame({'subgenre':groups,'clusters':support.values,'D_g':D,'D_g_ci_lo':ci[0],'D_g_ci_hi':ci[1],
                     'r_g':relative,'r_g_ci_lo':rci[0],'r_g_ci_hi':rci[1],'beyond_SESOI':np.abs(relative-1)>.1})
    # H2: conventional CR1 WLS, whole-component scores.
    rv=np.asarray(raw[va.row],float); rt=np.asarray(raw[tr.row],float)
    bpm=rv[:,names.index('rhythm.bpm')]; spectral=rv[:,names.index('lowlevel.spectral_centroid.mean')]
    assert (bpm>0).all() and (spectral>0).all()
    median=float(np.median(rt[:,names.index('rhythm.bpm')]))
    x=np.log2(bpm)-np.log2(median); y=np.log(spectral)
    design=np.column_stack([indicators,indicators*x[:,None],va.essentia_version.eq('2.1-beta1')])
    beta,V,diagnostics2=cr1(design,y,w,va.cluster.to_numpy()); slopes=beta[k:2*k]; Vs=V[k:2*k,k:2*k]
    h2=wald(slopes,Vs,C); c=np.ones(k)/k; pool=float(c@slopes); poolse=float(np.sqrt(c@Vs@c)); poolci=[pool-1.96*poolse,pool+1.96*poolse]
    if not sufficient:h2.update(statistic=None,p=None,status='descriptive: insufficient cluster support')
    se=np.sqrt(np.diag(Vs)); h2.update(pooled_slope=pool,pooled_se=poolse,pooled_ci=poolci,
       training_bpm_median=median,beyond_SESOI=int((np.abs(slopes-pool)>.05).sum()),**diagnostics2)
    f2=pd.DataFrame({'subgenre':groups,'clusters':support.values,'slope_beta_g':slopes,'ci_lo':slopes-1.96*se,
       'ci_hi':slopes+1.96*se,'slope_contrast':slopes-pool,'beyond_SESOI':np.abs(slopes-pool)>.05,
       'pooled_slope':pool,'pooled_se':poolse,'pooled_ci_lo':poolci[0],'pooled_ci_hi':poolci[1]})
    # H3: marginal saturated linear probability model, no version adjustment.
    minor=rv[:,names.index('tonal.key_scale=minor')]; assert np.isin(minor,[0,1]).all()
    probs,Vp,diagnostics3=cr1(indicators,minor,w,va.cluster.to_numpy())
    direct=np.array([np.average(minor[indicators[:,j]==1],weights=w[indicators[:,j]==1]) for j in range(k)])
    assert np.allclose(probs,direct,atol=1e-12)
    h3=wald(probs,Vp,C)
    if not sufficient:h3.update(statistic=None,p=None,status='descriptive: insufficient cluster support')
    pse=np.sqrt(np.diag(Vp)); overall=float(probs.mean()); lo=probs-1.96*pse; hi=probs+1.96*pse
    h3.update(overall_minor=overall,min_share=float(probs.min()),max_share=float(probs.max()),
       range_pp=float(100*np.ptp(probs)),min_group=groups[int(probs.argmin())],max_group=groups[int(probs.argmax())],
       intervals_outside_unit=int(((lo<0)|(hi>1)).sum()),beyond_SESOI=int((np.abs(probs-overall)>.1).sum()),**diagnostics3)
    f3=pd.DataFrame({'subgenre':groups,'clusters':support.values,'minor_share':probs,'ci_lo':lo,'ci_hi':hi,
       'overall_minor':overall,'difference_pp':100*(probs-overall),'beyond_SESOI':np.abs(probs-overall)>.1})
    family=[h for h in [h2,h3] if h['p'] is not None]
    for h,p in zip(family,holm([h['p'] for h in family])):h['p_holm']=float(p)
    for h in [h2,h3]:h.setdefault('p_holm',None)
    h1['decision']='descriptive only' if h1['p'] is None else ('reject H0' if h1['p']<.05 else 'fail to reject H0')
    for h in [h2,h3]: h['decision']='descriptive only' if h['p_holm'] is None else ('reject H0' if h['p_holm']<.05 else 'fail to reject H0')
    h1['method']='whole-component bootstrap log-dispersion covariance; asymptotic chi-square Wald'
    h2['method']='WLS interaction; whole-component CR1; asymptotic chi-square Wald'
    h3['method']='exploratory saturated weighted LPM; whole-component CR1; asymptotic chi-square Wald'
    for frame,h in [(f1,h1),(f2,h2),(f3,h3)]:
        frame['wald_statistic']=h['statistic'];frame['df']=h['df'];frame['p_raw']=h['p'];frame['status']=h['status']
        frame['p_holm']=h.get('p_holm',np.nan)
        frame['decision']=h['decision'];frame['method']=h['method'];frame['alpha']=.05
    for frame,name in [(f1,'H1'),(f2,'H2'),(f3,'H3')]:frame.to_csv(A/f'04_{name}_results.csv',index=False)
    np.savez(A/'04_inference_arrays.npz',H1_boot=boot,H1_covlog=covlog,H1_D=D,H2_beta=beta,H2_cov=V,H3_probs=probs,H3_cov=Vp)
    result={'counts':counts,'H1':h1,'H2':h2,'H3':h3,'secondary_family':[name for name,h in [('H2',h2),('H3 exploratory',h3)] if h['p'] is not None],
       'weighting':'1/(n_cg*m_g); components linked by all valid artists or album; whole-cluster covariance/resampling',
       'validity':'At least 20 represented clusters per subgenre; full-rank finite contrast covariance; condition <=1e10. These screens do not prove asymptotic calibration.',
       'inputs':{str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha(p) for p in [A/'03_X_val_R1.npy',A/'03_meta_train.csv',A/'03_meta_val.csv',DATA/'X.npy',DATA/'metadata.csv',A/'04_hypothesis_tests.py',A/'statistical_core.py']}}
    (A/'04_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    text=['# Final validation results — 27 October 2026','',json.dumps(counts,indent=2),'',
      'Weights: 1/(n_cg m_g), equal component weight within each subgenre. Components use valid artist/album links and retain mixed-subgenre observations together.',
      'H1 is primary at 0.05; H2 and exploratory H3 use Holm (sorted cumulative maximum). Omnibus alternatives are non-directional; test statistics use the upper tail.',
      'All confidence intervals are pointwise. SESOI values are project-defined analytical benchmarks, not validated perceptual thresholds.']
    descriptions=[('H1',h1,f1,'H0: D_1=...=D_17. Whole-cluster bootstrap covariance of log mean distances; chi-square(16) Wald calibration. No small-sample dispersion multiplier or ridge. Raw-label permutation would not isolate dispersion from centroid differences; PERMANOVA tests location. Bootstrap assumes independent components and adequate effective sample size; training transforms are treated as fixed. Ratio extrema are selected and descriptive.'),
      ('H2',h2,f2,'H0: beta_1=...=beta_17. WLS subgenre intercepts/slopes plus extractor-version indicator; CR1 factor M/(M-1)*(N-1)/(N-K). Whole-cluster score covariance, chi-square(16) Wald. Pooled means equal-subgenre average, not an imposed common slope. Normal 95% CIs. IID regression ignores repeated recordings; mixed models require a specified random-effect model; Spearman answers a different question. Requires useful log-linear specification and enough independent clusters.'),
      ('H3 exploratory',h3,f3,'H0: p_1=...=p_17. Saturated weighted linear probability model, whole-cluster CR1 covariance, chi-square(16) Wald. Normal 95% CIs are not clipped. Marginal association, not version-adjusted. Cluster-label permutation is inappropriate for mixed-subgenre components; track-level chi-square ignores dependence. Estimated mode is imperfect. Exploratory because validation mode counts were inspected during design.')]
    for name,h,frame,desc in descriptions:
        text += ['',f'## {name}',desc,'',json.dumps(h,indent=2),'',frame.to_csv(index=False)]
    text += ['','## Limitations','Artist overlap between training and validation remains; no split was changed. Unobserved artist links can leave residual dependence. CR1 and Wald calibration are asymptotic, especially uncertain for subgenres with fewer clusters. Pointwise CIs and counts beyond SESOI do not constitute simultaneous post-hoc testing. No unrun sensitivity analyses are claimed. Bootstrap Monte Carlo precision is finite. Failure to reject is not evidence of equality; dispersion is descriptor variation, not creativity.']
    (A/'04_results.md').write_text('\n'.join(text))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-bootstrap',action='store_true',help='Reuse saved H1 draws after verifying unchanged numerical inputs; refresh all summaries/regression estimates')
    main(parser.parse_args().reuse_bootstrap)
