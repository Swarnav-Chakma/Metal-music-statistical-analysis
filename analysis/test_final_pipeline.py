"""Targeted final-method checks; no inference bootstrap rerun required."""
import unittest,json
import numpy as np
import pandas as pd
from scipy import stats
from statistical_core import *
A=ROOT/'analysis'

class FinalChecks(unittest.TestCase):
    def test_average_rank_frozen_map(self):
        train=np.array([0,0,0,1,2,2,4.])[:,None]; maps=fit_int(train)
        expected=stats.norm.ppf((stats.rankdata(train[:,0])-.375)/(len(train)+.25))
        np.testing.assert_allclose(apply_int(train,maps)[:,0],expected)
        before=[(v.copy(),z.copy()) for v,z in maps]
        vals=np.array([-1,0,.5,1,2,3,4,100.])[:,None]; z=apply_int(vals,maps)
        self.assertTrue(np.isfinite(z).all());self.assertTrue((np.diff(z[:,0])>=0).all())
        apply_int(np.array([-1e20,1e20])[:,None],maps)
        np.testing.assert_array_equal(apply_int(vals,maps),z)
        for (v,z),(v0,z0) in zip(maps,before):np.testing.assert_array_equal(v,v0);np.testing.assert_array_equal(z,z0)
    def test_audit_tie_example(self):
        tr=pd.read_csv(A/'03_meta_train.csv'); names=pd.read_csv(DATA/'feature_names.csv').name.tolist()
        raw=np.load(DATA/'X.npy',mmap_mode='r'); x=np.array(raw[tr.row,names.index('lowlevel.silence_rate_30dB.min')],float)
        v,n=np.unique(x,return_counts=True); dominant=v[n.argmax()]; self.assertEqual(int(n.max()),1906)
        expected=stats.norm.ppf((stats.rankdata(x)[x==dominant][0]-.375)/(len(x)+.25))
        self.assertAlmostEqual(expected,-.5042617189793955)
        np.testing.assert_allclose(apply_int(x[:,None],fit_int(x[:,None]))[x==dominant,0],expected)
    def test_transitive_artist_album_components(self):
        ids=['00000000-0000-0000-0000-'+str(i).zfill(12) for i in range(8)]
        m=pd.DataFrame({'recordingmbid':['a','b','c','d'],'artist_mbids':[ids[0],ids[0]+'|'+ids[1],ids[1],'malformed'], 'release_group':[ids[2],ids[3],ids[4],ids[4]]})
        self.assertEqual(len(set(clusters(m))),1)
        shuffled=m.sample(frac=1,random_state=1); self.assertEqual(set(clusters(shuffled)),set(clusters(m)))
        self.assertEqual(valid_artists('usicBrainz Artist Id'),[])
    def test_whole_cluster_bootstrap(self):
        m=pd.DataFrame({'cluster':['a','a','b','b','c','c'],'subgenre_short':['x','y','x','x','y','y']})
        X=np.array([[1,2],[8,4],[3,4],[5,6],[4,1],[7,2]],float); engine=Dispersion(X,m)
        # 'a' occurs twice, carrying both subgenres on each occurrence.
        draw=['a','b','a','c']; parts=[]; xx=[]
        for occurrence,cl in enumerate(draw):
            rows=np.flatnonzero(m.cluster.eq(cl)); d=m.iloc[rows].copy(); d['cluster']=str(occurrence);parts.append(d);xx.extend(X[rows])
        expanded=pd.concat(parts,ignore_index=True); xx=np.array(xx); w=weights(expanded); direct=[]
        for g in engine.groups:
            rows=np.flatnonzero(expanded.subgenre_short.eq(g)); center=w[rows]@xx[rows];direct.append(w[rows]@np.linalg.norm(xx[rows]-center,axis=1))
        mult=np.array([draw.count(c) for c in engine.ids]);np.testing.assert_allclose(engine.compute(mult),direct,atol=1e-12)
    def test_wald_covariance(self):
        b=np.array([1,2,4.]);V=np.diag([.2,.4,.8]);C=contrasts(3)
        expected=(C@b)@np.linalg.solve(C@V@C.T,C@b)
        self.assertAlmostEqual(wald(b,V,C)['statistic'],expected)
        self.assertIsNone(wald(b,np.zeros((3,3)),C)['p'])
    def test_holm(self):
        np.testing.assert_allclose(holm([.04,.03]),[.06,.06])
        np.testing.assert_allclose(holm([.9,.01,.03]),[.9,.03,.06])
        from statsmodels.stats.multitest import multipletests
        p=np.array([.04,.03,.001,.9]);np.testing.assert_allclose(holm(p),multipletests(p,method='holm')[1])
    def test_saved_alignment_mapping_and_provenance(self):
        rawmeta=pd.read_csv(DATA/'metadata.csv');tr=pd.read_csv(A/'03_meta_train.csv');va=pd.read_csv(A/'03_meta_val.csv');m=pd.concat([tr,va]).sort_values('row')
        expected=rawmeta[retained(rawmeta)]
        np.testing.assert_array_equal(m.recordingmbid,expected.recordingmbid)
        np.testing.assert_array_equal(m.cluster,clusters(m))
        p=np.load(A/'03_prep_params.npz');off=p['int_offsets'];values=p['int_values'];scores=p['int_scores'];maps=[(values[off[j]:off[j+1]],scores[off[j]:off[j+1]]) for j in range(len(off)-1)]
        raw=np.load(DATA/'X.npy',mmap_mode='r')
        for meta,suffix in [(tr,'train'),(va,'val')]:
            np.testing.assert_array_equal(rawmeta.iloc[meta.row].recordingmbid,meta.recordingmbid)
            for start in range(0,len(meta),128):
                batch=meta.iloc[start:start+128]
                rawbatch=raw[batch.row][:,p['cols_H1']].astype(float)
                xt=apply_int(rawbatch,maps); xt[batch.is_beta1.eq(1)]-=p['beta1_coef'];std=(xt-p['col_mean'])/p['col_std']
                for r,expected in [('R1',std*p['weights']),('R2',std)]:
                    saved=np.load(A/f'03_X_{suffix}_{r}.npy',mmap_mode='r')[start:start+128]
                    np.testing.assert_array_equal(saved,expected.astype(np.float32));self.assertTrue(np.isfinite(saved).all())
                robust=np.clip((rawbatch-p['col_med'])/p['iqr'],-10,10);robust[batch.is_beta1.eq(1)]-=p['beta1_coef_r4'];robust=(robust-p['mean_r4'])/p['std_r4']
                saved=np.load(A/f'03_X_{suffix}_R4.npy',mmap_mode='r')[start:start+128]
                np.testing.assert_array_equal(saved,robust.astype(np.float32))
        summary=json.loads((A/'04_summary.json').read_text())
        for path,digest in summary['inputs'].items():self.assertEqual(sha(ROOT/path),digest)
        self.assertEqual(sha(DATA/'X.npy'),json.loads((DATA/'extraction_report.json').read_text())['X_sha256'])
    def test_regressions_against_independent_implementation(self):
        import statsmodels.api as sm
        from statsmodels.stats.sandwich_covariance import cov_cluster
        va=pd.read_csv(A/'03_meta_val.csv');tr=pd.read_csv(A/'03_meta_train.csv');names=pd.read_csv(DATA/'feature_names.csv').name.tolist();raw=np.load(DATA/'X.npy',mmap_mode='r');rv=raw[va.row].astype(float)
        groups=sorted(va.subgenre_short.unique());k=len(groups);I=np.column_stack([va.subgenre_short.eq(g).to_numpy(float) for g in groups]);w=weights(va)
        x=np.log2(rv[:,names.index('rhythm.bpm')])-np.log2(float(np.median(raw[tr.row,names.index('rhythm.bpm')].astype(float))))
        design=np.column_stack([I,I*x[:,None],va.essentia_version.eq('2.1-beta1')]);y=np.log(rv[:,names.index('lowlevel.spectral_centroid.mean')]);arrays=np.load(A/'04_inference_arrays.npz');s=json.loads((A/'04_summary.json').read_text())
        fit=sm.WLS(y,design,weights=w).fit();cov=cov_cluster(fit,va.cluster,use_correction=True)
        np.testing.assert_allclose(arrays['H2_beta'],fit.params,atol=1e-11);np.testing.assert_allclose(arrays['H2_cov'],cov,atol=1e-11)
        c=np.r_[np.zeros(k),np.ones(k)/k,0];pool=float(c@fit.params);se=float(np.sqrt(c@cov@c))
        np.testing.assert_allclose(s['H2']['pooled_ci'],[pool-1.96*se,pool+1.96*se],atol=1e-11)
        minor=rv[:,names.index('tonal.key_scale=minor')];fit3=sm.WLS(minor,I,weights=w).fit()
        np.testing.assert_allclose(arrays['H3_probs'],fit3.params,atol=1e-12);np.testing.assert_allclose(arrays['H3_cov'],cov_cluster(fit3,va.cluster),atol=1e-12)
        for j,g in enumerate(groups):self.assertAlmostEqual(fit3.params[j],np.average(minor[I[:,j]==1],weights=w[I[:,j]==1]))
    def test_saved_result_numerics(self):
        s=json.loads((A/'04_summary.json').read_text());p=np.load(A/'04_inference_arrays.npz');C=contrasts(17)
        for name,b,V in [('H1',np.log(p['H1_D']),p['H1_covlog']),('H2',p['H2_beta'][17:34],p['H2_cov'][17:34,17:34]),('H3',p['H3_probs'],p['H3_cov'])]:
            result=wald(b,V,C);self.assertAlmostEqual(result['statistic'],s[name]['statistic']);f=pd.read_csv(A/f'04_{name}_results.csv');np.testing.assert_allclose(f.p_raw,s[name]['p']);np.testing.assert_allclose(f.wald_statistic,result['statistic'])
        np.testing.assert_allclose(pd.read_csv(A/'04_H1_results.csv').D_g_ci_lo,np.percentile(p['H1_boot'],2.5,axis=0))
        np.testing.assert_allclose(pd.read_csv(A/'04_H2_results.csv').pooled_ci_lo,s['H2']['pooled_ci'][0])
        np.testing.assert_allclose(holm([s['H2']['p'],s['H3']['p']]),[s['H2']['p_holm'],s['H3']['p_holm']])
        m=pd.read_csv(A/'03_meta_val.csv');D=Dispersion(np.load(A/'03_X_val_R1.npy'),m).compute();np.testing.assert_allclose(D,p['H1_D'])
        self.assertEqual(p['H1_boot'].shape,(1999,17))
        md=(A/'04_results.md').read_text();self.assertIn(str(s['H2']['pooled_ci'][0]),md)

if __name__=='__main__':unittest.main(verbosity=2)
