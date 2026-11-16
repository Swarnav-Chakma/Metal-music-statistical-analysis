"""Population/design summaries, refreshed after the dated final-validation amendment.
Raw summaries are distinct from retained training and validation inference populations.
"""
import json
import numpy as np
import pandas as pd
from statistical_core import ROOT,DATA,clusters,valid_artists,retained
A=ROOT/'analysis'

def summary(m):
    return {'recordings':len(m),'clusters':m.cluster.nunique(),
      'clusters_crossing_subgenres':int((m.groupby('cluster').subgenre_short.nunique()>1).sum()),
      'clusters_crossing_splits':int((m.groupby('cluster').split.nunique()>1).sum()),
      'largest_component':int(m.groupby('cluster').size().max())}

def main():
    raw=pd.read_csv(DATA/'metadata.csv'); raw['subgenre_short']=raw.subgenre.str.replace(r'^.*---','',regex=True); raw['cluster']=clusters(raw)
    tr=pd.read_csv(A/'03_meta_train.csv'); va=pd.read_csv(A/'03_meta_val.csv'); kept=pd.concat([tr,va]).sort_values('row').reset_index(drop=True)
    assert np.array_equal(kept.cluster,clusters(kept))
    populations={'raw_extracted':raw,'retained':kept,'training':tr,'validation_inference':va}
    out={'populations':{n:summary(m) for n,m in populations.items()},'definition':'Connected components sharing any valid credited artist MBID or album release-group ID; malformed artist strings create no artist links.'}
    rows=[]
    for name,m in populations.items():
        for sg,g in m.groupby('subgenre_short'):
            sizes=g.groupby('cluster').size(); rows.append({'population':name,'subgenre':sg,'recordings':len(g),'clusters':len(sizes),'largest_cluster_cell':int(sizes.max())})
    pd.DataFrame(rows).to_csv(A/'02_clusters_by_subgenre.csv',index=False)
    largest=[]
    for cl in kept.cluster.value_counts().head(10).index:
        m=kept[kept.cluster.eq(cl)]
        largest.append({'cluster':cl,'recordings':len(m),'validation_recordings':int(m.split.eq('validation').sum()),
          'subgenres':sorted(m.subgenre_short.unique()),'artists':sorted(set(a for v in m.artist_mbids for a in valid_artists(v))),
          'albums':sorted(m.release_group.unique()),'recording_ids':m.recordingmbid.tolist()})
    out['largest_components_identifier_evidence']=largest
    out['artist_identifier_quality']={'nonempty_but_no_valid_artist':int(kept.artist_mbids.notna().to_numpy().dot(np.array([len(valid_artists(v))==0 for v in kept.artist_mbids],int))),
      'multiple_valid_artists':int(sum(len(valid_artists(v))>1 for v in kept.artist_mbids))}
    X=np.load(DATA/'X.npy',mmap_mode='r'); names=pd.read_csv(DATA/'feature_names.csv').name.tolist()
    mode=X[:,names.index('tonal.key_scale=minor')]; chords=X[:,names.index('tonal.chords_scale=minor')]
    out['estimated_mode_agreement']={'raw':float(np.mean(mode==chords)),'retained':float(np.mean(mode[kept.row]==chords[kept.row]))}
    manifest=json.loads((DATA.parent/'zenodo_manifest.json').read_text())
    out['provenance']={'local_manifest':str(DATA.parent/'zenodo_manifest.json'),'record_ids':sorted(set(e['links']['self'].split('/records/')[1].split('/')[0] for e in manifest)),
       'extractor_archives':sorted(raw.archive.unique()),'cohort_rows':len(pd.read_csv(DATA.parent/'cohort.csv')),
       'scope':'Extracted shard subset of the local cohort, not the entire benchmark. Manifest supports record 2553414; no independent web verification.'}
    (A/'02_design_summaries.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k!='largest_components_identifier_evidence'},indent=2))
if __name__=='__main__':main()
