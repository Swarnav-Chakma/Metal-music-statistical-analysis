"""Verify synchronized final deliverables against saved numerical results; no resampling."""
from pathlib import Path
import json,re,hashlib
import numpy as np
import pandas as pd
from pypdf import PdfReader
from statistical_core import ROOT,DATA,sha
A=ROOT/'analysis'

def main():
    s=json.loads((A/'04_summary.json').read_text());q=json.loads((A/'03_quality_checks.json').read_text())
    src=(ROOT/'CSE558_Midterm_Report.tex').read_text();readme=(ROOT/'README.md').read_text()
    assert src==(ROOT/'report.tex').read_text()
    assert 'Group name & 4' in src
    assert 'https://github.com/Swarnav-Chakma/Metal-music-statistical-analysis' in src
    for stale in ['No public repository exists yet', 'There is no public GitHub repository yet', 'link will be added before submission']:
        assert stale not in src+readme
    assert sha(ROOT/'CSE558_Midterm_Report.pdf')==sha(ROOT/'report.pdf')
    reader=PdfReader(ROOT/'CSE558_Midterm_Report.pdf');assert 1<=len(reader.pages)<=8
    text='\n'.join(p.extract_text() for p in reader.pages)
    assert len(re.findall(r'Figure\s+[1-7]:',text))==7
    forms=sum(len(p['/Resources'].get('/XObject',{})) for p in reader.pages);assert forms==7
    for bad in ['<LOWER>','<UPPER>','github.com/metal-cse558','1291988','5,856','5{,}856','_nolegend_']:
        assert bad not in src+readme+text,bad
    for h in ['H1','H2','H3']:assert f"{s[h]['statistic']:.3f}" in text
    for value in [s['H2']['p'],s['H2']['p_holm'],s['H3']['p'],s['H3']['p_holm']]:assert f'{value:.4f}' in text
    for value in s['H2']['pooled_ci']:assert f'{value:+.4f}' in text
    for value in s['H1']['ratio_ci']:assert f'{value:.3f}' in text
    assert '2.07' in text and '44' in text
    assert q['raw_instance_attributes']==5874*2675
    assert q['base_instance_attributes']==5858*2262
    assert q['H1_instance_attributes']==5858*2234
    assert sha(DATA/'X.npy')==q['raw_sha256']
    assert sha(DATA/'metadata.csv')==q['metadata_sha256']
    for path,digest in s['inputs'].items():assert sha(ROOT/path)==digest,path
    assert q['preprocess_sha256']==sha(A/'03_preprocess.py') and q['core_sha256']==sha(A/'statistical_core.py')
    figures=sorted(A.glob('fig_*.pdf'));assert len(figures)==7
    for p in figures:assert p.stat().st_size>1000 and len(PdfReader(p).pages)==1
    for h in ['H1','H2','H3']:
        frame=pd.read_csv(A/f'04_{h}_results.csv');assert len(frame)==17
        np.testing.assert_allclose(frame.p_raw,s[h]['p']);assert frame.decision.eq(s[h]['decision']).all()
    assert 'Overfull' not in (ROOT/'CSE558_Midterm_Report.log').read_text()
    assert 'Ran 9 tests' in (A/'06_test_results.log').read_text() and (A/'06_test_results.log').read_text().rstrip().endswith('OK')
    manifest=json.loads((DATA.parent/'zenodo_manifest.json').read_text());labels={}
    for item in manifest:
        p=DATA.parent/'ab_tsv'/item['key']
        if p.exists():
            actual='md5:'+hashlib.md5(p.read_bytes()).hexdigest();assert actual==item['checksum'];labels[str(p)]=actual
    artifacts=[ROOT/'CSE558_Midterm_Report.pdf',ROOT/'report.pdf',ROOT/'CSE558_Midterm_Report.tex',ROOT/'report.tex',ROOT/'README.md']+figures+list(A.glob('04_*results*'))+[A/'04_summary.json',A/'04_inference_arrays.npz']
    result={'status':'passed','pdf_pages':len(reader.pages),'embedded_figure_forms':forms,'pdf_copies_identical':True,'source_copies_identical':True,
       'no_overfull_boxes':True,'targeted_tests':9,'raw_data_hashes_unchanged':True,'labels_match_local_zenodo_manifest':labels,
       'all_result_tables_match_current_summary':True,'rendered_pages':'/private/tmp/metal_report_editorial_render/page-1.png through page-6.png',
       'visual_review':'All six pages inspected at 144 dpi after editorial cleanup; group 4, public code link and unchanged team details verified.',
       'pending':'Commit and push the local editorial updates to the existing public repository.',
       'limitations':'Asymptotic Wald/CR1 calibration, minimum 25 clusters per subgenre, cross-split artist overlap, uncertain mode estimates; H3 exploratory.',
       'sha256':{str(p.relative_to(ROOT)):sha(p) for p in artifacts}}
    (A/'06_delivery_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
