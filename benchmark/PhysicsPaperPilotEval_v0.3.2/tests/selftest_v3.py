#!/usr/bin/env python3
"""Self-test: gold answers and gold section citations should score perfectly.
Creates temporary predictions, runs evaluator, and checks expected perfect values.
This tests code consistency, not scientific validity of the candidate answer key.
"""
from __future__ import annotations
import csv, json, subprocess, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def main():
    sys.path.insert(0, str(ROOT/'scripts'))
    import evaluate_v3
    # Unit checks for evidence-group alternatives and invalid citations.
    g=[{'evidence_id':'E1','accepted_sections':['Abstract','III Method']}]
    result=evaluate_v3.evidence_section_score('Abstract; III Method', g)
    assert result[0:3] == (1,0,0), result
    result=evaluate_v3.evidence_section_score('Unrelated section', g)
    assert result[0:3] == (0,1,1), result
    # v0.3.1: the same heading cited as printed in the PDF (REVTeX letter, arabic
    # number, no number) matches the HTML-numbered gold label; a different title does not.
    g=[{'evidence_id':'E1','accepted_sections':['II.1 Graph creation']}]
    for cited in ['II.1 Graph creation','A. Graph creation','II.A Graph creation','Graph creation','Sec. II.A Graph creation']:
        result=evaluate_v3.evidence_section_score(cited, g)
        assert result[0:3] == (1,0,0), (cited, result)
    for cited in ['II. DATASET','A. Data preprocessing']:
        result=evaluate_v3.evidence_section_score(cited, g)
        assert result[0:3] == (0,1,1), (cited, result)
    g=[{'evidence_id':'E1','accepted_sections':['Introduction']}]
    for cited in ['1 Introduction','I. INTRODUCTION']:
        assert evaluate_v3.evidence_section_score(cited, g)[0:3] == (1,0,0), cited
    # v0.3.1: superscript exponents are read as powers of ten, also by the key-point normalizer.
    assert evaluate_v3.extract_numbers('up to ~10⁵× on GPU') == [100000], evaluate_v3.extract_numbers('up to ~10⁵× on GPU')
    assert evaluate_v3.extract_numbers('δ = 10⁻⁶') == [1e-06], evaluate_v3.extract_numbers('δ = 10⁻⁶')
    assert evaluate_v3.normalize_text('δ = 10⁻⁶') == evaluate_v3.normalize_text('delta = 10^{-6}') == 'delta 10 to minus 6'
    assert evaluate_v3.normalize_text('10⁵') == '10 to 5'
    key = read(ROOT/'02_private_answer_key'/'answer_key_private.csv')
    questions = read(ROOT/'01_questions_for_models'/'questions_public.csv')
    qmap = {q['question_id']: q for q in questions}
    pred=[]
    for k in key:
        groups=json.loads(k.get('gold_evidence_groups_json','[]'))
        citations=[]
        for g in groups:
            sections=g.get('accepted_sections',[])
            # v0.3.1: cite the bare title (no section number), as a reader of the PDF
            # may write it; it must still cover the group.
            if sections: citations.append(evaluate_v3.SECTION_NUMBER_RE.sub('', sections[0], count=1))
        gold=k['gold_answer']
        pred.append({
            'model':'SELFTEST_GOLD',
            'question_id':k['question_id'],
            'model_answer':gold,
            'mcq_choice':gold if qmap[k['question_id']]['question_type']=='multiple_choice' else '',
            'cited_evidence':'; '.join(citations)
        })
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); inp=td/'predictions.csv'; out=td/'out'
        with inp.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['model','question_id','model_answer','mcq_choice','cited_evidence'])
            w.writeheader(); w.writerows(pred)
        proc=subprocess.run([sys.executable,str(ROOT/'scripts'/'evaluate_v3.py'),'--input',str(inp),'--outdir',str(out)],capture_output=True,text=True)
        if proc.returncode:
            print(proc.stdout); print(proc.stderr,file=sys.stderr); raise SystemExit(proc.returncode)
        scored=read(out/'scored_answers.csv')
        bad=[]
        for r in scored:
            qtype=qmap[r['question_id']]['question_type']
            if qtype=='multiple_choice' and float(r['mcq_correct'])!=1: bad.append((r['question_id'],'mcq',r['mcq_correct']))
            if qtype in ('short_answer','open_ended') and float(r['keypoint_coverage'])!=1: bad.append((r['question_id'],'keypoints',r['keypoint_coverage'],r['keypoints_missing_ids']))
            if float(r['evidence_section_f1'])!=1: bad.append((r['question_id'],'evidence',r['evidence_section_f1'],r['evidence_sections_missing']))
        if bad:
            print('SELF-TEST FAILED')
            for x in bad: print(x)
            raise SystemExit(1)
    print('SELF-TEST PASSED: 45 gold-reference rows score 100% on MCQ accuracy (where applicable), key-point coverage (short/open), and evidence-section F1.')
    print('Note: this validates evaluator consistency only, not the scientific correctness of the candidate gold annotations.')

if __name__=='__main__': main()
