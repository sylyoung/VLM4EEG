import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from raicl import binary_metrics, evaluated, image_path, key_of, load_index, parse_decision, prompt_from_paths, select_examples
from raicl.paths import DEFAULT_PROMPT, PROMPTS, TRAINING_MANIFEST

ROOT = Path(__file__).resolve().parents[1]


def item(subject, label, trial, value):
    vector = np.zeros(512, dtype=np.float16)
    vector[:3] = value
    return {"path": "data/CHSZ/time-chan/Vision/%s/class_%d/trial_%d_Vision.png" % (subject,label,trial), "label": label, "trial": trial, "key": (subject,label,trial), "vec": vector}


@pytest.fixture
def index():
    return {
        "sub_00": [item("sub_00",0,10,[1,1,0]),item("sub_00",0,20,[1,0.9,0]),item("sub_00",1,30,[0,1,1]),item("sub_00",0,50,[1,0,0])],
        "sub_01": [item("sub_01",0,10,[1,0,0]),item("sub_01",1,11,[0,1,0.9]),item("sub_01",1,12,[0,1,1])],
        "sub_02": [item("sub_02",0,10,[1,0,0]),item("sub_02",1,11,[0,1,0.8])],
        "sub_03": [item("sub_03",0,10,[1,0,0]),item("sub_03",1,11,[0,0.8,1])],
    }


@pytest.mark.parametrize("text,expected", [("FINAL DECISION: SEIZURE",1),("FINAL DECISION: NON-SEIZURE",0),("**FINAL DECISION:** **SEIZURE**",1),("final decision: non-seizure",0),("FINAL DECISION: SEIZURE\nFINAL DECISION: NON-SEIZURE",0),("I think this is seizure",-1),("FINAL DECISION: UNKNOWN",-1),("",-1)])
def test_final_decision(text, expected):
    assert parse_decision(text) == expected


@pytest.mark.parametrize("dataset,label,trial,expected", [("CHSZ",1,11,True),("CHSZ",0,11,False),("CHSZ",0,20,True),("NICU",1,11,False),("NICU",0,20,True)])
def test_evaluation_list(dataset,label,trial,expected):
    assert evaluated(label,trial,dataset) == expected


def test_unknown_dataset_fails():
    with pytest.raises(ValueError): evaluated(0,10,"wrong")


def test_render_paths_share_key():
    vision = "data/CHSZ/time-chan/Vision/sub_00/class_1/trial_30_Vision.png"
    vlm = image_path(vision)
    assert vlm.endswith("/VLM/sub_00/class_1/trial_30_VLM.png")
    assert key_of(vision) == key_of(vlm)
    assert image_path(vlm,"Vision") == vision


def test_retrieval_pool_rules(index):
    ns,sz,fallback = select_examples(index,"sub_00",image_path(index['sub_00'][2]['path']))
    assert not fallback
    assert len(ns)==len(sz)==2
    assert all(r['key'][0]=='sub_00' and r['label']==0 and r['trial']<30 for r in ns)
    assert len({r['key'][0] for r in sz})==2
    assert all(r['key'][0]!='sub_00' and r['label']==1 for r in sz)


def test_held_out_patient_excluded_from_training_references(index):
    _,sz,_ = select_examples(index,"sub_00",image_path(index['sub_00'][2]['path']),excluded=('sub_01',))
    assert {r['key'][0] for r in sz} == {'sub_02','sub_03'}


def test_later_anchor_exception_is_explicit(index):
    ns,_,fallback = select_examples(index,"sub_00",image_path(index['sub_00'][0]['path']))
    assert fallback and all(r['key'] != ('sub_00',0,10) for r in ns)
    assert all(r['trial']>10 for r in ns)


def test_missing_query_does_not_fall_back(index):
    with pytest.raises(ValueError): select_examples(index,"sub_00","data/VLM/sub_00/class_1/trial_99_VLM.png")


def test_missing_seizure_patients_does_not_duplicate(index):
    with pytest.raises(ValueError): select_examples(index,"sub_00",image_path(index['sub_00'][2]['path']),excluded=('sub_01','sub_02'))


def test_prompt_slot_order(index):
    ns,sz,_=select_examples(index,'sub_00',index['sub_00'][2]['path'])
    refs=[r['path'] for r in ns+sz]
    prompt,images=prompt_from_paths(index['sub_00'][2]['path'],refs,DEFAULT_PROMPT)
    assert images == [image_path(refs[3]),image_path(refs[2]),image_path(refs[0]),image_path(refs[1]),image_path(index['sub_00'][2]['path'])]
    assert prompt.count('[IMG: ')==5


def test_base_has_only_query(index):
    _,images=prompt_from_paths(index['sub_00'][2]['path'],[],PROMPTS/'binary/base.txt')
    assert len(images)==1


def test_insufficient_reference_slots_fail(index):
    with pytest.raises(ValueError):prompt_from_paths(index['sub_00'][2]['path'],['one'],DEFAULT_PROMPT)


def test_no_decision_is_scored_non_seizure():
    rows=[{'subject_id':'sub_00','true_label':1,'pred_label':-1,'error':'parse-fail'},{'subject_id':'sub_00','true_label':0,'pred_label':-1,'error':'parse-fail'}]
    scores,_=binary_metrics(rows)
    assert scores['BCA']==50 and scores['Sensitivity']==0 and scores['Specificity']==100
    assert scores['Macro F1']==pytest.approx(100/3)


def test_failed_requests_are_not_measurements():
    with pytest.raises(ValueError):binary_metrics([{'subject_id':'sub_00','true_label':1,'pred_label':-1,'error':'NETWORK_ERROR'}])


def test_averaging_is_per_patient_not_pooled():
    rows=[{'subject_id':'sub_00','true_label':1,'pred_label':1},{'subject_id':'sub_00','true_label':0,'pred_label':0}]
    rows += [{'subject_id':'sub_01','true_label':1,'pred_label':0} for _ in range(10)]
    rows += [{'subject_id':'sub_01','true_label':0,'pred_label':1} for _ in range(10)]
    scores,_=binary_metrics(rows)
    assert scores['BCA']==50


def test_manifest_has_exact_balanced_auxiliary_queries():
    folds=json.loads(TRAINING_MANIFEST.read_text())
    assert len(folds)==27
    for held_out,paths in folds.items():
        assert len(paths)==len(set(paths))
        labels=[int(p.split('/class_')[1][0]) for p in paths]
        assert labels.count(0)==labels.count(1)
        assert all(p.split('/')[0]!=held_out for p in paths)
        assert all(int(p.split('/trial_')[1])%10 != 0 for p in paths if '/class_0/' in p)


def test_only_two_authorized_workbooks_are_released():
    assert {p.name for p in (ROOT/'results').glob('*.xlsx')}=={'CHSZ_Gemini-3-Flash_RAICL.xlsx','CHSZ_Qwen3.5-9B_RAICL-finetuned.xlsx'}


def test_vlm_render_uses_paper_clipping_fraction():
    import matplotlib.pyplot as plt
    from raicl.workflows.render import EEGProcessor
    signal = np.arange(1000, dtype=float).reshape(1, -1)
    figure, axis = plt.subplots()
    EEGProcessor.plot_vlm(axis, signal, 250, 1.0, plt.get_cmap('tab20'), ['test'])
    assert np.max(axis.lines[0].get_ydata()) <= 0.425
    assert np.min(axis.lines[0].get_ydata()) >= -0.425
    plt.close(figure)


def test_reference_files_have_no_unresolved_placeholder(index):
    ns,sz,_=select_examples(index,'sub_00',index['sub_00'][2]['path'])
    refs=[r['path'] for r in ns+sz]
    for path in [DEFAULT_PROMPT, PROMPTS/'experimental/decision_only.txt', PROMPTS/'experimental/concrete_wording.txt']:
        text,images=prompt_from_paths(index['sub_00'][2]['path'],refs,path)
        assert len(images)==5 and '<Test EEG Image>' not in text


def test_index_validation(index,tmp_path):
    file=tmp_path/'index.pkl'
    file.write_bytes(pickle.dumps(index))
    assert len(load_index(file))==4
    index['sub_00'][0]['vec']=np.zeros(512)
    file.write_bytes(pickle.dumps(index))
    with pytest.raises(ValueError):load_index(file)
