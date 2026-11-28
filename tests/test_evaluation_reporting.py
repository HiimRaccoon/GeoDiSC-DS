import csv, io, json
import numpy as np
import pytest
import torch
from geobench.evaluation import prediction_exports


def classes():
    return [{'class_id':str(c),'class_name':str(c),'group':'extreme-tail','n_original':'2','n_train':'1','singleton':'false'} for c in range(124)]


def test_raw_exports_recompute_rankings_and_full_confusion():
    rows=[{'image_id':'a','relative_path':'a.png','class_id':'0'},{'image_id':'b','relative_path':'b.png','class_id':'1'}]
    scores=torch.zeros(2,124); scores[1,1]=.123456789
    files,metrics=prediction_exports(rows,scores,{'extreme-tail':[0,1]},classes())
    raw=np.load(io.BytesIO(files['logits']),allow_pickle=False)
    assert np.array_equal(raw,scores.numpy()) and raw.dtype==np.float32
    predictions=list(csv.DictReader(io.StringIO(files['predictions'].decode())))
    assert json.loads(predictions[0]['top5_class_ids'])==[0,1,2,3,4]
    assert predictions[1]['logits_row']=='1' and metrics['top1_accuracy']==1
    assert len(metrics['confusion_matrix'])==124 and sum(map(sum,metrics['confusion_matrix']))==2
    assert metrics['per_class'][123]['f1'] is None
    with pytest.raises(ValueError): prediction_exports(rows[::-1],scores,{},classes())
    with pytest.raises(ValueError): prediction_exports([rows[0],rows[0]],scores,{},classes())
    with pytest.raises(ValueError): prediction_exports(rows,scores[:1],{},classes())
    scores[0,0]=float('nan')
    with pytest.raises(ValueError): prediction_exports(rows,scores,{},classes())


@pytest.mark.parametrize('corruption',['wrong-label','row-map','missing-seed'])
def test_complete_reporting_recomputes_eighteen_locks_and_aliases(tmp_path,monkeypatch,corruption):
    from geobench.artifacts import publish_records,sha256_file,verify_lock
    from geobench.bindings import encoded,CONDITIONS,SEEDS
    from geobench.evaluation import csv_bytes
    from geobench import reporting
    rows=[{'image_id':'a','relative_path':'a.png','class_id':'0'}]
    class_rows=classes();groups={'head':[],'medium':[],'extreme-tail':list(range(124))}
    files,metrics=prediction_exports(rows,torch.zeros(1,124),groups,class_rows)
    data=publish_records(tmp_path,'splits','common-data:fixture',{
        'val.csv':csv_bytes(rows,list(rows[0])),'test.csv':csv_bytes(rows,list(rows[0])),
        'class_groups.csv':csv_bytes(class_rows,list(class_rows[0]))},{},owner_token='fixture')
    source=publish_records(tmp_path,'source','run',{'best_checkpoint.pt':b'checkpoint'},{},owner_token='fixture')
    def checkpoint(root,run_id):
        return {'dependencies':{'data':{'path':'splits/lock.json'}}},source,source,{'epoch':1},'run'
    monkeypatch.setattr(reporting,'official_checkpoint',checkpoint)
    for condition in CONDITIONS:
        for seed in SEEDS:
            contents={'metrics.json':encoded({'validation':metrics,'test':metrics}),
                      'evaluation_metadata.json':encoded({'run_id':f'{condition}/{seed}','checkpoint_sha256':sha256_file(source.parent/'best_checkpoint.pt')})}
            for split in ('val','test'):
                for kind,content in files.items():contents[f'{kind}_{split}.{"npy" if kind=="logits" else "csv"}']=content
            publish_records(tmp_path,f'runs/{condition}/seed_{seed}',f'run:{condition}/{seed}',contents,
                {'splits/lock.json':sha256_file(data),'source/lock.json':sha256_file(source)},owner_token='fixture',lock_name='evaluation_lock.json')
    lock=reporting.report(tmp_path,owner_token='fixture')
    assert len(verify_lock(lock,boundary=tmp_path)['dependencies'])==18
    main=list(csv.DictReader(io.StringIO((lock.parent/'main_comparison.csv').read_text(encoding='utf-8'))))
    ablation=list(csv.DictReader(io.StringIO((lock.parent/'ablation_comparison.csv').read_text(encoding='utf-8'))))
    assert {r['condition'] for r in main}=={'E0','E1E2','E3-A4'}
    assert {r['condition'] for r in ablation if r['alias']=='Base'}=={'E1E2'}
    assert all(r['display_percent']=='100.00 ± 0.00' for r in main if r['scope']=='overall' and r['metric']=='macro_f1')
    assert reporting.report(tmp_path,owner_token='fixture')==lock
    target=tmp_path/'runs/E0/seed_42'
    if corruption=='missing-seed':
        (target/'evaluation_lock.json').unlink()
    else:
        path=target/'predictions_test.csv'
        predictions=list(csv.DictReader(io.StringIO(path.read_text())))
        predictions[0]['true_class_id' if corruption=='wrong-label' else 'logits_row']='1'
        path.write_bytes(csv_bytes(predictions,list(predictions[0])))
        # Deliberately forge a new fixture lock to test recomputation beyond checksums.
        lock_path=target/'evaluation_lock.json';record=json.loads(lock_path.read_text())
        record['files']['predictions_test.csv']=sha256_file(path);lock_path.write_text(json.dumps(record))
    with pytest.raises(ValueError):reporting.report(tmp_path,owner_token='fixture')
