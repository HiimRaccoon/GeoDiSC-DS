"""Recompute locked predictions and aggregate exactly three official seeds."""
import csv
import json
from pathlib import Path
import numpy as np
from .artifacts import publish_records,sha256_file,verify_lock
from .bindings import CONDITIONS,SEEDS,encoded
from .evaluation import prediction_exports,csv_bytes,official_checkpoint
from .metrics import aggregate_seeds,format_aggregate

OVERALL=('macro_f1','top1_accuracy','top5_accuracy','weighted_f1','balanced_accuracy')


def report(root,*,owner_token):
    root=Path(root).resolve();deps={};per_seed=[];values={};support={};class_values={};provenance=[]
    for condition in CONDITIONS:
        for seed in SEEDS:
            run_id=f'{condition}/{seed}'
            descriptor,source,authority,best,identity=official_checkpoint(root,run_id)
            target=root/f'runs/{condition}/seed_{seed}';path=target/'evaluation_lock.json'
            lock=verify_lock(path,boundary=root);deps[path.relative_to(root).as_posix()]=sha256_file(path)
            metadata=json.loads((target/'evaluation_metadata.json').read_text())
            if metadata['run_id']!=run_id or metadata['checkpoint_sha256']!=sha256_file(source.parent/'best_checkpoint.pt'):
                raise ValueError('evaluation checkpoint identity mismatch')
            data=root/descriptor['dependencies']['data']['path'];verify_lock(data,boundary=root)
            def read(name):
                with (data.parent/name).open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))
            classes=read('class_groups.csv');groups={name:[int(r['class_id']) for r in classes if r['group']==name] for name in ('head','medium','extreme-tail')}
            recorded=json.loads((target/'metrics.json').read_text())
            for split in ('val','test'):
                rows=sorted(read(split+'.csv'),key=lambda r:(int(r['class_id']),r['relative_path']))
                logits=np.load(target/f'logits_{split}.npy',allow_pickle=False)
                exports,metrics=prediction_exports(rows,logits,groups,classes)
                for kind,content in exports.items():
                    filename=f'{kind}_{split}.{"npy" if kind=="logits" else "csv"}'
                    if filename not in lock['files'] or (target/filename).read_bytes()!=content:
                        raise ValueError('prediction recomputation mismatch')
                if recorded['validation' if split=='val' else 'test']!=metrics:
                    raise ValueError('stored metrics differ from raw predictions')
                current={'samples':len(rows),'output_classes':124,'supports':[r['support'] for r in metrics['per_class']],
                         'evaluated_class_ids':[i for i,r in enumerate(metrics['per_class']) if r['evaluable']],
                         'groups':{g:{key:v[key] for key in ('support','num_evaluable_classes')} for g,v in metrics['groups'].items()},
                         'classes':classes,'class_groups_sha256':sha256_file(data.parent/'class_groups.csv'),
                         'manifest_sha256':sha256_file(data.parent/(split+'.csv'))}
                if split in support and support[split]!=current: raise ValueError('evaluation supports differ across official runs')
                support[split]=current
                scopes={'overall':{key:metrics[key] for key in OVERALL},**{
                    group:{key:result[key] for key in ('macro_f1','macro_recall')} for group,result in metrics['groups'].items()}}
                for scope,results in scopes.items():
                    for metric,value in results.items():
                        key=(condition,split,scope,metric)
                        values.setdefault(key,[]).append({'seed':seed,'value':value})
                        per_seed.append({'condition':condition,'seed':seed,'split':split,'scope':scope,'metric':metric,'value':value,
                            'selected_epoch':best['epoch'],'checkpoint_sha256':metadata['checkpoint_sha256'],'source':identity})
                for c,result in enumerate(metrics['per_class']):
                    for metric in ('f1','recall'):
                        class_values.setdefault((condition,split,c,metric),[]).append({'seed':seed,'value':result[metric]})
            provenance.append({'run_id':run_id,'selected_epoch':best['epoch'],'source_lock':source.relative_to(root).as_posix(),
                               'evaluation_lock':path.relative_to(root).as_posix(),'dependencies':descriptor['dependencies']})
    aggregate=[]
    for (condition,split,scope,metric),records in values.items():
        aggregate.append({'condition':condition,'split':split,'scope':scope,'metric':metric,
                          'seeds':records,**aggregate_seeds(records)})
    per_class=[]
    for (condition,split,c,metric),records in class_values.items():
        summary=aggregate_seeds(records)
        info=next(row for row in support[split]['classes'] if int(row['class_id'])==c)
        per_class.append({'condition':condition,'split':split,'class_id':c,'class_name':info['class_name'],'group':info['group'],
                          'n_original':info['n_original'],'n_train':info['n_train'],'singleton':info['singleton'],'metric':metric,
                          'support':support[split]['supports'][c],**summary,'display':format_aggregate(summary)})
    def comparison(aliases):
        rows=[]
        for alias,condition in aliases:
            for item in aggregate:
                if item['condition']==condition and item['split']=='test':
                    rows.append({'alias':alias,'condition':condition,'scope':item['scope'],'metric':item['metric'],
                                 'display_percent':format_aggregate(item)})
        return csv_bytes(rows,list(rows[0]))
    main=comparison([(c,c) for c in ('E0','E1E2','E3-A4')])
    ablation=comparison([('Base','E1E2'),*[(f'A{i}',f'E3-A{i}') for i in range(1,5)]])
    reused=all((root/f'runs/E1E2/seed_{seed}/official_binding.json').exists() for seed in SEEDS)
    text=['# GeoDiSC-DS benchmark report','',
          'RQ1 compares E0 with E1E2. RQ2 compares E1E2 with E3-A4. Overall Macro F1 is the primary metric.',
          'Results retain three seeds and sample standard deviations. Unsupported classes have N/A scores.',
          f'Official identities: 18. Unique classifier executions: {24 if reused else 27}, excluding retries.',
          'Validation and test use fixed real manifests. Singleton classes remain in the classifier output space.',
          'Main and ablation tables share E1E2/Base and A4 artifacts; no extra test executions are introduced.','',
          '| Condition | Test Macro F1 (%) |','|---|---|']
    for item in aggregate:
        if item['split']=='test' and item['scope']=='overall' and item['metric']=='macro_f1':
            text.append(f'| {item["condition"]} | {format_aggregate(item)} |')
    text += ['','## Evidence','', 'See evaluation_support.json, per_seed_metrics.csv, aggregate_metrics.json, main_comparison.csv, ablation_comparison.csv and per_class_comparison.csv.']
    for entry in provenance:
        relative=entry['evaluation_lock'].removesuffix('evaluation_lock.json')
        text.append(f'- {entry["run_id"]}: epoch {entry["selected_epoch"]}; [validation confusion](../{relative}confusion_matrix_val.csv), [test confusion](../{relative}confusion_matrix_test.csv).')
    contents={'evaluation_support.json':encoded(support),'per_seed_metrics.csv':csv_bytes(per_seed,list(per_seed[0])),
        'aggregate_metrics.json':encoded({'results':aggregate,'provenance':provenance}),
        'main_comparison.csv':main,'ablation_comparison.csv':ablation,
        'per_class_comparison.csv':csv_bytes(per_class,list(per_class[0])), 'final_report.md':('\n'.join(text)+'\n').encode()}
    return publish_records(root,'reports','benchmark-report',contents,deps,owner_token=owner_token,lock_name='report_lock.json')
