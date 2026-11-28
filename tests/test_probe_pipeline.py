import csv,io,json
import torch
from PIL import Image
from geobench.artifacts import publish_records,sha256_file
from geobench.images import decoded_pixel_sha256
from geobench.config import METHODS


def test_dino_probe_calls_shared_extractor_and_records_observed_evidence(tmp_path,monkeypatch):
    from geobench.probes import run_probe
    dataset=tmp_path/'dataset';dataset.mkdir();root=tmp_path/'artifacts';root.mkdir()
    path=dataset/'a.png';Image.new('RGB',(8,8)).save(path)
    def publish(relative,identity,files):
        return publish_records(root,relative,identity,files,{},owner_token='fixture')
    model=publish('models/dino','dinov3-snapshot',{'model.json':json.dumps({'snapshot':'snapshot',
        'repository':METHODS['mining']['model'],'revision':METHODS['mining']['revision']}).encode(),'snapshot/config.json':b'{}'})
    publish('deployment/probe_inputs/dino','probe-input:dino',{'input.json':json.dumps({'model_lock':model.relative_to(root).as_posix()}).encode()})
    def table(rows):
        out=io.StringIO(newline='');writer=csv.DictWriter(out,list(rows[0]));writer.writeheader();writer.writerows(rows);return out.getvalue().encode()
    rows=[{'image_id':str(i),'relative_path':'a.png','sha256':sha256_file(path),'decoded_pixel_sha256':decoded_pixel_sha256(path)} for i in range(33)]
    publish('splits','common-data:fixture',{'train.csv':table(rows),'class_groups.csv':b'class_id,class_name\n0,x\n'})
    publish('deployment/data_readiness','data-readiness',{'V01.json':json.dumps({'dataset_root':str(dataset)}).encode()})
    monkeypatch.setenv('SLURM_JOB_ID','fixture');monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    for name in ('reset_peak_memory_stats','synchronize','empty_cache'):monkeypatch.setattr(torch.cuda,name,lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:1024)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'fixture')
    monkeypatch.setattr('geobench.augmentation.mining._load_dino',lambda snapshot:(object(),object(),'cpu'))
    calls=[]
    def extract(model,processor,images,device):
        calls.append(len(images));return torch.ones(len(images),768)
    monkeypatch.setattr('geobench.augmentation.mining.extract_cls',extract)
    lock=run_probe(root,'dino',owner_token='fixture')
    result=json.loads((lock.parent/'probe.json').read_text())
    assert calls==[32,32,1] and result['status']=='observed' and result['state_discarded']
