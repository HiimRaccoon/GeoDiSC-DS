import csv, io, json
import numpy as np
import torch
from PIL import Image
from geobench.artifacts import claim_writer, staging_path, publish_lock, verify_lock, sha256_file
from geobench.images import decoded_pixel_sha256


def test_full_reference_extraction_and_cached_mining(tmp_path,monkeypatch):
    from geobench.augmentation.mining import extract_and_mine
    from geobench.config import METHODS
    def lock(relative,identity,files,name='lock.json'):
        target=tmp_path/relative; token=claim_writer(target,identity); stage=staging_path(target,token)
        for key,content in files.items():
            path=stage/key;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        return publish_lock(target,{'identity':identity,'files':list(files),'dependencies':{},'artifact_root':tmp_path,'claim':token,'lock_name':name})
    path=tmp_path/'source.png';Image.new('RGB',(8,8),'red').save(path)
    rows=[{'image_id':f'{c:03}_{i:02}','class_id':c,'relative_path':'source.png','sha256':sha256_file(path),
           'decoded_pixel_sha256':decoded_pixel_sha256(path),'sample_type':'reference'} for c in range(124) for i in range(20)]
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
    lock('augmentation/E3/A2/references','reference-bank',{'manifest.csv':stream.getvalue().encode()},'bank_lock.json')
    model=lock('models/dino','dinov3-snapshot',{'model.json':json.dumps({'repository':METHODS['mining']['model'],
        'revision':METHODS['mining']['revision'],'snapshot':'snapshot'}).encode(),'snapshot/config.json':b'{}'})
    calls=[]
    class Processor:
        def __call__(self,images,return_tensors):
            calls.append(len(images));return {'pixel_values':torch.ones(len(images),3,8,8)}
    class Model(torch.nn.Module):
        def forward(self,pixel_values):
            out=torch.zeros(len(pixel_values),1,768);out[:,:,0]=1
            return type('Output',(),{'last_hidden_state':out})()
    monkeypatch.setattr('geobench.augmentation.mining._load_dino',lambda path:(Model().eval(),Processor(),'cpu'))
    result=extract_and_mine(tmp_path,model_lock=model.relative_to(tmp_path).as_posix(),owner_token='fixture')
    assert verify_lock(result,boundary=tmp_path)['identity']=='mining'
    assert sum(calls)==2480 and calls[-1]==16
    features=result.parent/'features'
    assert np.load(features/'embeddings.npy',allow_pickle=False).shape==(2480,768)
    assert np.load(features/'centroids.npy',allow_pickle=False).shape==(124,768)
    mapping=json.loads((result.parent/'confusers.json').read_text())
    assert mapping['0']==1 and mapping['1']==0
    assert extract_and_mine(tmp_path,model_lock=model.relative_to(tmp_path).as_posix(),owner_token='fixture')==result
    assert sum(calls)==2480
