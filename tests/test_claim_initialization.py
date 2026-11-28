import hashlib,json
import pytest
from geobench.artifacts import claim_writer,staging_path


def test_owner_recovers_interrupted_claim_and_cannot_steal_committed_intent(tmp_path):
    record=json.dumps({'identity':'run','token_sha256':hashlib.sha256(b'owner').hexdigest()},sort_keys=True).encode()
    (tmp_path/'.writer.intent').write_bytes(record)
    (tmp_path/'.writer').mkdir()
    (tmp_path/'.writer/claim.json').write_bytes(record[:10])
    with pytest.raises(ValueError):claim_writer(tmp_path,'run',token='stranger',resume=True)
    token=claim_writer(tmp_path,'run',token='owner',resume=True)
    assert staging_path(tmp_path,token).is_dir()
    token.close()
    assert (tmp_path/'.writer/claim.json').read_bytes()==record
