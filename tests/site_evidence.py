"""Fictional private-site fixture; never real deployment evidence."""
import json
from geobench.artifacts import claim_writer, staging_path, publish_lock


def publish_site(root):
    destination=root/'deployment/private_site'
    token=claim_writer(destination,'private-site')
    stage=staging_path(destination,token)
    site={'status':'verified','ssh_target':'user@fixture.invalid:22','login_host':'login.fixture.invalid',
          'workspace':'/fixture/workspace/','evidence_files':['observation.txt']}
    (stage/'site.json').write_text(json.dumps(site))
    (stage/'observation.txt').write_text('fixture-only observation')
    return publish_lock(destination,{'identity':'private-site','files':['site.json','observation.txt'],'dependencies':{},'claim':token})
