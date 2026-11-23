"""Prepare two optional upstream files from the exact archived PolyEKF revision."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

root = Path(__file__).resolve().parent
commit = 'cbb1a0a0bd911ecc7c5ff3d034e8336f31e1dfea'
hashes = {
    'baseline_prob_ssm.py':'5d6d417169adf2f426f2418abdfe'+'b86741ce7b1e1a671f745375206261785bc1',
    'util_func.py':'f5404214cb844e94e5bb425e08a0dd55f3ff209fccfad0d217ea1cddcec9dfd4'
}
parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,help='Directory of original upstream Python modules')
args=parser.parse_args()
target=root/'theta_u_topology_20261023/baseline_polyekf_source_20261025/raw_snapshot'
for name,expected in hashes.items():
    if args.source:
        data=(args.source/name).read_bytes()
    else:
        url=f'https://api.github.com/repos/lital-dgold/PolyEKF/contents/{name}?ref={commit}'
        with urlopen(url,timeout=20) as stream:
            data=base64.b64decode(json.load(stream)['content'])
    if hashlib.sha256(data).hexdigest()!=expected:
        raise ValueError(f'{name} does not match the frozen revision')
    target.mkdir(parents=True,exist_ok=True)
    (target/name).write_bytes(data)
print('Verified both upstream modules at',commit)

