"""Execute notebooks with an ephemeral kernelspec, preserving partial results on failure."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient

parser=argparse.ArgumentParser()
parser.add_argument('--python',default=sys.executable)
parser.add_argument('notebooks',nargs='*')
args=parser.parse_args()
here=Path(__file__).resolve().parent
package_root=here.parents[1]
os.environ['PYTHONPATH']=str(package_root)+os.pathsep+os.environ.get('PYTHONPATH','')
spec=Path(tempfile.mkdtemp(prefix='sola-notebook-kernel-'))
(spec/'kernels'/'sola-demo').mkdir(parents=True)
(spec/'kernels'/'sola-demo'/'kernel.json').write_text(json.dumps({
    'argv':[args.python,'-m','ipykernel_launcher','-f','{connection_file}'],
    'display_name':'SOLA demo','language':'python'}))
os.environ['JUPYTER_PATH']=str(spec)+os.pathsep+os.environ.get('JUPYTER_PATH','')
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
(here/'output').mkdir(exist_ok=True)
status_path=here/'output'/'execution.json'
status=json.loads(status_path.read_text()) if status_path.exists() else {}
for name in args.notebooks or ['3_pixels_sola.ipynb','4_sobolev_sola.ipynb']:
    path=here/name
    nb=nbformat.read(path,as_version=4)
    start=time.perf_counter()
    try:
        NotebookClient(nb,kernel_name='sola-demo',timeout=1800,
                       resources={'metadata':{'path':str(here)}}).execute()
        status[name]={'status':'pass','seconds':time.perf_counter()-start,'host':socket.gethostname()}
    except Exception as exc:
        status[name]={'status':'failed','seconds':time.perf_counter()-start,'error':repr(exc)}
        raise
    finally:
        nbformat.write(nb,path)
        (here/'output'/'execution.json').write_text(json.dumps(status,indent=2)+'\n')
    print(f'executed {name}: {status[name]["seconds"]:.1f}s',flush=True)
