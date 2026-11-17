"""Execute part 1 with a temporary kernelspec and explicit local package imports."""
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
parser.add_argument('notebooks',nargs='*')
parser.add_argument('--threads',type=int,default=1)
args=parser.parse_args()
here=Path(__file__).resolve().parent
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]=str(args.threads)
os.environ['PYTHONPATH']=os.pathsep.join(map(str,(here,here.parent,here.parents[2])))+os.pathsep+os.environ.get('PYTHONPATH','')
os.environ['MPLBACKEND']='Agg'
spec=Path(tempfile.mkdtemp(prefix='sola-part1-kernel-'))
(spec/'kernels'/'sola-part1').mkdir(parents=True)
(spec/'kernels'/'sola-part1'/'kernel.json').write_text(json.dumps(dict(
    argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
    display_name='SOLA part 1',language='python')))
os.environ['JUPYTER_PATH']=str(spec)+os.pathsep+os.environ.get('JUPYTER_PATH','')
out=here/'output';out.mkdir(exist_ok=True)
status_path=out/'execution.json'
status=json.loads(status_path.read_text()) if status_path.exists() else {}
for name in args.notebooks or ['01_classic_pixels.ipynb','02_sobolev_galerkin.ipynb',
                              '03_sobolev_continuum.ipynb','04_costs_and_comparison.ipynb']:
    path=here/name;nb=nbformat.read(path,as_version=4);start=time.perf_counter()
    try:
        NotebookClient(nb,kernel_name='sola-part1',timeout=2400,
                       resources={'metadata':{'path':str(here)}}).execute()
        status[name]=dict(status='pass',seconds=time.perf_counter()-start,
                          host=socket.gethostname(),threads=args.threads)
    except Exception as exc:
        status[name]=dict(status='failed',seconds=time.perf_counter()-start,error=repr(exc))
        raise
    finally:
        nbformat.write(nb,path)
        status_path.write_text(json.dumps(status,indent=2)+'\n')
    print(f"{name}: {status[name]['seconds']:.1f}s",flush=True)
