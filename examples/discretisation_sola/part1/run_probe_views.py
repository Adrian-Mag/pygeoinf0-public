"""Execute only new physical/probe cells, preserving all previous notebook outputs.

Run from package root src after the four original notebooks. Reuses their saved
weights; it does not solve SOLA or rerun the timing benchmark. Provenance records
explicitly distinguish this execution from the original notebook runs.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient

HERE=Path(__file__).resolve().parent
OUT=HERE/'output'
OUT.mkdir(exist_ok=True)
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONPATH']=os.pathsep.join(map(str,(HERE,HERE.parent,HERE.parents[2])))+os.pathsep+os.environ.get('PYTHONPATH','')
os.environ['MPLBACKEND']='Agg'
spec=Path(tempfile.mkdtemp(prefix='sola-probe-kernel-'))
(spec/'kernels'/'sola-probe').mkdir(parents=True)
(spec/'kernels'/'sola-probe'/'kernel.json').write_text(json.dumps(dict(
    argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
    display_name='SOLA physical probes',language='python')))
os.environ['JUPYTER_PATH']=str(spec)+os.pathsep+os.environ.get('JUPYTER_PATH','')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

inputs={f:sha(OUT/f) for f in ('pixel_weights.npz','galerkin_weights.npz',
    'continuum_reference.npz','pixels.json','galerkin.json','runtime.json')}
if (OUT/'artifact_hashes.json').exists():
    expected=json.loads((OUT/'artifact_hashes.json').read_text())
    for name,digest in inputs.items():
        if expected.get(name)!=digest:
            raise RuntimeError(f'{name} differs from the original validated input manifest')
provenance=dict(status='running',scope='new physical/probe cells only; original cells/outputs retained',
    host=socket.gethostname(),pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    input_sha256=inputs,helper_sha256=sha(HERE/'probe_views.py'),runner_sha256=sha(__file__),
    notebooks={})
if (OUT/'execution_manifest.json').exists():
    provenance['weights_source']=json.loads((OUT/'execution_manifest.json').read_text())['source']['execution_commit']
record=OUT/'probe_execution.json'
record.write_text(json.dumps(provenance,indent=2)+'\n')

setup=r'''%matplotlib inline
from pathlib import Path
import hashlib,json,os,socket,sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_info
import study as F
import pygeoinf0 as pg
OUT=Path('output')
runtime=F.load(OUT/'runtime.json')
assert hashlib.sha256(Path(F.S.P.__file__).read_bytes()).hexdigest()==runtime['problem_sha256']
ex=F.Experiment(centres=tuple(tuple(c) for c in runtime['targets']),radius=runtime['radius'],
                lam=runtime['lambda_value'],length_scale=runtime['length_scale'])
plt.rcParams.update({'figure.dpi':110,'axes.grid':True,'grid.alpha':.2,'font.size':10})
F.save(OUT/'probe_runtime.json',dict(host=socket.gethostname(),python=sys.version,numpy=np.__version__,
       scipy=scipy.__version__,package_source=pg.__file__,threadpools=threadpool_info(),
       length_scale=ex.length_scale,centres=ex.centres,radius=ex.radius))
'''

for path in sorted(HERE.glob('0*.ipynb')):
    notebook=nbformat.read(path,as_version=4)
    indices=[i for i,c in enumerate(notebook.cells) if c.cell_type=='code' and c.source.startswith('# physical-probe:')]
    if len(indices)!=1:raise RuntimeError(f'Expected one new physical/probe cell in {path.name}')
    index=indices[0]
    extra=''
    if path.name.startswith('01'):
        extra="\nrows=F.load(OUT/'pixels.json')['rows']\nwith np.load(OUT/'pixel_weights.npz') as saved:weights=saved['weights']\n"
    elif path.name.startswith('02'):
        extra="\nrows=F.load(OUT/'galerkin.json')['rows']\nwith np.load(OUT/'galerkin_weights.npz') as saved:weights=saved['weights'];Xref=saved['reference']\n"
    else:
        extra="\nwith np.load(OUT/'continuum_reference.npz') as saved:Xref=saved['X']\n"
    temporary=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(setup+extra),notebook.cells[index]])
    start=time.perf_counter();status='failed'
    try:
        NotebookClient(temporary,kernel_name='sola-probe',timeout=1200,
            resources={'metadata':{'path':str(HERE)}}).execute()
        status='pass'
    finally:
        notebook.cells[index]=temporary.cells[-1]
        info=dict(status=status,seconds=time.perf_counter()-start,host=socket.gethostname(),
                  scope=provenance['scope'],source_cell_sha256=hashlib.sha256(notebook.cells[index].source.encode()).hexdigest())
        notebook.metadata['sola_probe_execution']=info
        provenance['notebooks'][path.name]=info
        nbformat.write(notebook,path)
        nbformat.write(temporary,OUT/f'probe_execution_{path.name}')
        record.write_text(json.dumps(provenance,indent=2)+'\n')
    print(f'{path.name}: {status}, {info["seconds"]:.2f}s',flush=True)
provenance['status']='pass'
provenance['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
record.write_text(json.dumps(provenance,indent=2)+'\n')
