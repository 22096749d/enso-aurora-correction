"""GitHub allowlist export, diagnostic return bundles, and local acceptance."""
import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import zipfile
from pathlib import Path
import pandas as pd
from common import ROOT, MODEL, KEY, digest, fresh, read_json, write_json, read_plan
from exchange import verify_plan, collect, verify_return

def export_repo(destination, job):
    verify_plan(ROOT/'jobs'/job)
    if destination.exists():
        raise FileExistsError('Export destination must be NEW; keep the previous Git checkout.')
    destination.mkdir(parents=True)
    # Deliberate allowlist: never recursively copy the working project.
    for name in ['README.md','START_HERE.md','requirements-local.txt','requirements-server.txt','.gitignore',
                 '.gitattributes','config.json']:
        shutil.copy2(ROOT/name,destination/name)
    for name in ['scripts','configs','tests','docs']:
        shutil.copytree(ROOT/name,destination/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    target=destination/'jobs'/job
    target.mkdir(parents=True)
    for name in ['plan.csv','manifest.json']:
        shutil.copy2(ROOT/'jobs'/job/name,target/name)
    write_json(destination/'export_manifest.json',{
        'job':job,'files':{str(p.relative_to(destination)).replace('\\','/'):digest(p)
          for p in destination.rglob('*') if p.is_file()}})
    print('GITHUB EXPORT READY:',destination)
    print('Upload ONLY this directory. No raw data, labels, credentials, weights, or results included.')

def environment(out):
    packages={}
    for name in ['torch','microsoft-aurora','numpy','pandas','xarray','netCDF4','cdsapi','huggingface-hub']:
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]='NOT INSTALLED'
    def command(args):
        try:
            r=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=20)
            return {'returncode':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr.strip()}
        except (OSError,subprocess.TimeoutExpired) as e:return {'error':str(e)}
    write_json(out,{'python':platform.python_version(),'packages':packages,
        'git_commit':command(['git','rev-parse','HEAD']),
        'git_changes':command(['git','status','--porcelain']),
        'gpu':command(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv']),
        'source_hashes':{p.name:digest(p) for p in (ROOT/'scripts').glob('*.py')}})
    print('Environment inventory saved:',out)

def bundle(job, output, destination):
    plan_dir=ROOT/'jobs'/job
    verify_plan(plan_dir)
    if destination.exists():raise FileExistsError('Choose a new return directory/version.')
    destination.mkdir(parents=True)
    try:
        collect(plan_dir,output,destination/'result')
        status='forecasts_available'
    except ValueError as e:
        if not str(e).startswith('No completed outputs'):raise
        status='diagnostics_only'
    for name in ['plan.csv','manifest.json']:
        shutil.copy2(plan_dir/name,destination/name)
    # All timestamps, including failures; omit massive weather fields.
    for stamp in read_plan(plan_dir/'plan.csv').stamp.unique():
        for name in ['done.json','error.json','tracker_errors.json']:
            source=output/stamp/name
            if source.exists():
                target=destination/'diagnostics'/stamp/name
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    environment(destination/'environment.json')
    if (ROOT/'models/checkpoint.json').exists():
        shutil.copy2(ROOT/'models/checkpoint.json',destination/'checkpoint.json')
    # Only logs from this job. User should review exception logs before sharing.
    if (ROOT/'logs'/job).exists():shutil.copytree(ROOT/'logs'/job,destination/'logs')
    write_json(destination/'bundle_manifest.json',{'job':job,'status':status,
        'plan_sha256':digest(plan_dir/'plan.csv'),
        'files':{str(p.relative_to(destination)).replace('\\','/'):digest(p)
                 for p in destination.rglob('*') if p.is_file()}})
    archive=destination.with_suffix('.zip');fresh(archive)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in destination.rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(destination)))
    print('SEND THIS ZIP:',archive,'STATUS:',status)

def accept(directory, job, run):
    meta=read_json(directory/'bundle_manifest.json')
    for name,sha in meta['files'].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory.resolve()):raise ValueError('Invalid bundle path')
        if digest(path)!=sha:raise ValueError('Corrupt return: '+name)
    verify_plan(ROOT/'jobs'/job)
    if meta['job']!=job or meta['plan_sha256']!=digest(ROOT/'jobs'/job/'plan.csv'):
        raise ValueError('Returned plan is NOT your original job.')
    original=read_json(ROOT/'jobs'/job/'manifest.json')
    if original['samples_sha256']!=digest(run/'samples.csv'):
        raise ValueError('Local samples changed after planning. Restore the frozen samples.')
    if meta['status']=='diagnostics_only':raise ValueError('Diagnosis only: no forecasts; send errors for repair, do not train.')
    verify_return(directory/'result')
    identity=read_json(directory/'result/return_manifest.json')['identity']
    if identity.get('model_id')!=MODEL or identity.get('steps')!=20:
        raise ValueError('Wrong backbone or not a 120-hour run.')
    d=pd.read_csv(directory/'result/forecasts.csv')
    if d.duplicated(KEY).any():raise ValueError('Duplicate forecast keys')
    if not d.base_lat.between(-90,90).all() or not d.base_lon.between(-180,180).all():
        raise ValueError('Invalid coordinates')
    if set(d.model_id)!={MODEL}:raise ValueError('Mixed model IDs')
    if not d.lead_h.isin(range(6,121,6)).all():raise ValueError('Unexpected lead times')
    expected=read_plan(ROOT/'jobs'/job/'plan.csv')
    pairs=set(zip(expected.sid,pd.to_datetime(expected.init_time,utc=True)))
    if not set(zip(d.sid,pd.to_datetime(d.init_time,utc=True)))<=pairs:raise ValueError('Unplanned predictions')
    coverage=pd.read_csv(directory/'result/coverage.csv')
    print(coverage.groupby(['split','status']).size().to_string())
    print('Cases with fewer than 20 leads:',int((coverage.available_leads<20).sum()))
    if not coverage.status.eq('complete').all():
        raise ValueError('Unrun/failed global initializations remain. Finish or document a new protocol before fitting.')
    print('ACCEPTED. Review tracker failures and sample counts before training; this is not scientific validation.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['export','inventory','bundle','accept'])
    p.add_argument('--job',default='pilot')
    p.add_argument('--destination',default='github_export')
    p.add_argument('--output',default='server_runs/pilot')
    p.add_argument('--directory',default='received/pilot_v1')
    p.add_argument('--run',default='runs/aurora_samples')
    a=p.parse_args()
    if not a.job.replace('_','').replace('-','').isalnum():p.error('Job name: letters/numbers/underscore/hyphen only')
    if a.command=='export':export_repo(ROOT/a.destination,a.job)
    elif a.command=='inventory':environment(ROOT/a.destination)
    elif a.command=='bundle':bundle(a.job,ROOT/a.output,ROOT/a.destination)
    else:accept(ROOT/a.directory,a.job,ROOT/a.run)

if __name__=='__main__':main()
