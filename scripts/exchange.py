"""LOCAL planning and SERVER/LOCAL transfer validation. Does not run GPU work."""
import argparse
import shutil
import zipfile
import pandas as pd
from common import ROOT,KEY,MODEL,digest,write_json,read_json,fresh,read_plan,utc

def plan(run,destination,limit):
    fresh(destination/'plan.csv')
    d=pd.read_csv(run/'samples.csv')
    d['init_time']=pd.to_datetime(d.init_time)
    d=d[(d.split!='base')&(d.init_time.dt.hour==0)]
    d=d[['sid','init_time','lat0','lon0','split']].drop_duplicates(['sid','init_time'])
    d=d.sort_values(['init_time','sid']).reset_index(drop=True)
    # Pilot samples uniformly through each split, independent of forecast skill.
    if limit:
        import numpy as np
        parts=[]
        for _,g in d.groupby('split'):
            parts.append(g.iloc[np.linspace(0,len(g)-1,min(limit,len(g)),dtype=int)])
        d=pd.concat(parts).sort_values(['init_time','sid'])
    if d.empty:raise ValueError('No eligible cases.')
    d['stamp']=d.init_time.dt.strftime('%Y%m%d%H')
    d.to_csv(destination/'plan.csv',index=False)
    manifest={'plan_sha256':digest(destination/'plan.csv'),'samples_sha256':digest(run/'samples.csv'),
              'scope':'engineering_pilot' if limit else 'full_retrospective_plan',
              'max_cases_per_split':limit,'model_id':MODEL,
              'note':'No future target coordinates or ENSO features transferred to the server.'}
    write_json(destination/'manifest.json',manifest)
    if (run/'data_manifest.json').exists():shutil.copy2(run/'data_manifest.json',destination/'data_manifest.json')
    print(d.groupby('split').size().to_string())
    print('Storm/init cases:',len(d),'Unique initial timestamps:',d.stamp.nunique())

def pack(source,out):
    fresh(out)
    if not (source/'plan.csv').exists():raise ValueError('Missing plan.')
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(source.iterdir()):
            if p.is_file():z.write(p,p.name)
    print(out)

def verify_plan(directory):
    meta=read_json(directory/'manifest.json')
    if digest(directory/'plan.csv')!=meta['plan_sha256']:raise ValueError('Plan hash mismatch: transfer/edit corruption.')
    print('PLAN OK',len(read_plan(directory/'plan.csv')))

def collect(plan_dir,outdir,destination):
    fresh(destination/'forecasts.csv')
    verify_plan(plan_dir)
    d=read_plan(plan_dir/'plan.csv');files=[];audit=[];identity=None
    for stamp,group in d.groupby('stamp',sort=True):
        folder=outdir/stamp
        done=folder/'done.json'
        status='complete' if done.exists() else ('failed' if (folder/'error.json').exists() else 'not_run')
        sub=pd.DataFrame()
        if status=='complete':
            info=read_json(done)
            if info['plan_subset_sha256']!=digest_subset(group):raise ValueError('Plan changed since inference; use a fresh server run.')
            if digest(folder/'forecasts.csv')!=info['forecasts_sha256']:raise ValueError('Forecast file changed.')
            this_identity=info['identity']
            if identity is None:identity=this_identity
            elif identity!=this_identity:raise ValueError('Mixed checkpoint/software/protocol versions. Separate the runs.')
            sub=pd.read_csv(folder/'forecasts.csv')
            if not sub.empty:files.append(sub)
        for row in group.itertuples():
            part=sub[sub.sid==row.sid] if not sub.empty else sub
            audit.append(dict(sid=row.sid,init_time=row.init_time,stamp=stamp,split=row.split,status=status,
                available_leads=0 if part.empty else part.lead_h.nunique(),
                fallback_steps=0 if part.empty else int(part.tracker_fallback.fillna(False).sum())))
    destination.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(audit).to_csv(destination/'coverage.csv',index=False)
    if not files:raise ValueError('No completed outputs. Coverage saved for diagnosis; no training file created.')
    forecasts=pd.concat(files,ignore_index=True)
    if forecasts.duplicated(KEY).any():raise ValueError('Duplicate forecasts.')
    forecasts.to_csv(destination/'forecasts.csv',index=False)
    shutil.copy2(plan_dir/'plan.csv',destination/'plan.csv')
    write_json(destination/'return_manifest.json',{'identity':identity,'original_plan_manifest':read_json(plan_dir/'manifest.json'),
        'files':{p.name:digest(p) for p in destination.iterdir() if p.is_file() and p.name!='return_manifest.json'},
        'planned_cases':len(d),'completed_init_times':sum((outdir/s/'done.json').exists() for s in d.stamp.unique())})
    print(destination,'Check coverage before comparing conditional errors.')

def digest_subset(d):
    import hashlib
    text=d[['sid','init_time','stamp','lat0','lon0','split']].sort_values('sid').to_csv(index=False,float_format='%.8f')
    return hashlib.sha256(text.encode()).hexdigest()

def verify_return(directory):
    manifest=read_json(directory/'return_manifest.json')
    for name,value in manifest['files'].items():
        if digest(directory/name)!=value:raise ValueError(f'Hash mismatch: {name}')
    print('RETURN OK. Identity:',manifest['identity'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','pack','verify-plan','collect','verify-return'])
    p.add_argument('--run',default='runs/aurora_samples')
    p.add_argument('--directory',default='transfer/pilot')
    p.add_argument('--max-per-split',type=int,default=2)
    p.add_argument('--outdir',default='server_runs/pilot')
    p.add_argument('--output',default='transfer/pilot.zip')
    a=p.parse_args();directory=ROOT/a.directory
    if a.max_per_split<0:p.error('Use 0 for all cases, positive integer for a pilot.')
    if a.command=='plan':plan(ROOT/a.run,directory,a.max_per_split)
    elif a.command=='pack':pack(directory,ROOT/a.output)
    elif a.command=='verify-plan':verify_plan(directory)
    elif a.command=='collect':collect(directory,ROOT/a.outdir,ROOT/a.output)
    else:verify_return(directory)

if __name__=='__main__':main()
