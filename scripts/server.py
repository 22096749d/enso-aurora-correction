"""Linux GPU server only: pinned Aurora 0.25 Pretrained + ERA5 + official Tracker.
No training, no future best tracks, no ENSO labels are read by this program.
"""
import argparse
import importlib.metadata
import os
import platform
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT,MODEL,LEVELS,digest,write_json,read_json,read_plan,times_for_init,normalise_track
from exchange import verify_plan,digest_subset

REPO='microsoft/aurora'
CHECKPOINT='aurora-0.25-pretrained.ckpt'
SURF={'2t':'t2m','10u':'u10','10v':'v10','msl':'msl'}
ATMOS=('t','u','v','q','z')
STATIC={'lsm':'lsm','slt':'slt','z':'z'}

def setup_cache():
    # Do not repurpose HOME; put the task's model cache beneath this project.
    os.environ.setdefault('HF_HOME',str(ROOT/'models/hf_cache'))

def weights():
    setup_cache()
    from huggingface_hub import HfApi,hf_hub_download
    lock=ROOT/'models/checkpoint.json'
    if lock.exists():
        old=read_json(lock)
        path=hf_hub_download(REPO,CHECKPOINT,revision=old['revision'])
        if digest(path)!=old['sha256']:raise ValueError('Checkpoint hash mismatch.')
        print('Locked weights already available:',old['revision']);return
    revision=HfApi().model_info(REPO).sha
    path=hf_hub_download(REPO,CHECKPOINT,revision=revision)
    write_json(lock,{'repo':REPO,'checkpoint':CHECKPOINT,'revision':revision,'sha256':digest(path),
        'downloaded_utc':pd.Timestamp.now(tz='UTC'),'model_class':'AuroraPretrained','input_source':'ERA5'})
    print('WEIGHTS OK. Exact repository revision and file SHA256 saved.')

def check():
    import torch
    from aurora import AuroraPretrained,Tracker
    version=importlib.metadata.version('microsoft-aurora')
    result={'python':platform.python_version(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,
        'aurora_package':version,'cuda_available':torch.cuda.is_available()}
    if not torch.cuda.is_available():raise RuntimeError(f'CUDA unavailable: {result}')
    result.update(gpu=torch.cuda.get_device_name(0),gpu_total_gib=torch.cuda.get_device_properties(0).total_memory/2**30)
    if version!='2.0.1':raise RuntimeError('This adapter targets microsoft-aurora==2.0.1; create a separate versioned experiment to change it.')
    a=torch.ones((256,256),device='cuda'); b=a@a
    torch.cuda.synchronize()
    if not torch.isfinite(b).all():raise RuntimeError('GPU arithmetic check failed.')
    write_json(ROOT/'logs/server_check.json',result)
    print(result)
    print('GPU arithmetic OK, NOT a full Aurora memory test.')

def retrieve_file(client,dataset,request,target):
    stamp=target.with_suffix('.manifest.json')
    if target.exists():
        if not stamp.exists() or read_json(stamp)['request']!=request or digest(target)!=read_json(stamp)['sha256']:
            raise ValueError(f'Incomplete/unverified cached file: {target}. Inspect/archive this exact file and manifest.')
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    partial=target.with_suffix('.part')
    if partial.exists():raise FileExistsError(f'Interrupted download: {partial}. Inspect/archive before retrying.')
    client.retrieve(dataset,request,str(partial))
    partial.rename(target)
    write_json(stamp,{'dataset':dataset,'request':request,'sha256':digest(target)})

def request_at(t,variables):
    return {'product_type':['reanalysis'],'year':[t.strftime('%Y')],'month':[t.strftime('%m')],
        'day':[t.strftime('%d')],'time':[t.strftime('%H:00')],
        'variable':variables,'data_format':'netcdf','download_format':'unarchived'}

def data(plan,limit):
    import cdsapi
    client=cdsapi.Client()
    static_request=request_at(pd.Timestamp('2000-01-01'),['land_sea_mask','soil_type','geopotential'])
    retrieve_file(client,'reanalysis-era5-single-levels',static_request,ROOT/'data/era5/static.nc')
    for stamp in sorted(plan.stamp.unique())[:limit]:
        for t in times_for_init(stamp):
            directory=ROOT/'data/era5'/t.strftime('%Y%m%d%H')
            request=request_at(t,['2m_temperature','10m_u_component_of_wind','10m_v_component_of_wind','mean_sea_level_pressure'])
            retrieve_file(client,'reanalysis-era5-single-levels',request,directory/'surface.nc')
            request=request_at(t,['temperature','u_component_of_wind','v_component_of_wind','specific_humidity','geopotential'])
            request['pressure_level']=[str(v) for v in LEVELS]
            retrieve_file(client,'reanalysis-era5-pressure-levels',request,directory/'upper.nc')
        print('ERA5 initial pair ready:',stamp,flush=True)

def fields(path,mapping,expected_time=None,upper=False):
    import xarray as xr
    manifest=path.with_suffix('.manifest.json')
    if not manifest.exists() or digest(path)!=read_json(manifest)['sha256']:
        raise ValueError(f'Unverified ERA5 file: {path}')
    with xr.open_dataset(path) as ds:
        names={k:v for k,v in [('valid_time','time'),('pressure_level','level'),('lat','latitude'),('lon','longitude')] if k in ds.dims}
        ds=ds.rename(names)
        if 'time' in ds.dims:
            if ds.sizes['time']!=1:raise ValueError('Each cached file must contain exactly one time.')
            if expected_time is not None and pd.Timestamp(ds.time.values[0])!=expected_time.tz_localize(None):
                raise ValueError('ERA5 time mismatch')
            ds=ds.isel(time=0,drop=True)
        elif expected_time is not None:raise ValueError('Missing ERA5 valid time.')
        ds=ds.assign_coords(longitude=ds.longitude%360).sortby('longitude').sortby('latitude',ascending=False)
        lat=ds.latitude.values;lon=ds.longitude.values
        if len(lat)!=721 or len(lon)!=1440:raise ValueError('Need global 0.25 degree grid (721,1440).')
        if not np.allclose(lat,np.arange(90,-90.1,-.25)) or not np.allclose(lon,np.arange(0,360,.25)):
            raise ValueError('Unexpected latitude/longitude coordinates.')
        if upper:ds=ds.sel(level=list(LEVELS))
        dims=['level','latitude','longitude'] if upper else ['latitude','longitude']
        result={}
        for k,name in mapping.items():
            if name not in ds:raise ValueError(f'{path} missing variable {name}')
            units=ds[name].attrs.get('units','')
            arr=ds[name].transpose(*dims).values.astype('float32')
            if not np.isfinite(arr).all():raise ValueError(f'Nonfinite {name}; do not fill missing weather with zero.')
            if name=='z' and units in ('m','gpm'):raise ValueError('Geopotential required, not geopotential height.')
            if name=='msl' and not 20000 < float(np.median(arr)) < 120000:raise ValueError('MSLP must be Pa, not hPa.')
            result[k]=arr
        return result,lat,lon

def make_batch(stamp):
    import torch
    from aurora import Batch,Metadata
    surface=[];atmos=[]
    for t in times_for_init(stamp):
        directory=ROOT/'data/era5'/t.strftime('%Y%m%d%H')
        s,lat,lon=fields(directory/'surface.nc',SURF,t)
        a,_,_=fields(directory/'upper.nc',{k:k for k in ATMOS},t,upper=True)
        surface.append(s);atmos.append(a)
    static,_,_=fields(ROOT/'data/era5/static.nc',STATIC)
    batch=Batch(surf_vars={k:torch.from_numpy(np.stack([s[k] for s in surface])[None]) for k in SURF},
        atmos_vars={k:torch.from_numpy(np.stack([a[k] for a in atmos])[None]) for k in ATMOS},
        static_vars={k:torch.from_numpy(v) for k,v in static.items()},
        metadata=Metadata(lat=torch.from_numpy(lat),lon=torch.from_numpy(lon),
            time=(times_for_init(stamp)[1].tz_localize(None).to_pydatetime(),),atmos_levels=LEVELS))
    return batch

def run(plan,output,limit,steps,save_msl):
    setup_cache()
    import torch
    from aurora import AuroraPretrained,Tracker,rollout
    if importlib.metadata.version('microsoft-aurora')!='2.0.1':raise RuntimeError('Unexpected Aurora package version.')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no automatic CPU fallback.')
    lock=read_json(ROOT/'models/checkpoint.json')
    from huggingface_hub import hf_hub_download
    cached=hf_hub_download(REPO,CHECKPOINT,revision=lock['revision'])
    if digest(cached)!=lock['sha256']:raise ValueError('Cached checkpoint differs from the locked SHA256.')
    identity={'checkpoint_revision':lock['revision'],'checkpoint_sha256':lock['sha256'],
        'model_id':MODEL,'aurora_package':importlib.metadata.version('microsoft-aurora'),
        'torch':str(torch.__version__),'precision':'default_fp32','steps':steps,
        'static_sha256':digest(ROOT/'data/era5/static.nc'),
        'adapter_sha256':digest(Path(__file__)),'common_sha256':digest(ROOT/'scripts/common.py')}
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    model=AuroraPretrained()
    model.load_checkpoint(REPO,CHECKPOINT,revision=lock['revision'])
    model.eval();model=model.to('cuda')
    completed=0
    for stamp,group in plan.groupby('stamp',sort=True):
        directory=output/stamp
        if (directory/'done.json').exists():
            old=read_json(directory/'done.json')
            if old['identity']!=identity or old['plan_subset_sha256']!=digest_subset(group):
                raise ValueError('Existing output uses another plan/config/version. Use a new server output directory.')
            continue
        if directory.exists():raise FileExistsError(f'Incomplete output: {directory}. Inspect/archive that exact timestamp directory.')
        directory.mkdir(parents=True)
        start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        try:
            batch=make_batch(stamp).to('cuda')
            active={row.sid:Tracker(init_lat=float(row.lat0),init_lon=float(row.lon0)%360,
                init_time=times_for_init(stamp)[1].tz_localize(None).to_pydatetime()) for row in group.itertuples()}
            rows=[];errors=[]
            with torch.inference_mode():
                for i,pred in enumerate(rollout(model,batch,steps=steps),1):
                    cpu=pred.to('cpu')
                    expected=times_for_init(stamp)[1].tz_localize(None)+pd.Timedelta(hours=6*i)
                    if pd.Timestamp(cpu.metadata.time[0])!=expected:raise ValueError('Forecast timestamp mismatch.')
                    for arrays in (cpu.surf_vars,cpu.atmos_vars):
                        if any(not torch.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite forecast.')
                    if save_msl:
                        np.savez_compressed(directory/f'msl_{i*6:03d}.npz',msl=cpu.surf_vars['msl'][0,0].numpy(),
                            lat=cpu.metadata.lat.numpy(),lon=cpu.metadata.lon.numpy())
                    for sid,tracker in list(active.items()):
                        failures_before=int(tracker.fails)
                        try:
                            tracker.step(cpu)
                            result=normalise_track(tracker.results().tail(1),sid,times_for_init(stamp)[1])
                            result['tracker_fallback']=int(tracker.fails)>failures_before
                            result['fallback_count']=int(tracker.fails)
                            result['tracker_status']='official_tracker_not_yet_validated_on_our_cases'
                            rows.extend(result.to_dict('records'))
                        except Exception as exc:
                            errors.append({'sid':sid,'lead_h':i*6,'type':type(exc).__name__,'error':str(exc)})
                            del active[sid]  # Retain all failures in audit; never use future truth to restart.
                    print(stamp,'lead_h',i*6,'active_tracks',len(active),flush=True)
                    del cpu
            columns=['sid','init_time','lead_h','model_id','base_lat','base_lon','tracker_fallback','fallback_count','tracker_status']
            forecasts=pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
            forecasts.to_csv(directory/'forecasts.csv',index=False)
            write_json(directory/'tracker_errors.json',errors)
            torch.cuda.synchronize()
            inputs=[ROOT/'data/era5/static.nc']+[ROOT/'data/era5'/t.strftime('%Y%m%d%H')/f for t in times_for_init(stamp) for f in ['surface.nc','upper.nc']]
            write_json(directory/'done.json',{'identity':identity,'plan_subset_sha256':digest_subset(group),
                'forecasts_sha256':digest(directory/'forecasts.csv'),'wall_seconds':time.perf_counter()-start,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
                'inputs_sha256':{str(p.relative_to(ROOT)):digest(p) for p in inputs},
                'planned_storms':len(group),'tracker_failures':errors})
            del batch
        except Exception as exc:
            write_json(directory/'error.json',{'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
            raise
        completed+=1
        if completed>=limit:break
    print('New global initializations completed:',completed)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','weights','data','run'])
    p.add_argument('--plan',default='transfer/pilot/plan.csv')
    p.add_argument('--output',default='server_runs/pilot')
    p.add_argument('--limit',type=int,default=1)
    p.add_argument('--steps',type=int,default=1,choices=[1,4,20])
    p.add_argument('--save-msl',action='store_true')
    a=p.parse_args()
    if a.limit<1:p.error('--limit must be positive')
    if a.command=='check':check()
    elif a.command=='weights':weights()
    else:
        path=ROOT/a.plan;verify_plan(path.parent);plan=read_plan(path)
        if a.command=='data':data(plan,a.limit)
        else:run(plan,ROOT/a.output,a.limit,a.steps,a.save_msl)

if __name__=='__main__':main()
