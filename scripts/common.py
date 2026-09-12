"""Shared small-file protocol. No torch import: runnable on the local PC."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
KEY=['sid','init_time','lead_h']
MODEL='aurora025_pretrained_era5_officialtracker'
LEVELS=(50,100,150,200,250,300,400,500,600,700,850,925,1000)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def fresh(path):
    path=Path(path)
    if path.exists():raise FileExistsError(f'Keeping existing output: {path}. Choose a new path/run.')
    path.parent.mkdir(parents=True,exist_ok=True)

def utc(value):
    return pd.Timestamp(value).tz_localize('UTC') if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value).tz_convert('UTC')

def times_for_init(stamp):
    t=pd.to_datetime(str(stamp),format='%Y%m%d%H',utc=True)
    if t.hour%6:raise ValueError('Initialization must be a six-hour UTC synoptic time.')
    return (t-pd.Timedelta(hours=6),t)

def read_plan(path):
    d=pd.read_csv(path,dtype={'sid':str,'stamp':str})
    required=['sid','init_time','stamp','lat0','lon0','split']
    if not set(required)<=set(d):raise ValueError(f'Plan needs {required}')
    if d.empty or d.duplicated(['sid','init_time']).any():raise ValueError('Empty or duplicate plan.')
    if not (d.lat0.between(-90,90).all() and d.lon0.between(-180,360).all()):raise ValueError('Invalid initial coordinates.')
    for row in d.itertuples():
        if utc(row.init_time)!=times_for_init(row.stamp)[1]:raise ValueError('Stamp/init_time mismatch.')
    return d

def normalise_track(track,sid,init_time,model_id=MODEL):
    """Only returns the public forecast interface, never any future truth."""
    if not {'time','lat','lon'}<=set(track):raise ValueError('Tracker result schema changed.')
    out=track.rename(columns={'lat':'base_lat','lon':'base_lon'}).copy()
    valid=pd.to_datetime(out['time'],utc=True)
    hours=(valid-utc(init_time)).dt.total_seconds()/3600
    if not np.allclose(hours,np.round(hours)):raise ValueError('Non-integer tracker lead.')
    out['lead_h']=hours.astype(int)
    out=out[out.lead_h>0].copy()
    out['sid']=sid;out['init_time']=utc(init_time).tz_localize(None).strftime('%Y-%m-%d %H:%M:%S')
    out['model_id']=model_id
    out['base_lon']=(out.base_lon+180)%360-180
    return out
