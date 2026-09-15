"""Research-only track postprocessing. No operational warning capability.
Commands use paths relative to the project root, not the terminal cwd.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
R = 6371.0088
KEY = ['sid', 'init_time', 'lead_h']
BASE = ['lat0', 'sinlon', 'coslon', 'dx6', 'dy6', 'dx12', 'dy12',
        'dx24', 'dy24', 'month_sin', 'month_cos']
ENSO = ['enso', 'enso_prev', 'enso_prev2', 'enso_trend']
URLS = {
    'ibtracs.WP.list.v04r01.csv': 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.WP.list.v04r01.csv',
    'nina34.anom.data': 'https://psl.noaa.gov/data/correlation/nina34.anom.data'
}

def validate_download(name, path):
    """Cheap guards against an HTTP response that ended early but raised no error."""
    size = path.stat().st_size
    if name.startswith('ibtracs.'):
        if size < 50_000_000:
            raise ValueError(f'IBTrACS download is truncated: {size} bytes; expected a large WP CSV '
                             '(currently about 114 MB). Archive this exact file and retry.')
        with path.open('rb') as f:
            header = f.readline()
        if not (b'SID' in header and b'ISO_TIME' in header and b'USA_LAT' in header and b'USA_LON' in header):
            raise ValueError('IBTrACS header is invalid; the response may be an HTML error page.')
    else:
        if size < 3_000:
            raise ValueError(f'ENSO download is unexpectedly small: {size} bytes.')
        parse_enso(path)

def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def fresh(p):
    if p.exists():
        raise FileExistsError(f'{p} already exists. Use a new --run or archive it manually.')
    p.parent.mkdir(parents=True, exist_ok=True)

def xy(lat0, lon0, lat, lon):
    """Spherical azimuthal-equidistant log map: east/north km."""
    a, b = np.radians(lat0), np.radians(lat)
    d = np.radians((np.asarray(lon)-lon0+180) % 360-180)
    east = np.cos(b)*np.sin(d)
    north = np.cos(a)*np.sin(b)-np.sin(a)*np.cos(b)*np.cos(d)
    sine = np.hypot(east, north)
    cosine = np.sin(a)*np.sin(b)+np.cos(a)*np.cos(b)*np.cos(d)
    angle = np.arctan2(sine, cosine)
    factor = np.divide(R*angle, sine, out=np.full_like(sine, R, dtype=float), where=sine>1e-12)
    return np.stack([east*factor, north*factor], axis=-1)

def latlon(lat0, lon0, offsets):
    x, y = np.asarray(offsets).T
    distance = np.hypot(x, y)
    c, a = distance/R, np.radians(lat0)
    direction = np.arctan2(x, y)
    lat = np.arcsin(np.clip(np.sin(a)*np.cos(c)+np.cos(a)*np.sin(c)*np.cos(direction), -1, 1))
    lon = np.radians(lon0)+np.arctan2(np.sin(direction)*np.sin(c)*np.cos(a), np.cos(c)-np.sin(a)*np.sin(lat))
    return np.degrees(lat), (np.degrees(lon)+180) % 360-180

def distance(lat1, lon1, lat2, lon2):
    return np.linalg.norm(xy(lat1, lon1, lat2, lon2), axis=-1)

def parse_enso(p):
    values = {}
    for line in p.read_text(encoding='utf-8-sig').splitlines():
        bits = line.split()
        if len(bits) != 13 or not bits[0].isdigit():
            continue
        year = int(bits[0])
        if not 1850 <= year <= 2100:
            continue
        for month, text in enumerate(bits[1:], 1):
            value = float(text)
            if abs(value) < 10:
                values[pd.Period(year=year, month=month, freq='M')] = value
    if not values:
        raise ValueError('ENSO file is not a valid PSL year + 12 monthly values file.')
    return values

def enso_at(date, values, lag):
    # A two-month lag is conservative, NOT a reconstructed release-time archive.
    month = pd.Timestamp(date).to_period('M')-lag
    e = [values.get(month-i, np.nan) for i in range(3)]
    return e+[e[0]-e[1]]

def split_year(y, cfg):
    if cfg['first_year'] <= y <= cfg['base_end']: return 'base'
    if cfg['base_end'] < y <= cfg['patch_end']: return 'patch'
    if cfg['patch_end'] < y <= cfg['val_end']: return 'val'
    if cfg['val_end'] < y <= cfg['test_end']: return 'test'
    return 'exclude'

def download(raw):
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        target = raw/name
        if target.exists():
            validate_download(name,target)
            print('Keeping existing file:', target)
            continue
        print('Downloading', url, flush=True)
        temporary = target.with_suffix(target.suffix+'.part')
        if temporary.exists():
            raise FileExistsError(f'Incomplete download found: {temporary}. Inspect/rename it before retrying.')
        request = urllib.request.Request(url, headers={'User-Agent': 'ENSO-research/1.0'})
        received = 0
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open('wb') as f:
            expected = response.headers.get('Content-Length')
            while True:
                data = response.read(1048576)
                if not data: break
                f.write(data)
                received += len(data)
        if expected is not None and received != int(expected):
            raise IOError(f'Incomplete HTTP response for {name}: received {received}, expected {expected}. '
                          f'The partial file remains at {temporary}.')
        validate_download(name,temporary)
        temporary.rename(target)
        save_json(raw/(name+'.download.json'), {'url': url, 'retrieved_utc': pd.Timestamp.now(tz='UTC'), 'sha256': sha(target)})
    print('Downloaded files retained unchanged. Next: prepare')

def prepare(raw, run, cfg):
    fresh(run/'samples.csv')
    source = raw/'ibtracs.WP.list.v04r01.csv'
    data = pd.read_csv(source, skiprows=[1], dtype=str, low_memory=False,
                       usecols=lambda c: c.strip() in ['SID','ISO_TIME','USA_LAT','USA_LON'])
    data.columns = data.columns.str.strip()
    cols = ['SID', 'ISO_TIME', 'USA_LAT', 'USA_LON']
    if not set(cols).issubset(data.columns):
        raise ValueError('Expected IBTrACS v04r01 CSV with USA_LAT and USA_LON; do not rename another agency.')
    data = data[cols].rename(columns=dict(zip(cols, ['sid','time','lat','lon'])))
    raw_rows = len(data)
    data['sid'] = data.sid.str.strip()
    # Recent pandas versions infer one timestamp format for a whole Series.  IBTrACS
    # releases can contain more than one ISO representation, so parse each format.
    try:
        data['time'] = pd.to_datetime(data.time.str.strip(), errors='coerce', format='mixed')
    except TypeError:  # Compatibility with older pandas.
        data['time'] = pd.to_datetime(data.time.str.strip(), errors='coerce')
    for c in ['lat','lon']: data[c] = pd.to_numeric(data[c], errors='coerce')
    data = data.dropna().sort_values(['sid','time'])
    data = data[data.lat.between(-90,90) & data.lon.between(-180,360)]
    data['lon'] = (data.lon+180) % 360-180
    data = data[(data.time.dt.hour % 6 == 0) & (data.time.dt.minute == 0) & (data.time.dt.second == 0)]
    six_hour_rows = len(data)
    if data.empty:
        raise ValueError(f'IBTrACS has no usable USA 6-hour positions (raw rows={raw_rows}). '
                         'Check that the downloaded file is the WP v04r01 CSV, not an HTML page.')
    if data.duplicated(['sid','time']).any():
        raise ValueError('Duplicate SID/time found; inspect raw track rows before proceeding.')
    values = parse_enso(raw/'nina34.anom.data')
    rows = []
    audit = dict(raw_rows=raw_rows, usa_valid_6h_rows=six_hour_rows,
                 included_storms=0, candidate_initializations=0,
                 with_required_history=0, with_enso=0, with_any_future_target=0)
    for sid, group in data.groupby('sid', sort=False):
        year = int(str(sid)[:4])  # IBTrACS SID starts with storm genesis year.
        split = split_year(year, cfg)
        if split == 'exclude': continue
        audit['included_storms'] += 1
        g = group.set_index('time')
        for t, now in g.iterrows():
            # Avoid crossing calendar split boundaries; no interpolation from future fixes.
            if split_year(t.year,cfg) != split: continue
            audit['candidate_initializations'] += 1
            # Only require history actually used by BASE (6, 12, and 24 h).
            history = [t-pd.Timedelta(hours=h) for h in [6,12,24]]
            if not all(k in g.index for k in history): continue
            audit['with_required_history'] += 1
            enso = enso_at(t,values,cfg['enso_lag_months'])
            if not np.isfinite(enso).all(): continue
            audit['with_enso'] += 1
            feats = dict(lat0=now.lat,lon0=now.lon,sinlon=np.sin(np.radians(now.lon)),coslon=np.cos(np.radians(now.lon)))
            for h in [6,12,24]:
                old = g.loc[t-pd.Timedelta(hours=h)]
                delta = -xy(now.lat, now.lon, old.lat, old.lon)
                feats.update({f'dx{h}':delta[0], f'dy{h}':delta[1]})
            feats.update(month_sin=np.sin(2*np.pi*t.month/12),month_cos=np.cos(2*np.pi*t.month/12))
            feats.update(dict(zip(ENSO,enso)))
            before = len(rows)
            for lead in cfg['leads']:
                valid = t+pd.Timedelta(hours=lead)
                if valid not in g.index or split_year(valid.year,cfg) != split: continue
                truth = g.loc[valid]
                delta = xy(now.lat,now.lon,truth.lat,truth.lon)
                rows.append(dict(sid=sid,year=year,init_time=t,valid_time=valid,lead_h=lead,split=split,
                    truth_lat=truth.lat,truth_lon=truth.lon,target_x=delta[0],target_y=delta[1],**feats))
            if len(rows) > before: audit['with_any_future_target'] += 1
    samples = pd.DataFrame(rows)
    print('PREPARE AUDIT:', json.dumps(audit, ensure_ascii=False))
    if samples.empty:
        raise ValueError('No usable samples. Read PREPARE AUDIT above: zero usa_valid_6h_rows means '
                         'the USA coordinates/time failed; zero with_required_history means track '
                         'continuity failed; zero with_enso means ENSO dates failed; zero '
                         'with_any_future_target means future track points failed.')
    samples.to_csv(run/'samples.csv',index=False)
    counts = samples.groupby(['split','lead_h']).agg(rows=('sid','size'),storms=('sid','nunique'),years=('year','nunique'))
    counts.to_csv(run/'sample_counts.csv')
    save_json(run/'data_manifest.json', {'config':cfg,'raw':[{'file':str(raw/n),'sha256':sha(raw/n)} for n in URLS],
        'enso_footer':(raw/'nina34.anom.data').read_text().splitlines()[-10:],
        'position_agency':'USA/JTWC','purpose':'retrospective best-track-input experiment; not operational reforecast',
        'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__})
    print(counts.to_string())

def frame(run, name):
    return pd.read_csv(run/name)

def ridge(alpha): return make_pipeline(StandardScaler(),Ridge(alpha=alpha))

def base(run,cfg):
    fresh(run/'forecasts.csv')
    d = frame(run,'samples.csv')
    models = {}
    for lead in cfg['leads']:
        ix = d.lead_h == lead
        train = ix & (d.split == 'base')
        if train.sum()<20: raise ValueError(f'Not enough base training samples at {lead} h.')
        m = ridge(100.)
        m.fit(d.loc[train,BASE],d.loc[train,['target_x','target_y']])
        models[lead]=m
        d.loc[ix,['base_x','base_y']]=m.predict(d.loc[ix,BASE])
    d['base_lat'], d['base_lon']=latlon(d.lat0.to_numpy(),d.lon0.to_numpy(),d[['base_x','base_y']].to_numpy())
    d['model_id']='track_ridge_frozen'
    d.to_csv(run/'forecasts.csv',index=False)
    joblib.dump(models,run/'base_models.joblib')
    print('Frozen baseline saved. Patch training will not use base-period rows.')

def external(run,path,model_id):
    fresh(run/'forecasts.csv')
    d=frame(run,'samples.csv')
    f=pd.read_csv(path,dtype={'sid':str,'model_id':str})
    required=KEY+['model_id','base_lat','base_lon']
    if not set(required).issubset(f): raise ValueError(f'External CSV needs {required}')
    f['sid']=f.sid.str.strip();d['sid']=d.sid.astype(str).str.strip()
    f['model_id']=f.model_id.str.strip()
    f=f[f.model_id==model_id].copy()
    # Never merge timestamps through dtype-dependent string formatting.  Pandas
    # may render an all-midnight Series as YYYY-MM-DD while another Series uses
    # YYYY-MM-DD HH:MM:SS although they identify the same UTC instant.
    f_time=pd.to_datetime(f.init_time,utc=True,errors='coerce')
    d_time=pd.to_datetime(d.init_time,utc=True,errors='coerce')
    if f_time.isna().any() or d_time.isna().any():raise ValueError('Unparseable initialization time.')
    f['_init_ns']=f_time.astype('int64');d['_init_ns']=d_time.astype('int64')
    f['lead_h']=pd.to_numeric(f.lead_h,errors='coerce')
    d['lead_h']=pd.to_numeric(d.lead_h,errors='coerce')
    if f.lead_h.isna().any() or d.lead_h.isna().any():raise ValueError('Non-numeric lead_h.')
    f['lead_h']=f.lead_h.astype('int64');d['lead_h']=d.lead_h.astype('int64')
    merge_key=['sid','_init_ns','lead_h']
    if f.empty or f.duplicated(merge_key).any(): raise ValueError('Empty model subset or duplicate normalized forecast keys.')
    if not (f.base_lat.between(-90,90).all() and f.base_lon.between(-180,360).all()):
        raise ValueError('Invalid forecast coordinates.')
    f['base_lon']=(f.base_lon+180)%360-180
    right=['sid','_init_ns','lead_h','model_id','base_lat','base_lon']
    joined=d.merge(f[right],on=merge_key,how='inner',validate='one_to_one')
    if joined.empty:
        sid_overlap=len(set(d.sid)&set(f.sid))
        pair_overlap=len(set(zip(d.sid,d._init_ns))&set(zip(f.sid,f._init_ns)))
        key_overlap=len(set(zip(d.sid,d._init_ns,d.lead_h))&set(zip(f.sid,f._init_ns,f.lead_h)))
        raise ValueError(f'No matching normalized keys: SID={sid_overlap}, SID+time={pair_overlap}, full={key_overlap}.')
    audit=d.groupby(['split','lead_h']).size().rename('eligible').to_frame()
    audit['matched']=joined.groupby(['split','lead_h']).size()
    audit['matched']=audit.matched.fillna(0).astype(int)
    audit['coverage']=audit.matched/audit.eligible
    audit.to_csv(run/'external_coverage.csv')
    joined[['base_x','base_y']]=xy(joined.lat0.to_numpy(),joined.lon0.to_numpy(),joined.base_lat.to_numpy(),joined.base_lon.to_numpy())
    joined=joined.drop(columns=['_init_ns'])
    joined.to_csv(run/'forecasts.csv',index=False)
    save_json(run/'external_manifest.json',{'source':str(path),'sha256':sha(path),'model_id':model_id,
        'warning':'User must document backbone training dates, forecast initialization and tracker failures.'})
    print(audit.to_string())

def features(d, use_enso):
    cols=BASE+['base_x','base_y']+(ENSO if use_enso else [])
    return d[cols]

def corrected_error(d,p):
    a,b=latlon(d.lat0.to_numpy(),d.lon0.to_numpy(),p)
    return distance(a,b,d.truth_lat.to_numpy(),d.truth_lon.to_numpy())

def fit_patch(run,cfg):
    fresh(run/'patch_models.joblib')
    d=frame(run,'forecasts.csv')
    expected=d.model_id.unique()
    if len(expected)!=1: raise ValueError('Train separately for each base model.')
    bundle={'models':{},'model_id':expected[0],'cfg':cfg,'features':BASE,'versions':{'sklearn':sklearn.__version__}}
    scores=[]
    for lead in cfg['leads']:
        tr=d[(d.split=='patch') & (d.lead_h==lead)]
        va=d[(d.split=='val') & (d.lead_h==lead)]
        if min(len(tr),len(va))<20: raise ValueError(f'Need >=20 patch/validation rows for {lead} h; inspect coverage.')
        target=tr[['target_x','target_y']].to_numpy()-tr[['base_x','base_y']].to_numpy()
        offset=target.mean(axis=0)
        bundle['models'][(lead,'mean')]=(None,1.,False,offset)
        for family in ['ridge','tree']:
            for use in [False,True]:
                name=family+('_enso' if use else '_noenso')
                best=(float('inf'),None)
                for parameter in ([10.,100.,1000.] if family=='ridge' else [7,15]):
                    m=ridge(parameter) if family=='ridge' else MultiOutputRegressor(HistGradientBoostingRegressor(
                        max_leaf_nodes=parameter,max_iter=cfg['tree_iterations'],learning_rate=.05,
                        min_samples_leaf=30,l2_regularization=10.,early_stopping=False,random_state=cfg['seed']))
                    m.fit(features(tr,use),target)
                    delta=m.predict(features(va,use))
                    for shrink in [0.,.25,.5,1.]:
                        pred=va[['base_x','base_y']].to_numpy()+shrink*delta
                        error=float(corrected_error(va,pred).mean())
                        scores.append(dict(lead_h=lead,method=name,parameter=parameter,shrink=shrink,val_error_km=error))
                        if error<best[0]: best=(error,(m,shrink,use,np.zeros(2)))
                bundle['models'][(lead,name)]=best[1]
                print(lead,name,'validation km=',round(best[0],2),'shrink=',best[1][1],flush=True)
    joblib.dump(bundle,run/'patch_models.joblib')
    pd.DataFrame(scores).to_csv(run/'validation_search.csv',index=False)

def patch_predict(d,bundle,method):
    result=np.empty((len(d),2))
    for lead in d.lead_h.unique():
        ix=np.flatnonzero(d.lead_h.to_numpy()==lead)
        model,shrink,use,offset=bundle['models'][(int(lead),method)]
        sub=d.iloc[ix]
        delta=offset if model is None else shrink*model.predict(features(sub,use))
        result[ix]=sub[['base_x','base_y']].to_numpy()+delta
    return result

def evaluate(run,cfg):
    fresh(run/'test_metrics.csv')
    d=frame(run,'forecasts.csv')
    d=d[d.split=='test'].copy().reset_index(drop=True)
    if d.empty: raise ValueError('No test rows.')
    bundle=joblib.load(run/'patch_models.joblib')
    methods=['base','mean','ridge_noenso','ridge_enso','tree_noenso','tree_enso']
    output=[]
    for name in methods:
        p=d[['base_x','base_y']].to_numpy() if name=='base' else patch_predict(d,bundle,name)
        out=d[KEY+['year','truth_lat','truth_lon','enso']].copy()
        out['method']=name
        out['pred_lat'],out['pred_lon']=latlon(d.lat0.to_numpy(),d.lon0.to_numpy(),p)
        out['error_km']=corrected_error(d,p)
        # Simple lagged monthly anomaly bins, not official ONI event labels.
        out['enso_bin']=np.where(out.enso>=.5,'warm',np.where(out.enso<=-.5,'cold','neutral'))
        output.append(out)
    result=pd.concat(output,ignore_index=True)
    result.to_csv(run/'test_predictions.csv',index=False)
    def summarize(group):
        return dict(n=len(group),storms=group.sid.nunique(),years=group.year.nunique(),
            mean_km=group.error_km.mean(),median_km=group.error_km.median(),p90_km=group.error_km.quantile(.9))
    metrics=[]
    for (method,lead),g in result.groupby(['method','lead_h']):
        metrics.append(dict(method=method,lead_h=lead,enso_bin='all',**summarize(g)))
        for category,sub in g.groupby('enso_bin'):
            metrics.append(dict(method=method,lead_h=lead,enso_bin=category,**summarize(sub)))
    pd.DataFrame(metrics).to_csv(run/'test_metrics.csv',index=False)
    rng=np.random.default_rng(cfg['seed'])
    comparisons=[]
    pairs=[('base',m) for m in methods[1:]]+[('ridge_noenso','ridge_enso'),('tree_noenso','tree_enso')]
    for lead in cfg['leads']:
        table=result[result.lead_h==lead].pivot(index=KEY+['year'],columns='method',values='error_km').reset_index()
        if table.empty: continue
        for reference,candidate in pairs:
            table['gain']=table[reference]-table[candidate]
            annual=table.groupby('year').gain.agg(['sum','count']).to_numpy()
            indices=rng.integers(0,len(annual),(cfg['bootstrap'],len(annual)))
            sampled=annual[indices].sum(axis=1)
            boot=sampled[:,0]/sampled[:,1]
            lo,hi=np.quantile(boot,[.025,.975])
            comparisons.append(dict(lead_h=lead,reference=reference,candidate=candidate,
                gain_km=table.gain.mean(),ci_low=lo,ci_high=hi,years=len(annual),
                skill_percent=100*table.gain.mean()/max(table[reference].mean(),1e-9)))
    pd.DataFrame(comparisons).to_csv(run/'paired_year_bootstrap.csv',index=False)
    print(pd.DataFrame(metrics).query("enso_bin=='all'").to_string(index=False))
    print('Positive gain_km = improvement. Few independent test years => exploratory uncertainty only.')

def apply_new(run,input_path,output_path,method):
    fresh(output_path)
    d=pd.read_csv(input_path)
    bundle=joblib.load(run/'patch_models.joblib')
    needed=KEY+BASE+['lon0','base_lat','base_lon','model_id']+ENSO
    missing=set(needed)-set(d)
    if missing: raise ValueError(f'Inference input missing {sorted(missing)}; see inference_input.csv generated by smoke.')
    if d.duplicated(KEY).any(): raise ValueError('Duplicate forecast keys.')
    if not d.model_id.eq(bundle['model_id']).all(): raise ValueError('Base-model mismatch. Retrain a model-specific patch.')
    numeric=BASE+['lon0','base_lat','base_lon']+ENSO
    if not np.isfinite(d[numeric].to_numpy(dtype=float)).all(): raise ValueError('Missing/nonfinite inference features.')
    if not (d.lat0.between(-90,90).all() and d.base_lat.between(-90,90).all() and
            d.lon0.between(-180,360).all() and d.base_lon.between(-180,360).all()):
        raise ValueError('Inference coordinates out of bounds.')
    if not d.lead_h.isin(bundle['cfg']['leads']).all(): raise ValueError('Unsupported lead time.')
    d[['base_x','base_y']]=xy(d.lat0.to_numpy(),d.lon0.to_numpy(),d.base_lat.to_numpy(),d.base_lon.to_numpy())
    pred=patch_predict(d,bundle,method)
    d['corrected_lat'],d['corrected_lon']=latlon(d.lat0.to_numpy(),d.lon0.to_numpy(),pred)
    d.to_csv(output_path,index=False)
    print(output_path)

def smoke(run,cfg):
    """Synthetic engineering test ONLY: generated data cannot support scientific claims."""
    fresh(run/'samples.csv')
    rng=np.random.default_rng(42)
    rows=[]
    for year in [1990,1992,2004,2008,2016,2017,2020,2021,2023]:
        for storm in range(24):
            lat0,lon0=15+rng.normal(0,3),140+rng.normal(0,5)
            e=np.sin(year)+rng.normal(0,.1)
            f=dict(zip(BASE,rng.normal(size=len(BASE))))
            f.update(lat0=lat0,lon0=lon0,sinlon=np.sin(np.radians(lon0)),coslon=np.cos(np.radians(lon0)))
            f.update(dict(zip(ENSO,[e,e-.1,e-.2,.1])))
            for lead in cfg['leads']:
                target=np.array([-lead*7,lead*2])+rng.normal(0,20,2)+[e*lead,0]
                la,lo=latlon(np.array([lat0]),np.array([lon0]),target[None,:])
                rows.append(dict(sid=f'{year}{storm:03d}N00000',year=year,init_time=f'{year}-08-01 00:00:00',lead_h=lead,
                    split=split_year(year,cfg),target_x=target[0],target_y=target[1],truth_lat=la[0],truth_lon=lo[0],**f))
    pd.DataFrame(rows).to_csv(run/'samples.csv',index=False)
    save_json(run/'SYNTHETIC_ONLY.json',{'warning':'Not real weather data. Engineering test only.'})
    base(run,cfg); fit_patch(run,cfg); evaluate(run,cfg)
    d=frame(run,'forecasts.csv').query("split=='test'").head(5)
    d[KEY+BASE+['lon0','base_lat','base_lon','model_id']+ENSO].to_csv(run/'inference_input.csv',index=False)
    apply_new(run,run/'inference_input.csv',run/'inference_output.csv','ridge_enso')
    print('SMOKE PASS: training, validation, held-out testing and label-free inference.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['download','prepare','base','external','train','test','apply','smoke'])
    p.add_argument('--run',default='runs/phase1_v1')
    p.add_argument('--raw',default='data/raw')
    p.add_argument('--config',default='config.json')
    p.add_argument('--input'); p.add_argument('--output'); p.add_argument('--model-id')
    p.add_argument('--method',choices=['mean','ridge_noenso','ridge_enso','tree_noenso','tree_enso'],default='ridge_enso')
    args=p.parse_args()
    cfg=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    boundaries=[cfg[k] for k in ['first_year','base_end','patch_end','val_end','test_end']]
    if boundaries != sorted(set(boundaries)): raise ValueError('Year boundaries must be strictly increasing.')
    if cfg['enso_lag_months']<1: raise ValueError('Never use a full current-month ENSO mean.')
    run,raw=ROOT/args.run,ROOT/args.raw
    command=args.command
    if (run/'data_manifest.json').exists() and command in ['base','external','train','test']:
        recorded=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'))['config']
        if recorded!=cfg:
            raise ValueError('Configuration differs from prepare stage. Use the same --config or a new run.')
    if command=='download': download(raw)
    elif command=='prepare': prepare(raw,run,cfg)
    elif command=='base': base(run,cfg)
    elif command=='external':
        if not args.input or not args.model_id: p.error('external requires --input and --model-id')
        external(run,ROOT/args.input,args.model_id)
    elif command=='train': fit_patch(run,cfg)
    elif command=='test': evaluate(run,cfg)
    elif command=='apply':
        if not args.input or not args.output: p.error('apply requires --input and --output')
        apply_new(run,ROOT/args.input,ROOT/args.output,args.method)
    else:
        cfg['tree_iterations']=10; cfg['bootstrap']=100
        smoke(run,cfg)

if __name__=='__main__': main()
