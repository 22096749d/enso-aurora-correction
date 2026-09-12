"""Export one deterministic historical case without targets; plot held-out paths."""
import argparse
import html
import numpy as np
import pandas as pd
from pipeline import ROOT,KEY,BASE,ENSO,fresh

def select(d):
    # Pick first chronologically complete forecast, not greatest improvement.
    counts=d.groupby(['sid','init_time']).lead_h.nunique()
    eligible=counts[counts==5].reset_index()[['sid','init_time']]
    if eligible.empty: raise ValueError('No complete 5-lead case.')
    first=eligible.sort_values(['init_time','sid']).iloc[0]
    return d[(d.sid==first.sid)&(d.init_time==first.init_time)].sort_values('lead_h')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['export','plot'])
    p.add_argument('--run',default='runs/phase1_v1');p.add_argument('--output',required=True)
    a=p.parse_args();run=ROOT/a.run;out=ROOT/a.output
    fresh(out)
    if a.command=='export':
        d=pd.read_csv(run/'forecasts.csv')
        d=select(d[d.split=='test'])
        d[KEY+BASE+['lon0','base_lat','base_lon','model_id']+ENSO].to_csv(out,index=False)
    else:
        d=select(pd.read_csv(run/'test_predictions.csv'))
        paths={}
        for method in ['base','ridge_enso','tree_enso']:
            z=d[d.method==method].sort_values('lead_h')
            paths[method]=(z.pred_lon.to_numpy(),z.pred_lat.to_numpy())
        z=d[d.method=='base'].sort_values('lead_h')
        paths['truth']=(z.truth_lon.to_numpy(),z.truth_lat.to_numpy())
        # Unwrap all paths relative to first truth position around the dateline.
        anchor=paths['truth'][0][0]
        paths={k:(anchor+(x-anchor+180)%360-180,y) for k,(x,y) in paths.items()}
        points=np.concatenate([np.column_stack(v) for v in paths.values()])
        lo=points.min(axis=0)-1;hi=points.max(axis=0)+1
        colors={'base':'#dc2626','ridge_enso':'#2563eb','tree_enso':'#059669','truth':'#111827'}
        svg=['<rect width="800" height="520" fill="white"/>']
        for name,(xs,ys) in paths.items():
            px=60+(xs-lo[0])/(hi[0]-lo[0])*680;py=450-(ys-lo[1])/(hi[1]-lo[1])*390
            poly=' '.join(f'{x:.1f},{y:.1f}' for x,y in zip(px,py))
            svg.append(f'<polyline points="{poly}" fill="none" stroke="{colors[name]}" stroke-width="2"/>')
            for x,y,lead in zip(px,py,z.lead_h):
                svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{colors[name]}"/><text x="{x+5}" y="{y-6}" font-size="11">{lead}h</text>')
        title=html.escape(str(d.sid.iloc[0])+' / '+str(d.init_time.iloc[0])+' UTC')
        legend=' | '.join(f'<span style="color:{v}">{k}</span>' for k,v in colors.items())
        out.write_text(f'<!doctype html><meta charset="utf-8"><title>Track inspection</title><h2>{title}</h2><p>{legend}</p>'
            f'<p>经纬度示意图：向右为东、向上为北；无海岸线、非等距离投影。点为24至120小时。</p>'
            f'<svg viewBox="0 0 800 520" width="800" role="img" aria-label="Forecast track comparison">'+''.join(svg)+'</svg>'
            '<p>该案例按时间选取，不代表总体改善。完整指标见test_metrics.csv。</p>',encoding='utf-8')
    print(out)

if __name__=='__main__':main()
