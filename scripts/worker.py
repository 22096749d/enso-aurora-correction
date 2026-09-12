"""Colleague runner: cumulative download horizon and bounded new GPU initializations."""
import argparse
import subprocess
import sys
from datetime import datetime,timezone
from common import ROOT,read_plan,read_json
from exchange import verify_plan

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['data','run'])
    p.add_argument('--job',default='pilot')
    p.add_argument('--through',type=int,default=1,help='First N unique UTC initializations, cumulatively')
    a=p.parse_args()
    if not a.job.replace('_','').replace('-','').isalnum() or a.through<1:p.error('Invalid job or through')
    directory=ROOT/'jobs'/a.job;verify_plan(directory)
    stamps=sorted(read_plan(directory/'plan.csv').stamp.unique())
    n=min(a.through,len(stamps))
    out=ROOT/'server_runs'/a.job
    if a.command=='run':
        # Check all earlier completions are a contiguous prefix, not arbitrary skipped cases.
        done=[(out/s/'done.json').exists() for s in stamps]
        if any(done[n:]):raise ValueError('through is behind completed work; use a larger cumulative horizon')
        count=sum(done[:n])
        if done[:n]!=[True]*count+[False]*(n-count):raise ValueError('Non-contiguous completion; inspect output before resuming')
        if count==n:
            print('Requested prefix already complete');return
        args=['run','--output',str(out),'--steps','20','--limit',str(n-count)]
    else:args=['data','--limit',str(n)]
    logdir=ROOT/'logs'/a.job;logdir.mkdir(parents=True,exist_ok=True)
    log=logdir/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'_'+a.command+'.log')
    command=[sys.executable,'-u',str(ROOT/'scripts/server.py'),*args,'--plan',str(directory/'plan.csv')]
    print('Running:',command,flush=True)
    with log.open('w',encoding='utf-8') as f:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in proc.stdout:
            print(line,end='',flush=True);f.write(line);f.flush()
        code=proc.wait()
    print('Log:',log)
    if code:raise SystemExit(code)

if __name__=='__main__':main()
