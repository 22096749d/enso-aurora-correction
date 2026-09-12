import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import times_for_init,normalise_track,MODEL,digest,write_json
from exchange import plan,verify_plan,collect,verify_return,digest_subset

class ExchangeTests(unittest.TestCase):
    def test_utc_pair_crosses_year(self):
        older,current=times_for_init('2023010100')
        self.assertEqual(str(older),'2022-12-31 18:00:00+00:00')
        self.assertEqual((current-older).total_seconds(),21600)

    def test_tracker_interface(self):
        d=pd.DataFrame({'time':['2023-08-01 00:00:00','2023-08-02 00:00:00'],
                        'lat':[20,21],'lon':[359,1],'msl':[None,99000]})
        result=normalise_track(d,'s1','2023-08-01')
        self.assertEqual(result.lead_h.tolist(),[24])
        self.assertEqual(result.model_id.iloc[0],MODEL)

    def test_plan_return_and_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'run';run.mkdir()
            rows=[]
            for year,split in [(2018,'patch'),(2021,'val'),(2024,'test')]:
                for day in [1,2,3]:
                    rows.append({'sid':f'{year}001N20000','init_time':f'{year}-08-{day:02d} 00:00:00',
                        'lat0':20.,'lon0':130.,'split':split,'truth_lat':99,'truth_lon':99})
            pd.DataFrame(rows).to_csv(run/'samples.csv',index=False)
            pl=root/'plan';plan(run,pl,2);verify_plan(pl)
            d=pd.read_csv(pl/'plan.csv',dtype={'stamp':str})
            self.assertNotIn('truth_lat',d.columns)
            self.assertEqual(len(d),6)
            for stamp,g in d.groupby('stamp'):
                folder=root/'server'/stamp;folder.mkdir(parents=True)
                rows=[]
                for row in g.itertuples():
                    rows.append(dict(sid=row.sid,init_time=row.init_time,lead_h=24,model_id=MODEL,
                        base_lat=21,base_lon=129,tracker_fallback=False))
                pd.DataFrame(rows).to_csv(folder/'forecasts.csv',index=False)
                write_json(folder/'done.json',{'plan_subset_sha256':digest_subset(g),
                    'forecasts_sha256':digest(folder/'forecasts.csv'),'identity':{'revision':'fixture'}})
            collect(pl,root/'server',root/'returned');verify_return(root/'returned')
            self.assertEqual(len(pd.read_csv(root/'returned/forecasts.csv')),6)
            with (root/'returned/forecasts.csv').open('a') as f:f.write('CORRUPTED')
            with self.assertRaises(ValueError):verify_return(root/'returned')

if __name__=='__main__':unittest.main()
