import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pipeline import xy,latlon,distance,parse_enso,enso_at,split_year,ROOT,prepare,external,fresh
import json

class CoreTests(unittest.TestCase):
    def test_roundtrip_dateline(self):
        lat=np.array([0.,20.,-10.,50.]);lon=np.array([179.,120.,-179.,0.])
        target_lat=lat+np.array([1.,4.,-3.,0.]);target_lon=np.array([-179.,135.,179.,0.])
        offsets=xy(lat,lon,target_lat,target_lon)
        a,b=latlon(lat,lon,offsets)
        np.testing.assert_allclose(a,target_lat,atol=1e-8)
        np.testing.assert_allclose(b,target_lon,atol=1e-8)
        np.testing.assert_allclose(distance(lat,lon,lat,lon),0,atol=1e-8)

    def test_month_lag(self):
        import pandas as pd
        values={pd.Period('2023-10',freq='M'):1.,pd.Period('2023-11',freq='M'):2.,pd.Period('2023-12',freq='M'):3.,pd.Period('2024-01',freq='M'):99.}
        self.assertEqual(enso_at('2024-02-01',values,2),[3.,2.,1.,1.])

    def test_psl_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'enso.txt'
            p.write_text('2000 2000\n2000 '+' '.join(['1']*11+['-99.99'])+'\n-99.99\n')
            self.assertEqual(len(parse_enso(p)),11)

    def test_splits(self):
        cfg=json.loads((ROOT/'config.json').read_text())
        self.assertEqual([split_year(y,cfg) for y in [1990,2005,2016,2022]],['base','patch','val','test'])

    def test_prepare_and_external(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);raw=root/'raw';run=root/'run';raw.mkdir()
            cfg=json.loads((ROOT/'config.json').read_text())
            records=['SID,ISO_TIME,USA_LAT,USA_LON',' , ,degrees_north,degrees_east']
            for year in [1990,2005,2016,2022]:
                for k,t in enumerate(pd.date_range(f'{year}-08-01',periods=30,freq='6h')):
                    records.append(f'{year}001N10000,{t},'+str(10+k*.1)+','+str(170+k*.1))
            (raw/'ibtracs.WP.list.v04r01.csv').write_text('\n'.join(records))
            (raw/'nina34.anom.data').write_text('1948 2024\n'+'\n'.join(str(y)+' '+' '.join(['1.0']*12) for y in [1990,2005,2016,2022]))
            prepare(raw,run,cfg)
            d=pd.read_csv(run/'samples.csv')
            self.assertEqual(set(d.split),{'base','patch','val','test'})
            self.assertEqual(d.groupby('sid').split.nunique().max(),1)
            self.assertTrue(np.all(d.enso==1.))
            f=d[['sid','init_time','lead_h','lat0','lon0']].rename(columns={'lat0':'base_lat','lon0':'base_lon'})
            f['model_id']='fixture_only';f.to_csv(root/'external.csv',index=False)
            external(run,root/'external.csv','fixture_only')
            joined=pd.read_csv(run/'forecasts.csv')
            self.assertEqual(len(joined),len(d))
            np.testing.assert_allclose(joined[['base_x','base_y']],0,atol=1e-8)

    def test_protect_existing(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'keep.txt';path.write_text('original')
            with self.assertRaises(FileExistsError):fresh(path)
            self.assertEqual(path.read_text(),'original')

if __name__=='__main__':unittest.main()
