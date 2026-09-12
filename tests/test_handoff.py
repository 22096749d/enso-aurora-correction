import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import handoff
from common import digest,write_json,MODEL
from exchange import plan,digest_subset

class HandoffTests(unittest.TestCase):
    def setup_project(self,root):
        run=root/'runs/samples';run.mkdir(parents=True)
        pd.DataFrame([dict(sid='2020001N20000',init_time='2020-08-01 00:00:00',
                           lat0=20.,lon0=130.,split='patch',truth_lat=25.)]).to_csv(run/'samples.csv',index=False)
        plan(run,root/'jobs/pilot',1)
        return run

    def test_all_failed_still_returns_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=self.setup_project(root)
            out=root/'server_runs/pilot';write_json(out/'2020080100/error.json',{'error':'fixture'})
            with patch.object(handoff,'ROOT',root):
                handoff.bundle('pilot',out,root/'returns/v1')
                self.assertTrue((root/'returns/v1.zip').exists())
                self.assertTrue((root/'returns/v1/diagnostics/2020080100/error.json').exists())
                with self.assertRaisesRegex(ValueError,'Diagnosis only'):
                    handoff.accept(root/'returns/v1','pilot',run)

    def test_success_identity_and_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=self.setup_project(root)
            group=pd.read_csv(root/'jobs/pilot/plan.csv',dtype={'stamp':str})
            out=root/'server_runs/pilot';folder=out/'2020080100';folder.mkdir(parents=True)
            pd.DataFrame([dict(sid='2020001N20000',init_time='2020-08-01 00:00:00',lead_h=h,
                base_lat=21.,base_lon=131.,model_id=MODEL,tracker_fallback=False) for h in range(6,121,6)]).to_csv(folder/'forecasts.csv',index=False)
            write_json(folder/'done.json',{'plan_subset_sha256':digest_subset(group),
                'forecasts_sha256':digest(folder/'forecasts.csv'),'identity':{'model_id':MODEL,'steps':20}})
            with patch.object(handoff,'ROOT',root):
                handoff.bundle('pilot',out,root/'returns/v1')
                handoff.accept(root/'returns/v1','pilot',run)
                with (root/'returns/v1/result/forecasts.csv').open('a') as f:f.write('tampered')
                with self.assertRaisesRegex(ValueError,'Corrupt return'):
                    handoff.accept(root/'returns/v1','pilot',run)

    def test_export_excludes_data_and_secret_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.setup_project(root)
            for name in ['README.md','START_HERE.md','requirements-local.txt','requirements-server.txt','.gitignore','.gitattributes','config.json']:
                (root/name).write_text('fixture')
            for name in ['scripts','tests','docs','configs','data','models']:(root/name).mkdir()
            (root/'.env').write_text('FAKE_SECRET_FIXTURE')
            (root/'data/labels.csv').write_text('future truth')
            with patch.object(handoff,'ROOT',root):handoff.export_repo(root/'export','pilot')
            self.assertFalse((root/'export/.env').exists())
            self.assertFalse((root/'export/data').exists())
            self.assertFalse((root/'export/models').exists())
            self.assertTrue((root/'export/jobs/pilot/plan.csv').exists())

if __name__=='__main__':unittest.main()
