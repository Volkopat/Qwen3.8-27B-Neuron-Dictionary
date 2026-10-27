"""Regression checks for the release audit, including deliberate corruption."""
import csv,importlib.util,json,shutil,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('verify_release',ROOT/'tools/verify_release.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class ReleaseTests(unittest.TestCase):
    def test_complete_release(self):
        r=audit.verify(ROOT);self.assertEqual(r['errors'],[]);self.assertEqual(r['effect_cells'],11264)

    def changed_copy(self,name,change):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name)/'release';shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))
        p=root/name
        with p.open(encoding='utf-8',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
        change(rows)
        with p.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        return root

    def test_signed_delta_corruption_detected(self):
        def change(rows):rows[0]['baseline_minus_intervention_logp_nats']=str(-float(rows[0]['baseline_minus_intervention_logp_nats']))
        r=audit.verify(self.changed_copy('data/probe_effects.csv',change));self.assertEqual(r['status'],'FAIL');self.assertTrue(any('effect mismatch' in x for x in r['errors']))

    def test_duplicate_key_detected(self):
        def change(rows):rows[1]=dict(rows[0])
        r=audit.verify(self.changed_copy('data/dictionary.csv',change),check_hashes=False);self.assertEqual(r['status'],'FAIL');self.assertTrue(any('unique entries' in x for x in r['errors']))

    def test_wrong_change_flag_detected(self):
        def change(rows):rows[0]['excerpt_differs']='false' if rows[0]['excerpt_differs']=='true' else 'true'
        r=audit.verify(self.changed_copy('data/examples.csv',change));self.assertEqual(r['status'],'FAIL');self.assertTrue(any('difference flag' in x for x in r['errors']))

    def test_unexpected_model_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'release';shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))
            p=root/'data/model_metadata.json';obj=json.loads(p.read_text());obj['model_id']='Qwen/Qwen2.5-7B-Instruct';p.write_text(json.dumps(obj))
            self.assertIn('wrong model attribution',audit.verify(root,check_hashes=False)['errors'])

    def test_wrong_revision_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'release';shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))
            p=root/'data/model_metadata.json';obj=json.loads(p.read_text());obj['revision']='0'*40;p.write_text(json.dumps(obj))
            self.assertIn('model reference revision mismatch',audit.verify(root,check_hashes=False)['errors'])

if __name__=='__main__':unittest.main()
