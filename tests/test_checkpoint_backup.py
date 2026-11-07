"""Small synthetic backup tests; never load or create real model weights."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from backup_a_checkpoints import checked_plan, create_backup, verify_archive


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'project'
        self.output=Path(self.temp.name)/'backup'
        self.root.mkdir()
        self.output.mkdir()
        files=[]
        for seed in (13,21,42):
            rel=f'reports/fixture_seed{seed}/checkpoint-final/pytorch_model.bin'
            data=f'tiny synthetic fixture for seed {seed}; not a real checkpoint'.encode()
            p=self.root/rel
            p.parent.mkdir(parents=True)
            p.write_bytes(data)
            files.append({'path':rel,'size':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        self.plan={'schema_version':1,'seeds':[13,21,42],'files':files}

    def create(self):
        return create_backup(self.root,self.output,self.plan,lambda _:None)

    def rewrite_zip(self,path,change):
        with zipfile.ZipFile(path) as archive:
            contents={name:archive.read(name) for name in archive.namelist()}
        change(contents)
        with zipfile.ZipFile(path,'w') as archive:
            for name,data in contents.items(): archive.writestr(name,data)

    def test_full_round_trip_preserves_sources_and_excludes_unlisted_files(self):
        before={f['path']:(self.root/f['path']).read_bytes() for f in self.plan['files']}
        (self.root/'unlisted.txt').write_text('not part of backup')
        report=self.create()
        self.assertEqual(report['status'],'PASS')
        self.assertEqual(report['verified_payload_files'],3)
        archive_path=Path(report['archive'])
        self.assertEqual(verify_archive(archive_path,self.plan)['archive_sha256'],report['archive_sha256'])
        with zipfile.ZipFile(archive_path) as archive:
            self.assertEqual(len(archive.namelist()),4)
            self.assertNotIn('unlisted.txt',archive.namelist())
        for rel,data in before.items(): self.assertEqual((self.root/rel).read_bytes(),data)

    def test_source_tampering_rejected_before_archive_creation(self):
        p=self.root/self.plan['files'][0]['path']
        p.write_bytes(b'x'*p.stat().st_size)
        with self.assertRaisesRegex(RuntimeError,'Source SHA256'):
            self.create()
        self.assertEqual(list(self.output.iterdir()),[])

    def test_missing_source_rejected(self):
        (self.root/self.plan['files'][0]['path']).unlink()
        with self.assertRaisesRegex(RuntimeError,'Missing/changed source'):
            self.create()

    def test_tampered_download_rejected_even_with_valid_zip_crc(self):
        path=Path(self.create()['archive'])
        first='rdrop-mrpc-project/'+self.plan['files'][0]['path']
        self.rewrite_zip(path,lambda values:values.update({first:b'x'*len(values[first])}))
        with self.assertRaisesRegex(RuntimeError,'Archive SHA256'):
            verify_archive(path,self.plan)

    def test_embedded_plan_does_not_replace_trusted_plan(self):
        path=Path(self.create()['archive'])
        def edit(values):
            info=json.loads(values['backup_manifest.json'])
            info['accepted_plan']['seeds']=[1,2,3]
            values['backup_manifest.json']=json.dumps(info).encode()
        self.rewrite_zip(path,edit)
        with self.assertRaisesRegex(RuntimeError,'Embedded plan'):
            verify_archive(path,self.plan)

    def test_unsafe_and_duplicate_paths_rejected(self):
        for rel in ('../outside','/reports/a','reports/../../outside','reports\\a','C:/reports/a'):
            plan=copy.deepcopy(self.plan)
            plan['files'][0]['path']=rel
            with self.assertRaises(RuntimeError): checked_plan(plan)
        plan=copy.deepcopy(self.plan)
        plan['files'].append(plan['files'][0])
        with self.assertRaisesRegex(RuntimeError,'Duplicate'): checked_plan(plan)

    def test_no_overwrite_and_low_space(self):
        path=Path(self.create()['archive'])
        before=path.read_bytes()
        with self.assertRaisesRegex(RuntimeError,'overwrite'): self.create()
        self.assertEqual(path.read_bytes(),before)
        with patch('backup_a_checkpoints.shutil.disk_usage',return_value=type('Disk',(),{'free':0})()):
            with self.assertRaisesRegex(RuntimeError,'Insufficient free space'): self.create()

    def test_cli_backup_and_verify_only(self):
        (self.root/'scripts').mkdir()
        (self.root/'configs').mkdir()
        script=self.root/'scripts/backup_a_checkpoints.py'
        shutil.copyfile(ROOT/'scripts/backup_a_checkpoints.py',script)
        (self.root/'configs/a_checkpoint_backup_v1.json').write_text(json.dumps(self.plan))
        run=subprocess.run([sys.executable,str(script)],capture_output=True,text=True,timeout=30)
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)
        feedback=list((self.root/'reports/backups').rglob('feedback_step7.zip'))
        self.assertEqual(len(feedback),1)
        with zipfile.ZipFile(feedback[0]) as z:
            report=json.loads(z.read('backup_result.json'))
            self.assertEqual(report['overall_status'],'PASS')
            self.assertFalse(any(n.endswith('.bin') for n in z.namelist()))
        verify=subprocess.run([sys.executable,str(script),'--verify-only',report['archive_verification']['archive']],capture_output=True,text=True,timeout=30)
        self.assertEqual(verify.returncode,0,verify.stdout+verify.stderr)
        self.assertEqual(json.loads(verify.stdout)['status'],'PASS')


if __name__=='__main__': unittest.main()
