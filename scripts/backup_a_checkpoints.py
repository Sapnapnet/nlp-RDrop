"""Read-only source backup of A's three accepted checkpoints; no third-party imports.

Default: create a timestamped ZIP, verify every archived byte against the accepted
manifest, then write feedback. --verify-only checks a downloaded ZIP without extraction.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'configs/a_checkpoint_backup_v1.json'


def digest_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def sha256(path):
    with Path(path).open('rb') as stream:
        return digest_stream(stream)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checked_plan(plan):
    require(plan['schema_version'] == 1 and plan['seeds'] == [13,21,42], 'Unexpected backup plan.')
    names = []
    for item in plan['files']:
        rel = PurePosixPath(item['path'])
        require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in item['path'] and ':' not in item['path'], 'Unsafe source path.')
        require(rel.parts and rel.parts[0] == 'reports', 'Backup plan is limited to reports/checkpoints.')
        require(isinstance(item['size'], int) and item['size'] >= 0, 'Invalid size.')
        require(len(item['sha256']) == 64 and all(c in '0123456789abcdef' for c in item['sha256']), 'Invalid SHA256.')
        names.append(item['path'])
    require(len(names) == len(set(names)) and names, 'Duplicate/empty backup paths.')
    return plan


def verify_archive(archive_path, plan):
    """Use the separately supplied accepted plan, not only an archive's own hashes."""
    checked_plan(plan)
    expected = {'rdrop-mrpc-project/' + item['path']: item for item in plan['files']}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        require(len(names) == len(expected) + 1 and set(names) == set(expected) | {'backup_manifest.json'}, 'Unexpected/missing/duplicate archive members.')
        manifest = json.loads(archive.read('backup_manifest.json'))
        require(manifest['accepted_plan'] == plan, 'Embedded plan differs from accepted local plan.')
        for name, item in expected.items():
            info = archive.getinfo(name)
            require(info.file_size == item['size'], 'Archive size mismatch: ' + name)
            with archive.open(name) as stream:
                require(digest_stream(stream) == item['sha256'], 'Archive SHA256 mismatch: ' + name)
    return {'status':'PASS', 'archive':str(Path(archive_path).resolve()), 'archive_bytes':Path(archive_path).stat().st_size,
            'archive_sha256':sha256(archive_path), 'verified_payload_files':len(expected), 'seeds':[13,21,42],
            'scope':'Byte integrity against accepted checkpoints; no model loading or inference.'}


def create_backup(root, output, plan, say=print):
    checked_plan(plan)
    root, output = root.resolve(), output.resolve()
    entries = []
    total = sum(item['size'] for item in plan['files'])
    require(shutil.disk_usage(output).free >= total + 256 * 1024**2, 'Insufficient free space for uncompressed backup plus 256 MiB margin.')
    for item in plan['files']:
        path = root / item['path']
        # Refuse symlinked inputs, including parent directories, to avoid unrelated data.
        require(path.resolve().is_relative_to(root), 'Source path escaped project root.')
        rel_parts = PurePosixPath(item['path']).parts
        require(not any(root.joinpath(*rel_parts[:i]).is_symlink() for i in range(1,len(rel_parts)+1)), 'Symlinked backup source refused.')
        require(path.is_file() and path.stat().st_size == item['size'], 'Missing/changed source: '+item['path'])
        say('Checking source: '+item['path'])
        require(sha256(path) == item['sha256'], 'Source SHA256 mismatch: '+item['path'])
        entries.append((path, item['path']))
    target = output / 'baseline_checkpoints_seeds13_21_42.zip'
    partial = output / (target.name + '.partial')
    require(not target.exists() and not partial.exists(), 'Refusing to overwrite an earlier backup.')
    manifest = {'created_at_utc':datetime.now(timezone.utc).isoformat(), 'accepted_plan':plan,
                'contents':'Three final inference checkpoints and accepted run evidence. No optimizer/RNG resume state; no pretrained inputs, dependencies or credentials.'}
    # ZIP_STORED avoids CPU compression cost and has predictable space requirements.
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.writestr('backup_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
        for path, rel in entries:
            say('Archiving: '+rel)
            archive.write(path, arcname='rdrop-mrpc-project/'+rel)
    say('Verifying all archived payload hashes...')
    result = verify_archive(partial, plan)
    partial.rename(target)
    result['archive'] = str(target)
    (output/(target.name+'.sha256')).write_text(result['archive_sha256']+'  '+target.name+'\n',encoding='ascii')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', type=Path, help='Verify a downloaded ZIP without extraction or training.')
    parser.add_argument('--output-root', type=Path, default=ROOT/'reports/backups')
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8',errors='replace')
    plan = checked_plan(json.loads(PLAN.read_text(encoding='utf-8')))
    if args.verify_only:
        try:
            print(json.dumps(verify_archive(args.verify_only,plan),ensure_ascii=False,indent=2),flush=True)
            return 0
        except Exception:
            print(traceback.format_exc(),flush=True)
            return 1
    args.output_root.mkdir(parents=True,exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('a_checkpoints_%Y%m%d_%H%M%S_UTC_')
    output = Path(tempfile.mkdtemp(prefix=stamp,dir=args.output_root)).resolve()
    report = {'step':7,'purpose':'Backup accepted A checkpoints without training','overall_status':'RUNNING',
              'started_at_utc':datetime.now(timezone.utc).isoformat(),'output_directory':str(output),
              'accepted_plan_sha256':sha256(PLAN),'backup_script_sha256':sha256(Path(__file__)),
              'source_project':str(ROOT)}
    with (output/'backup.log').open('w',encoding='utf-8') as log:
        def say(message):
            print(message,flush=True)
            log.write(message+'\n')
            log.flush()
        say('Backup directory: '+str(output))
        try:
            report['archive_verification'] = create_backup(ROOT,output,plan,say)
            report['overall_status'] = 'PASS'
        except BaseException:
            report['overall_status'] = 'FAIL'
            report['traceback'] = traceback.format_exc()
            say(report['traceback'])
        say('Overall: '+report['overall_status'])
    report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(output/'backup_result.json',report)
    feedback = output/'feedback_step7.zip'
    with zipfile.ZipFile(feedback,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ('backup_result.json','backup.log','baseline_checkpoints_seeds13_21_42.zip.sha256'):
            if (output/name).is_file(): archive.write(output/name,arcname=name)
        archive.write(PLAN,arcname='a_checkpoint_backup_v1.json')
    print('Feedback ZIP: '+str(feedback),flush=True)
    if report['overall_status']=='PASS':
        print('Download model backup separately: '+report['archive_verification']['archive'],flush=True)
    return 0 if report['overall_status']=='PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
