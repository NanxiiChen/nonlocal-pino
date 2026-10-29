"""One-time, user-authorized preservation and untracking of generated outputs."""
import filecmp
import json
from pathlib import Path
import shutil
import subprocess

REPO=next(path for path in Path(__file__).resolve().parents if (path/'.git').exists())
ARCHIVE=Path('/root/autodl-tmp/data/nonlocal_pinn_artifacts/20261029-publication')
EXTENSIONS={'.png','.jpg','.jpeg','.webp','.svg','.pdf','.gif','.mp4','.webm','.mov',
            '.html','.npz','.npy','.eqx','.hlo','.pkl','.pickle','.pt','.pth','.safetensors'}
ARCHIVE.mkdir(parents=True,exist_ok=False)
tracked={p.decode() for p in subprocess.check_output(['git','ls-files','-z'],cwd=REPO).split(b'\0') if p}
records=[]
for source in sorted((REPO/'docs/tasks').rglob('*')):
    if not source.is_file() or source.suffix.lower() not in EXTENSIONS:
        continue
    relative=source.relative_to(REPO)
    target=ARCHIVE/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    assert filecmp.cmp(source,target,shallow=False),relative
    records.append(dict(path=str(relative),bytes=source.stat().st_size,
                        previously_tracked=str(relative) in tracked))
manifest=dict(schema=1,archive_root=str(ARCHIVE),
    policy='Generated research media, experiment arrays/checkpoints and compiler dumps are local artifacts. Existing files stay at original workspace paths. No history rewrite.',
    comparison='Byte-for-byte backup checks; no file hashes.',files=records,
    generated_suffixes=sorted(EXTENSIONS))
(REPO/'artifacts/local_files.json').write_text(json.dumps(manifest,indent=2)+'\n')
ignore=REPO/'.gitignore'
text=ignore.read_text().rstrip()+'''\n\n# Generated research outputs: metadata and regeneration scripts stay in Git.
# Local copies remain usable; restore through artifacts/local_files.json.
'''
text+=''.join('/docs/tasks/**/*'+ext+'\n' for ext in sorted(EXTENSIONS))
ignore.write_text(text)
paths=[r['path'] for r in records if r['previously_tracked']]
file=Path('/tmp/nonlocal-pinn-generated-paths')
file.write_bytes(b'\0'.join(p.encode() for p in paths)+b'\0')
result=subprocess.run(['git','rm','--cached','--pathspec-from-file='+str(file),'--pathspec-file-nul'],
    cwd=REPO,text=True,capture_output=True,check=False)
if result.returncode:
    raise RuntimeError(result.stderr)
summary=dict(preserved_files=len(records),preserved_bytes=sum(r['bytes'] for r in records),
    removed_from_tracking=len(paths),removed_tracked_bytes=sum(r['bytes'] for r in records if r['previously_tracked']),
    new_generated_bytes_excluded=sum(r['bytes'] for r in records if not r['previously_tracked']),
    all_local_files_kept=all((REPO/r['path']).is_file() for r in records),archive_root=str(ARCHIVE))
(REPO/'docs/tasks/operator-artifact-curation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
