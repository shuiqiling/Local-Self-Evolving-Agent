"""Restore previous code and raw-record visibility, preserving all new events."""
from pathlib import Path
import hashlib,json,shutil,sqlite3,subprocess,sys,time

ROOT=Path('/home/ubuntu/AI-Native/embodied-agent');WORK=Path(__file__).resolve().parent
report=json.loads((WORK/'activation.json').read_text());backup=Path(report['backup'])
assert backup.resolve().is_relative_to((ROOT/'deployments').resolve())
automatic='--automatic' in sys.argv
if not automatic:
    with sqlite3.connect('file:'+str(ROOT/'data/memory.sqlite3')+'?mode=ro',uri=True) as db:
        assert not db.execute("SELECT id FROM jobs WHERE status IN ('queued','running')").fetchall(),'Wait for active jobs before rollback'
for name,original in report['original_hashes'].items():
    p=ROOT/'app/lea'/name
    current=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    assert current in [original,report['installed_hashes'][name]],('App changed after deployment',name)
def system(*args):return subprocess.run(['systemctl','--user',*args],check=True,timeout=150)
system('stop','lea-agent.service')
sys.path.insert(0,str(ROOT/'app'))
from lea.store import Store
from rollback_state import restore_legacy_cursors
restored=restore_legacy_cursors(Store(ROOT/'data'))
for name,original in report['original_hashes'].items():
    target=ROOT/'app/lea'/name
    if original is None:target.unlink(missing_ok=True)
    else:shutil.copy2(backup/name,target)
units=Path.home()/'.config/systemd/user'
drop=units/'lea-agent.service.d/zzzz-native-kv.conf'
drop.unlink(missing_ok=True)
unit=units/'lea-native-kv.service'
if unit.exists():
    system('disable','--now','lea-native-kv.service');unit.unlink()
system('daemon-reload');system('start','lea-agent.service')
result=dict(rolled_back=True,at=time.time(),restored_tasks=restored,new_events_and_constraints_preserved=True,kv_checkpoints_preserved=True)
(WORK/'rollback-result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
