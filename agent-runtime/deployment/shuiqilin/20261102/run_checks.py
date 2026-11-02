"""Verify snapshot hashes and run isolated checks without a model or network."""
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'source-manifest.json').read_text(encoding='utf-8'))
for name, digest in manifest['files'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name

env = dict(os.environ, PYTHONUTF8='1')
env.pop('PYTHONPATH', None)
scripts = ('check_opt.py', 'check_behavior.py', 'check_auto_integration.py',
           'check_reset.py', 'test_development.py', 'test_sessions.py',
           'test_idle_regression.py', 'test_active_memory.py')
results = []
commands = [(name,[sys.executable, '-X', 'utf8', str(ROOT / 'tests' / name)]) for name in scripts]
commands.append(('test_continuity.py',[sys.executable, '-m', 'pytest', '-q', '-rx',
                str(ROOT / 'tests' / 'test_continuity.py')]))
for name,command in commands:
    print('Checking ' + name, flush=True)
    run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                         text=True, encoding='utf-8')
    output = run.stdout + run.stderr
    (ROOT / (name + '.local.log')).write_text(output, encoding='utf-8')
    match = re.search(r'Ran (\d+) tests', output)
    passed = int(match[1]) if match and run.returncode == 0 else 0
    if name == 'test_continuity.py':
        match = re.search(r'(\d+) passed', output)
        passed = int(match[1]) if match else 0
    expected = re.search(r'(\d+) xfailed', output)
    result = {'suite':name,'exit_code':run.returncode,'passed':passed,
              'known_expected_failures':int(expected[1]) if expected else 0,
              'sqlite_resource_warnings':'ResourceWarning: unclosed database' in output}
    results.append(result);print(json.dumps(result),flush=True)
    if run.returncode:
        print(output[-8000:]);raise SystemExit(run.returncode)
report = {'source_files_verified':len(manifest['files']),'suites':results,
          'passed':sum(r['passed'] for r in results),
          'known_expected_failures':sum(r['known_expected_failures'] for r in results),
          'model_called':False,'network_used':False,
          'known_defect':"The deployed antecedent matcher misses '第二种'; its strict xfail is not a passed case."}
(ROOT / 'program-checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False),flush=True)
