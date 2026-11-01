"""Run the published snapshot checks in separate processes; no model required."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
commands=[[sys.executable,str(root/'tests'/name)] for name in
 ('check_opt.py','check_behavior.py','check_auto_integration.py','check_reset.py')]
commands.append([sys.executable,'-m','pytest','-q',str(root/'tests/test_continuity.py')])
for command in commands:
    result=subprocess.run(command,cwd=root)
    if result.returncode:raise SystemExit(result.returncode)
