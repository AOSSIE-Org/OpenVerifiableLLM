import os,signal,subprocess,sys,time
from pathlib import Path
p=Path(sys.argv[1]);mode=sys.argv[2]
c=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'])
(p/'child.pid').write_text(str(c.pid))
(p/'main.pid').write_text(str(os.getpid()))
time.sleep(1)
if mode=='failure':raise SystemExit(124)
if mode=='success':raise SystemExit(0)
time.sleep(60)
