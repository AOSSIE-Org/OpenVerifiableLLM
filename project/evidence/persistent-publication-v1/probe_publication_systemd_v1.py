"""Owned local service mechanics only; explicit publisher substitute, no remote actions."""
from pathlib import Path
import json,os,subprocess,sys,time,uuid
ROOT=Path.cwd();out=ROOT/'.ovllm-cache/publisher-systemd-probe-v1';out.mkdir(exist_ok=False)
source=out/'worker.py';source.write_text('''import os,signal,subprocess,sys,time\nfrom pathlib import Path\np=Path(sys.argv[1]);mode=sys.argv[2]\nc=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'])\n(p/'child.pid').write_text(str(c.pid))\n(p/'main.pid').write_text(str(os.getpid()))\ntime.sleep(1)\nif mode=='failure':raise SystemExit(124)\nif mode=='success':raise SystemExit(0)\ntime.sleep(60)\n''')
units=Path.home()/'.config/systemd/user';results=[]
def ctl(*args):return subprocess.run(['systemctl','--user',*args],capture_output=True,text=True,check=True,timeout=20).stdout
def live(pid):
 p=Path('/proc')/str(pid)/'stat'
 return p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z'
for mode in ('timeout','failure','success'):
 d=out/mode;d.mkdir();name='ovllm-publication-probe-'+uuid.uuid4().hex+'.service';target=units/name
 text=f'''[Unit]\nDescription=OpenVerifiableLLM explicit local process probe\n[Service]\nType=exec\nExecStart={Path(sys.executable).resolve()} -I -B {source} {d} {mode}\nRestart=no\nRemainAfterExit=no\nKillMode=control-group\nSendSIGKILL=yes\nTimeoutStartSec=20\nTimeoutStopSec=5\nRuntimeMaxSec=6\nMemoryMax=4G\nStandardOutput=append:{d}/stdout.log\nStandardError=append:{d}/stderr.log\n'''
 target.write_text(text);(d/'unit.service').write_text(text);ctl('daemon-reload');started=time.time()
 try:
  # This separate coordinator subprocess exits after systemctl acknowledges exec.
  launcher=subprocess.run([sys.executable,'-c',"import subprocess,sys;subprocess.run(['systemctl','--user','start',sys.argv[1]],check=True)",name],capture_output=True,text=True,timeout=25)
  assert launcher.returncode==0
  until=time.time()+3
  while not(d/'main.pid').exists() and time.time()<until:time.sleep(.1)
  pid=int((d/'main.pid').read_text());child=int((d/'child.pid').read_text());assert live(pid) and live(child)
  observed=ctl('show',name,'-p','MainPID,ActiveState,SubState,Result,InvocationID,ControlGroup');(d/'after-coordinator-exit.txt').write_text(observed)
  until=started+18
  while (live(pid) or live(child)) and time.time()<until:time.sleep(.25)
  final=ctl('show',name,'-p','MainPID,ActiveState,SubState,Result,InvocationID,ControlGroup,ExecMainStatus');(d/'final.txt').write_text(final)
  assert not live(pid) and not live(child),final
  if mode=='timeout':assert 'Result=timeout' in final
  elif mode=='failure':assert 'Result=exit-code' in final
  else:assert 'Result=success' in final
  results.append({'mode':mode,'unit':name,'coordinator_exit':launcher.returncode,'main_pid':pid,'child_pid':child,'all_processes_gone':True,'elapsed_ms':round((time.time()-started)*1000),'scope':'actual own user systemd process lifecycle; publisher/numerics substituted'})
 finally:
  ctl('stop',name);target.unlink();ctl('daemon-reload')
  subprocess.run(['systemctl','--user','reset-failed',name],capture_output=True)
(out/'result.json').write_text(json.dumps({'schema':'ovl.local-publication-service-probe.v1','result':'PASS','cases':results,'paid_resources':False,'production_admission':'NOT_RUN'},indent=2)+'\n')
print(json.dumps(results))
