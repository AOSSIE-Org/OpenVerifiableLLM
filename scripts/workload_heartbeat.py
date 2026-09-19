"""Keep the existing health observation fresh during bounded local operations.

This is not useful progress, export, provider evidence or a deadline extension.
The rental controller still stops on the original 300-second progress age and
1800-second export age, and the external watchdog retains its original clocks.
Only the single leased coordinator may use this context. It must check errors
before each new phase; exiting joins the helper and propagates any failure.
"""
from pathlib import Path
from threading import Event,Thread
from ovl_pipeline.canonical import EvidenceError
from ovl_pipeline.schema import integer
from workload_health import Health


class Heartbeat:
    def __init__(self,health,path,*,interval_seconds=10):
        if not isinstance(health,Health):raise EvidenceError('existing workload health required')
        integer(interval_seconds,1,15,'bounded health observation interval')
        self.health=health;self.path=Path(path);self.interval=interval_seconds
        self.stop=Event();self.thread=None;self.error=None

    def check(self):
        if self.error is not None:raise EvidenceError('workload heartbeat failed; preserve original guards') from self.error
        if self.health.journal._fd is None:raise EvidenceError('workload heartbeat lost its original lease')

    def __enter__(self):
        if self.thread is not None:raise EvidenceError('heartbeat context cannot be restarted')
        self.check();self.health.pulse(self.path)
        def run():
            while not self.stop.wait(self.interval):
                try:self.health.pulse(self.path)
                except Exception as error:self.error=error;self.stop.set();return
        self.thread=Thread(target=run,name='ovllm-health-observation',daemon=True);self.thread.start()
        return self

    def __exit__(self,kind,value,traceback):
        self.stop.set();self.thread.join(timeout=self.interval+5)
        if self.thread.is_alive():raise EvidenceError('workload heartbeat did not close')
        if kind is None:self.check()
        return False
