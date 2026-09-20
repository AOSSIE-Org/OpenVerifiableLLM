"""Shared off-pod lifecycle safeguards, not provider or billing attestations."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import re
import time

from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json
from ovl_pipeline.supervision import ControllerBusy


def boot_clock():
    return {'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'boottime_ms':time.clock_gettime_ns(time.CLOCK_BOOTTIME)//1_000_000}


class Lifetime:
    """Persist one deadline in CLOCK_BOOTTIME; restart/reboot never renews it."""
    def __init__(self,journal,plan,*,wall=time.time,clock=boot_clock,initialize=False):
        self.plan=plan;self.wall=wall;self.clock=clock
        saved=[e['body'] for e in journal.events if e['kind']=='decision' and e['body'].get('action')=='LIFETIME_CLOCK']
        if not saved and initialize:
            c=clock();now_ms=int(wall()*1000);now=now_ms//1000
            anchor={'action':'LIFETIME_CLOCK','plan_sha256':digest(plan),'boot_id':c['boot_id'],
                    'observed_epoch':now,'observed_epoch_ms':now_ms,'boottime_ms':c['boottime_ms'],
                    'deadline_boottime_ms':c['boottime_ms']+max(0,plan['external_terminate_epoch']*1000-now_ms)}
            journal.append('decision',anchor);saved=[anchor]
        self.anchor=saved[0] if len(saved)==1 else None
        if self.anchor is not None:
            a=self.anchor
            if (a.get('plan_sha256')!=digest(plan) or type(a.get('observed_epoch')) is not int
                or type(a.get('boottime_ms')) is not int or type(a.get('deadline_boottime_ms')) is not int
                or type(a.get('observed_epoch_ms')) is not int or a['observed_epoch_ms']//1000!=a['observed_epoch']
                or not re.fullmatch(r'[a-zA-Z0-9-]{1,96}',a.get('boot_id',''))
                or a['deadline_boottime_ms']!=a['boottime_ms']+max(0,plan['external_terminate_epoch']*1000-a['observed_epoch_ms'])):
                raise EvidenceError('invalid persisted lifetime clock')
        self.high_water=max([plan['input']['now_epoch']]+[
            b['observed_epoch'] for e in journal.events
            for b in (e['body'],e['body'].get('account',{}))
            if type(b.get('observed_epoch')) is int])

    def remaining(self):
        if self.anchor is None:return 0  # Legacy/unclocked adoption may only stop.
        c=self.clock();a=self.anchor;now=self.wall()
        if c['boot_id']!=a['boot_id'] or c['boottime_ms']<a['boottime_ms'] or now<self.high_water-5:return 0
        self.high_water=max(self.high_water,now)
        return max(0,min(self.plan['external_terminate_epoch']-now,(a['deadline_boottime_ms']-c['boottime_ms'])/1000))


def attributed_ids(provider_request,name,known=None):
    """The pre-request UUID and empty-account baseline attribute duplicate copies.

    Every exact UUID-name match is stopped; similar/unrelated names are untouched.
    This is attribution within the owner's account, not adversarial multi-tenant
    authorization or proof the provider implements exactly-once creation.
    """
    if not re.fullmatch(r'ovllm-[a-z0-9-]*[0-9a-f]{32}',name):raise EvidenceError('UUID-suffixed attempt name required')
    d,_,_=provider_request('identities');ids={x['id'] for x in d['myself']['pods'] if x['name']==name}
    if known is not None:ids.add(known)
    if any(not re.fullmatch('[A-Za-z0-9_-]{1,96}',x) for x in ids):raise EvidenceError('invalid attributed identity')
    return sorted(ids)


def creation_root():
    return Path.home()/'.local/share/openverifiablellm/runpod-creation-fences'


@contextmanager
def account_lease(root=None):
    """One active rental controller across all journal directories on this host."""
    root=root or creation_root()
    if root.is_symlink():raise EvidenceError('creation fence root is a symlink')
    root.mkdir(parents=True,mode=0o700,exist_ok=True)
    if root.stat().st_uid!=os.getuid() or root.stat().st_mode&0o077:raise EvidenceError('private owned creation fence root required')
    fd=os.open(root/'account.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ControllerBusy('another rental controller holds the account lease') from None
        yield root
    finally:os.close(fd)


def fence(root,value,journal):
    """Durable per-attempt request fence, independent of the caller's journal path."""
    name=value['payload']['name'];path=root/(name+'.json')
    expected={'schema':'ovl.one-shot-creation-fence.v1','intent_sha256':digest(value),'attempt_id':name}
    if path.exists():
        if read_json(path)!=expected:raise EvidenceError('attempt name reused for a different intent')
        return False
    # Caller holds the account-wide lock. Write before any remote mutation; an
    # interrupted write can lose availability but can never authorize a retry.
    write_json(path,expected)
    journal.append('decision',{'action':'CREATION_FENCED','fence_sha256':digest(expected)})
    return True
