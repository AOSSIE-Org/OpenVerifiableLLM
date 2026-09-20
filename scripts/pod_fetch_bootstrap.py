#!/usr/bin/env python3
"""Fetch two immutable public bootstrap archives; never install or execute them.

Standalone stdlib helper. Full byte identities, fresh destinations and the original
wall/monotonic deadline gate success. Failed partials remain private evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

REPO = 'AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence'
HOSTS = {'huggingface.co', 'cdn-lfs.huggingface.co', 'cdn-lfs.hf.co',
         'cdn-lfs-us-1.hf.co', 'cdn-lfs-eu-1.hf.co', 'cas-bridge.xethub.hf.co'}
NAMES = ['python.tar.gz', 'source.tar.gz']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def pairs(items):
    result = {}
    for k, v in items:
        if k in result: raise ValueError('duplicate key')
        result[k] = v
    return result


def selected(plan, expected):
    if any(p.is_symlink() for p in [plan, *plan.parents]): raise ValueError('plan symlink')
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected): raise ValueError('plan digest')
    raw = plan.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected: raise ValueError('plan identity differs')
    value = json.loads(raw, object_pairs_hook=pairs)
    if (type(value) is not dict or set(value) != {'schema', 'repo', 'files'}
        or value['schema'] != 'ovl.public-bootstrap.v1' or value['repo'] != REPO): raise ValueError('bootstrap plan schema')
    files = value['files']
    if type(files) is not list or len(files) != 2: raise ValueError('two archives required')
    for f in files:
        if type(f) is not dict or set(f) != {'path', 'repo_path', 'revision', 'bytes', 'sha256'}: raise ValueError('file schema')
        if type(f['bytes']) is not int or not 0 < f['bytes'] <= 64*1024**2: raise ValueError('archive size bound')
        for key, pattern in [('revision', '[0-9a-f]{40}'), ('sha256', '[0-9a-f]{64}'),
                             ('repo_path', r'[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*')]:
            if type(f[key]) is not str or not re.fullmatch(pattern, f[key]): raise ValueError('immutable confined file identity required')
        if any(p in ('.', '..') for p in f['repo_path'].split('/')): raise ValueError('repository path traversal')
    if [f['path'] for f in files] != NAMES: raise ValueError('exact ordered archive names required')
    return value


def public_url(value):
    p = urlsplit(value)
    if (p.scheme != 'https' or p.hostname not in HOSTS or p.port not in (None, 443)
        or p.username or p.password or p.fragment): raise ValueError('unselected public origin')
    return value


class Redirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def write(path, value, check=None):
    pending = path.with_name(path.name+'.pending') if check else path
    if check: check()
    with pending.open('x') as f:
        json.dump(value, f, sort_keys=True, separators=(',', ':')); f.write('\n'); f.flush(); os.fsync(f.fileno())
    if check:
        check(); os.link(pending, path)
    try:
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
        if check: check()
    except BaseException:
        if check: path.unlink()
        raise
    if check: pending.unlink()


def fetch(plan, expected, output, report, deadline, *, opener=None, wall=time.time, monotonic=time.monotonic):
    value = selected(Path(plan).absolute(), expected)
    output = Path(output).absolute(); report = Path(report).absolute()
    start = wall()
    if type(deadline) is not int or not 0 < deadline-start <= 90: raise ValueError('bounded original bootstrap deadline required')
    end = monotonic()+deadline-start
    for root in (output, report):
        if any(p.is_symlink() for p in [root, *root.parents]): raise ValueError('output symlink')
    if (output.exists() or report.exists() or report == output or output in report.parents
        or report in output.parents): raise ValueError('fresh separate output and report required')
    failure = report.with_name(report.name+'.failure.json')
    if failure.exists() or failure.is_symlink(): raise ValueError('preserve prior failure evidence')
    output.mkdir(mode=0o700, parents=True); report.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    client = opener or build_opener(ProxyHandler({}), Redirects())
    def left():
        seconds = min(deadline-wall(), end-monotonic())
        if seconds <= 0: raise TimeoutError('original bootstrap deadline expired')
        return seconds
    for index, item in enumerate(value['files']):
        partial = output/(item['path']+'.partial'); count = 0; h = hashlib.sha256()
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{item["revision"]}/{item["repo_path"]}'
        try:
            request = Request(url, headers={'Accept-Encoding': 'identity', 'User-Agent': 'OpenVerifiableLLM-bootstrap/1'})
            with client.open(request, timeout=min(15, left())) as response, partial.open('xb') as target:
                public_url(response.url)
                if response.status != 200 or response.headers.get('Content-Encoding', 'identity') != 'identity': raise ValueError('full identity response required')
                length = response.headers.get('Content-Length')
                if length is not None and (not length.isdecimal() or int(length) != item['bytes']): raise ValueError('public length differs')
                while True:
                    left(); chunk = response.read(min(1024**2, item['bytes']-count+1))
                    if not chunk: break
                    count += len(chunk)
                    if count > item['bytes']: raise ValueError('oversized archive')
                    target.write(chunk); h.update(chunk)
                target.flush(); os.fsync(target.fileno())
            if count != item['bytes'] or h.hexdigest() != item['sha256']: raise ValueError('complete archive identity differs')
            left(); os.link(partial, output/item['path']); partial.unlink()
            fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
        except Exception as error:
            write(failure, {'result': 'FAIL', 'path': item['path'], 'completed_files': index,
                            'received_bytes': count, 'partial_bytes': partial.stat().st_size if partial.exists() else 0, 'error_type': type(error).__name__})
            raise
    left()
    result = {'schema': 'ovl.public-bootstrap-result.v1', 'result': 'PASS', 'plan_sha256': expected,
              'files': value['files'], 'original_deadline_epoch': deadline,
              'scope': 'complete selected public archive bytes only; installation and CUDA NOT_RUN'}
    write(report, result, check=left)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'output', 'report'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--plan-sha256', required=True); p.add_argument('--deadline', type=int, required=True)
    a = p.parse_args(); fetch(a.plan, a.plan_sha256, a.output, a.report, a.deadline)


if __name__ == '__main__': main()
