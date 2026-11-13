#!/usr/bin/env python3
"""Launch controller for the full-adder truth table (docs/public-full-adder-truth-table.md, 4 jobs) and the
lane-layout check (docs/public-lane-layout-check.md, 16 jobs). Standard library only; started by
tools/launch_full_adder_and_lane_layout.sh, which keeps the command-line interface.

Usage: launch_full_adder_and_lane_layout.sh MODE SOURCE_CHECKOUT FROZEN_COMMIT RUN_ROOT [JOB_SET]
  MODE             preflight (no model load) or execute
  SOURCE_CHECKOUT  an isolated checkout of the frozen commit (git worktree); must be clean and at FROZEN_COMMIT
  RUN_ROOT         a new directory; an existing one is refused (preflight and execution use separate roots)
  JOB_SET          adder-and-lanes (default: 4 adder and 16 two-lane jobs) or single-lane (the 8 jobs of the
                   lane check's single-lane arm, docs/public-lane-layout-check.md)

Environment (all validated; a bad value refuses the launch with status 2): MAX_JOBS (overall bound on concurrent
jobs, >= 1, default 10), JOB_GIB (memory budget per job, >= 1, default 7; peak RSS was about 5.5 GB), RESERVE_GIB
(headroom kept for everything else, >= 0, default 8), DEADLINE (per job, e.g. 4h; > 0), GRACE (seconds between TERM
and KILL, >= 1, default 120), ADMIT_WAIT (seconds, >= 1, default 7200), MEMINFO (meminfo-format file read for
MemAvailable, default /proc/meminfo), LAUNCH_LOCK (default /archive/shrimp-memory-data/audits/.launch-controller.lock;
only one controller may hold it, because admission is decided per controller and two would count the same memory),
PY (interpreter for the jobs).

Revised after Codex's Task 8 recheck (docs/audits/2026-11-13-task8-recheck/review.md):
R1  One controller decides every start, so concurrent jobs cannot each count the same free memory. A job is
    admitted only while fewer than MAX_JOBS run and MemAvailable, minus the part of every running job's budget not
    yet resident (JOB_GIB minus its anonymous RSS, summed over its process group, never below zero), minus
    RESERVE_GIB, still holds a whole JOB_GIB. Resident memory is counted once: it has already left MemAvailable.
    The initial capacity, min(MAX_JOBS, (MemAvailable - reserve) // job), is recorded. Jobs are re-admitted as
    others finish. If nothing is running and memory stays short for ADMIT_WAIT seconds, the remaining jobs are
    recorded as not admitted.
R2  Each job runs in a new session (its own process group) that only it and its descendants can belong to. The
    controller signals only those groups, never its own group or its caller. On SIGINT or SIGTERM it stops
    admitting, sends TERM to every running job's group, waits up to GRACE seconds, sends KILL to what remains, and
    then writes a record for every scheduled job (finished, timed out, stopped, refused, not admitted or not
    started) and the summary, and exits with 128 + the signal number. Waiting is a short poll, so a stop is acted
    on within about a fifth of a second. A job also receives TERM if the controller itself dies (PR_SET_PDEATHSIG).
The per-job deadline is enforced here (TERM at DEADLINE, KILL after GRACE), not by an external timeout program.
Exit status: 0 if every job exited 0; 1 if any job failed or did not run; 2 for a refused launch; 128 + signal if
stopped.
"""
import ctypes
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
A = '/archive/shrimp-memory-data/audits'
COMMON = ['--cache-dir', '/data2/neural-shrimp-data/models/hf/hub',
          '--vectors', '/archive/shrimp-memory-data/inputs/p8-reference/vectors.npz',
          '--references', f'{A}/p6-recovery/20261025T221011Z-recovery/reference-cycles.npz',
          '--reference-manifest', f'{A}/p6-recovery/20261025T221011Z-recovery/reference-manifest.json',
          '--source-hold-results', f'{A}/p6-followup/20261025T220342Z-followup/continued-hold/results.json',
          '--recovery-results', f'{A}/p6-recovery/20261025T221011Z-recovery/run/results.json',
          '--references-sha256', 'd96f923d537a6bcd6b86be7fe91f29a4cba6a46e5e036c8dd408817281395fe0',
          '--reference-manifest-sha256', '17b8e826a63a01d442877b6cf837272f760f49f8f4ba26aa6192294f47b6853f',
          '--recovery-results-sha256', '5de613795c4bc10e0584072fd7cb12586de4db869286d15cbbdc7f57dab37ef5']
THREADS = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
JOB_SETS = {
    'adder-and-lanes': ([(f'adder-x{x}-y{y}', ['tools/run_public_full_adder.py', '--x', str(x), '--y', str(y)])
                         for x, y in ((0, 0), (0, 1), (1, 0), (1, 1))]
                        + [(f'lanes-{k:02d}', ['tools/run_public_lane_layout.py', '--shard', str(k), '--shards', '16'])
                           for k in range(16)]),
    'single-lane': [(f'single-{k:02d}', ['tools/run_public_lane_singleton.py', '--shard', str(k), '--shards', '8'])
                    for k in range(8)],
}
JOBS = JOB_SETS['adder-and-lanes']
KIB_PER_GIB = 1024 * 1024
POLL = 0.2


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def refuse(message):
    print(f'refusing: {message}', file=sys.stderr)
    sys.exit(2)


def setting(name, default, minimum):
    raw = os.environ.get(name, str(default))
    if not re.fullmatch(r'[0-9]+', raw) or int(raw) < minimum:
        refuse(f'{name}={raw!r} must be an integer >= {minimum}')
    return int(raw)


def deadline_seconds(raw):
    m = re.fullmatch(r'([0-9]+)([smhd]?)', raw)
    if not m or int(m.group(1)) == 0:
        refuse(f'DEADLINE={raw!r} must be a positive duration such as 90s, 30m or 4h')
    return int(m.group(1)) * {'': 1, 's': 1, 'm': 60, 'h': 3600, 'd': 86400}[m.group(2)]


def mem_available_kib(meminfo):
    for line in Path(meminfo).read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])
    raise RuntimeError(f'no MemAvailable in {meminfo}')


def groups_of_interest(pgids):
    """One pass over /proc: member PIDs and anonymous resident KiB (memory already gone from MemAvailable) for each
    process group asked about."""
    out = {g: dict(pids=[], anon_kib=0) for g in pgids}
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        try:
            stat = Path(f'/proc/{entry}/stat').read_text()
            pgid = int(stat[stat.rindex(')') + 2:].split()[2])
            if pgid not in out:
                continue
            out[pgid]['pids'].append(int(entry))
            for line in Path(f'/proc/{entry}/status').read_text().splitlines():
                if line.startswith('RssAnon:'):
                    out[pgid]['anon_kib'] += int(line.split()[1])
        except (OSError, ValueError, IndexError):
            continue                                    # the process ended while being read
    return out


_PRCTL = ctypes.CDLL(None, use_errno=True).prctl


def child_setup(parent):
    """Runs in the child after setsid and before exec: TERM the job if the controller dies, and give up at once if
    it already has."""
    _PRCTL(1, int(signal.SIGTERM))                      # PR_SET_PDEATHSIG
    if os.getppid() != parent:
        os._exit(1)


def exited(pid):
    """True once the job's leader has exited, without reaping it: while it is an unreaped zombie its PID, and so its
    process group ID, cannot be reused, so signalling that group still reaches only the job's own processes."""
    try:
        return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT | os.WNOHANG) is not None
    except ChildProcessError:
        return True


class Controller:
    def __init__(self, mode, src, frozen, run, job_set='adder-and-lanes'):
        self.mode, self.src, self.frozen, self.run = mode, Path(src), frozen, Path(run)
        self.job_set, self.jobs = job_set, JOB_SETS[job_set]
        self.max_jobs = setting('MAX_JOBS', 10, 1)
        self.job_kib = setting('JOB_GIB', 7, 1) * KIB_PER_GIB
        self.reserve_kib = setting('RESERVE_GIB', 8, 0) * KIB_PER_GIB
        self.grace = setting('GRACE', 120, 1)
        self.admit_wait = setting('ADMIT_WAIT', 7200, 1)
        self.deadline = deadline_seconds(os.environ.get('DEADLINE', '4h'))
        self.meminfo = os.environ.get('MEMINFO', '/proc/meminfo')
        self.py = os.environ.get('PY', '/archive/shrimp-memory-data/envs/llm-cpu/bin/python')
        self.extra = ['--preflight-only'] if mode == 'preflight' else []
        self.records = {name: dict(job=name, argv=argv, state='scheduled') for name, argv in self.jobs}
        self.running = {}                              # name -> dict(proc, pgid, started, term_sent, kill_sent)
        self.stop_signal = None

    # --------------------------------------------------------------------------------------------- records
    def write(self, name):
        rec = self.records[name]
        (self.run/f'{name}.json').write_text(json.dumps(rec, indent=1) + '\n')
        if 'exit' in rec:
            (self.run/f'{name}.exit').write_text(f'{rec["exit"]}\n')

    def finish(self, name, state, status=None):
        rec = self.records[name]
        rec.update(state=state, finished=utc())
        if status is not None:
            rec['exit'] = status
        self.write(name)
        (self.run/f'{name}.finished').write_text(rec['finished'] + '\n')

    # ------------------------------------------------------------------------------------------- admission
    def free_for_new_kib(self):
        groups = groups_of_interest({j['pgid'] for j in self.running.values()})
        pending = sum(max(0, self.job_kib - groups[j['pgid']]['anon_kib']) for j in self.running.values())
        return mem_available_kib(self.meminfo) - pending - self.reserve_kib

    def start(self, name, argv):
        rec = self.records[name]
        if (self.run/name).exists():                   # never reuse a job directory
            rec.update(reason='job directory already exists')
            self.finish(name, 'refused_existing', 3)
            return
        log = open(self.run/f'{name}.console.log', 'ab', buffering=0)
        cmd = [self.py, '-u', *argv, *COMMON, '--output-dir', str(self.run/name), *self.extra]
        try:
            parent = os.getpid()
            proc = subprocess.Popen(cmd, cwd=self.src, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                    env=dict(os.environ, **THREADS), start_new_session=True,
                                    preexec_fn=lambda: child_setup(parent))
        except OSError as exc:
            log.close()
            rec.update(reason=f'could not start: {type(exc).__name__}: {exc}')
            self.finish(name, 'failed_to_start', 126)
            return
        log.close()
        rec.update(state='running', pid=proc.pid, pgid=proc.pid, command=cmd, started=utc())
        self.running[name] = dict(proc=proc, pgid=proc.pid, t0=time.monotonic(), term_sent=None, kill_sent=None)
        (self.run/f'{name}.started').write_text(rec['started'] + '\n')
        self.write(name)

    def reap(self):
        for name, j in list(self.running.items()):
            now = time.monotonic()
            if not exited(j['proc'].pid):
                if j['term_sent'] is None and now - j['t0'] > self.deadline:
                    self.signal_group(name, signal.SIGTERM, 'deadline')
                elif j['term_sent'] is not None and j['kill_sent'] is None and now - j['term_sent'] > self.grace:
                    self.signal_group(name, signal.SIGKILL, self.records[name].get('stop_reason', 'grace'))
                continue
            left = [p for p in groups_of_interest({j['pgid']})[j['pgid']]['pids'] if p != j['proc'].pid]
            if left:                                    # descendants outlived the job: they are its own; end them
                os.killpg(j['pgid'], signal.SIGKILL)
                self.records[name]['descendants_killed'] = left
            status = j['proc'].wait()
            del self.running[name]
            reason = self.records[name].get('stop_reason')
            state = {'deadline': 'timed_out', 'stop': 'stopped'}.get(reason, 'finished')
            self.finish(name, state, status if status >= 0 else 128 - status)

    def signal_group(self, name, sig, reason):
        j, rec = self.running[name], self.records[name]
        try:
            os.killpg(j['pgid'], sig)                   # only this job's own session
        except ProcessLookupError:
            return
        rec.setdefault('stop_reason', reason)
        rec.setdefault('signals', []).append(dict(signal=signal.Signals(sig).name, utc=utc(), reason=reason))
        if sig == signal.SIGTERM:
            j['term_sent'] = time.monotonic()
        else:
            j['kill_sent'] = time.monotonic()
        self.write(name)

    # ---------------------------------------------------------------------------------------------- main loop
    def on_signal(self, signum, _frame):
        if self.stop_signal is None:
            self.stop_signal = signum

    def launch(self):
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, self.on_signal)
        queue = list(self.jobs)
        (self.run/'schedule.txt').write_text(''.join(f'{n} {" ".join(a)}\n' for n, a in queue))
        for name, _ in queue:
            self.write(name)
        blocked_since = None
        while (queue or self.running) and self.stop_signal is None:
            self.reap()
            if queue and len(self.running) < self.max_jobs and self.free_for_new_kib() >= self.job_kib:
                name, argv = queue.pop(0)
                self.start(name, argv)
                blocked_since = None
                continue
            if queue and not self.running:
                blocked_since = blocked_since or time.monotonic()
                if time.monotonic() - blocked_since > self.admit_wait:
                    for name, _ in queue:
                        self.records[name]['reason'] = f'memory short for {self.admit_wait} s with nothing running'
                        self.finish(name, 'not_admitted', 4)
                    queue = []
            time.sleep(POLL)
        if self.stop_signal is not None:
            for name in list(self.running):
                self.signal_group(name, signal.SIGTERM, 'stop')
            while self.running:
                self.reap()
                time.sleep(POLL)
            for name, _ in queue:
                self.records[name]['reason'] = f'launcher stopped by {signal.Signals(self.stop_signal).name}'
                self.finish(name, 'not_started')
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)          # the outcome is fixed; finish writing it
        summary = dict(mode=self.mode, job_set=self.job_set, finished=utc(), stopped_by=(signal.Signals(self.stop_signal).name
                                                                   if self.stop_signal else None),
                       jobs=[{k: self.records[n].get(k) for k in ('job', 'state', 'exit', 'started', 'finished',
                                                                  'stop_reason')} for n, _ in self.jobs])
        summary['all_exited_zero'] = all(j['exit'] == 0 for j in summary['jobs'])
        (self.run/'launch-summary.json').write_text(json.dumps(summary, indent=1) + '\n')
        if self.stop_signal is not None:
            return 128 + self.stop_signal
        return 0 if summary['all_exited_zero'] else 1


def main(argv):
    if len(argv) not in (4, 5):
        refuse('usage: MODE SOURCE_CHECKOUT FROZEN_COMMIT RUN_ROOT [JOB_SET]')
    mode, src, frozen, run = argv[:4]
    job_set = argv[4] if len(argv) == 5 else 'adder-and-lanes'
    if mode not in ('preflight', 'execute'):
        refuse('MODE must be preflight or execute')
    if job_set not in JOB_SETS:
        refuse(f'JOB_SET must be one of {sorted(JOB_SETS)}')
    controller = Controller(mode, src, frozen, run, job_set)  # validates the environment first
    lock_path = os.environ.get('LAUNCH_LOCK', '/archive/shrimp-memory-data/audits/.launch-controller.lock')
    try:
        lock = open(lock_path, 'a')
    except OSError as exc:
        refuse(f'cannot open the launch lock {lock_path} ({type(exc).__name__}); set LAUNCH_LOCK')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # held until this process exits
    except BlockingIOError:
        refuse(f'another launch controller holds {lock_path}')
    if Path(run).exists():
        refuse(f'{run} exists (use a new root)')
    head = subprocess.run(['git', '-C', src, 'rev-parse', 'HEAD'], capture_output=True, text=True)
    if head.returncode or head.stdout.strip() != frozen:
        refuse(f'{src} is not at {frozen}')
    status = subprocess.run(['git', '-C', src, 'status', '--porcelain'], capture_output=True, text=True)
    if status.returncode or status.stdout.strip():
        refuse(f'{src} is not clean')
    try:
        available = mem_available_kib(controller.meminfo)
    except (OSError, RuntimeError, ValueError) as exc:
        refuse(f'cannot read MemAvailable from {controller.meminfo} ({exc})')
    Path(run).mkdir(parents=True)
    py_version = subprocess.run([controller.py, '-V'], capture_output=True, text=True)
    record = dict(mode=mode, job_set=job_set, jobs=len(controller.jobs), source_checkout=src, frozen_commit=frozen,
                  utc=utc(), host=os.uname().nodename,
                  kernel=os.uname().release, cpus=os.cpu_count(), meminfo=controller.meminfo,
                  mem_available_kib=available, python=(py_version.stdout + py_version.stderr).strip(),
                  controller=dict(path=str(HERE), sha256=hashlib.sha256(HERE.read_bytes()).hexdigest(),
                                  python=sys.version.split()[0]),
                  max_jobs=controller.max_jobs, job_gib=controller.job_kib // KIB_PER_GIB,
                  reserve_gib=controller.reserve_kib // KIB_PER_GIB, deadline_seconds=controller.deadline,
                  grace_seconds=controller.grace, admit_wait_seconds=controller.admit_wait,
                  initial_capacity_jobs=min(controller.max_jobs,
                                            max(0, (available - controller.reserve_kib) // controller.job_kib)),
                  launch_lock=lock_path, launcher_pid=os.getpid())
    (Path(run)/'launch-record.json').write_text(json.dumps(record, indent=1) + '\n')
    return controller.launch()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
