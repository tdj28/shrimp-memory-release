#!/usr/bin/env python3
"""Bound readback output growth; preserve evidence and signal only its model.

Linux-only. The wrapper must start a separate session, pass its SID, and stop
this guard in its EXIT trap. Regular files below BASE count except return
packaging: return-shards directories and root final-artifacts.tar and
final-manifest.json. No files are deleted.
"""
import argparse
import json
import os
from pathlib import Path
import re
import select
import signal
import stat
import sys
import subprocess
import time

LIMIT_BYTES = 128 * 1024 * 1024
POLL_SECONDS = 5
MAX_SECONDS = 2820
TERM_GRACE_SECONDS = 30
SCRIPT = "tools/run_p6_readback.py"


def regular_file_bytes(base, *, _root=True):
    """Sum logical bytes, without following symlinks or silently skipping errors."""
    total = 0
    with os.scandir(base) as entries:
        for entry in entries:
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISREG(info.st_mode):
                if _root and entry.name in {"final-artifacts.tar", "final-manifest.json"}:
                    continue
                total += info.st_size
            elif stat.S_ISDIR(info.st_mode):
                if entry.name != "return-shards":
                    total += regular_file_bytes(entry.path, _root=False)
    return total


def matches_argv(argv, base):
    # timeout, uv and shell wrappers can contain all model arguments too.
    if not argv or not re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", Path(argv[0]).name):
        return False
    i = 1
    while i < len(argv) and argv[i] in {"-u", "-B", "-E", "-s", "-S", "-I"}:
        i += 1
    if i >= len(argv) or argv[i] != SCRIPT:
        return False
    args = argv[i + 1:]
    for flag, expected in (("--output-dir", str(base / "run")),
                           ("--packet", str(base / "packet.npz"))):
        if args.count(flag) != 1 or any(a.startswith(flag + "=") for a in args):
            return False
        index = args.index(flag)
        if index + 1 == len(args) or args[index + 1] != expected:
            return False
    return True


class Guard:
    """Dependencies are injectable so tests never inspect or signal real jobs."""

    def __init__(self, base, wrapper_sid, *, proc_root=Path("/proc"),
                 getsid=os.getsid, pidfd_open=None,
                 send_signal=None, close=os.close, platform=sys.platform,
                 monotonic=time.monotonic, sleep=time.sleep,
                 utc=lambda: subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
                 byte_count=regular_file_bytes,
                 pidfd_exited=lambda handle: bool(select.select([handle], [], [], 0)[0])):
        self.base = Path(base)
        if not self.base.is_absolute() or wrapper_sid <= 1:
            raise ValueError("BASE must be absolute and wrapper SID greater than 1")
        self.wrapper_sid = wrapper_sid
        self.proc_root = Path(proc_root)
        self.getsid = getsid
        self.pidfd_open = pidfd_open if pidfd_open is not None else getattr(os, "pidfd_open", None)
        self.send_signal = send_signal if send_signal is not None else getattr(signal, "pidfd_send_signal", None)
        self.close, self.platform = close, platform
        self.monotonic, self.sleep, self.utc = monotonic, sleep, utc
        self.byte_count = byte_count
        self.pidfd_exited = pidfd_exited
        self.target = None
        self.pidfd = None

    def event(self, event, **fields):
        record = {"utc": self.utc(), "event": event, **fields}
        line = json.dumps(record, sort_keys=True) + "\n"
        # stderr retains the event even if the output filesystem becomes full.
        print(line, end="", file=sys.stderr, flush=True)
        try:
            with (self.base / "output-byte-guard.jsonl").open("a") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            print(f"guard event append failed: {exc!r}", file=sys.stderr, flush=True)

    def identity(self, pid):
        """Require a live exact command, start tick, and the wrapper's session."""
        try:
            directory = self.proc_root / str(pid)
            raw = (directory / "stat").read_text()
            fields = raw[raw.rindex(")") + 2:].split()
            if fields[0] in {"Z", "X"}:
                return None
            start_tick = int(fields[19])  # /proc/PID/stat field 22
            argv = tuple(os.fsdecode(a) for a in (directory / "cmdline").read_bytes().split(b"\0") if a)
            if not matches_argv(argv, self.base) or self.getsid(pid) != self.wrapper_sid:
                return None
            return pid, start_tick, argv
        except (OSError, ValueError, IndexError):
            return None

    def discover(self):
        matches = []
        for entry in self.proc_root.iterdir():
            if not entry.name.isascii() or not entry.name.isdecimal():
                continue
            pid = int(entry.name)
            if pid <= 1 or pid == self.wrapper_sid:
                continue
            identity = self.identity(pid)
            if identity is not None:
                matches.append(identity)
        if len(matches) > 1:
            raise RuntimeError("multiple matching model processes; refusing to choose")
        if not matches:
            return
        candidate = matches[0]
        try:
            descriptor = self.pidfd_open(candidate[0], 0)
        except ProcessLookupError:
            return
        if self.identity(candidate[0]) != candidate:
            self.close(descriptor)
            return
        self.target, self.pidfd = candidate, descriptor
        self.event("target_bound", pid=candidate[0], start_tick=candidate[1])

    def still_matches(self):
        return self.target is not None and self.identity(self.target[0]) == self.target

    def signal_target(self, number, size, reason):
        if not self.still_matches():
            self.event("signal_skipped_identity_changed", size_bytes=size, reason=reason)
            return False
        try:
            # The retained pidfd always names the original process, even if its
            # numeric PID is reused between the check above and this syscall.
            self.send_signal(self.pidfd, number, None, 0)
        except ProcessLookupError:
            self.event("target_exited_before_signal", pid=self.target[0], size_bytes=size)
            return False
        self.event("signal_sent", pid=self.target[0], signal=number,
                   size_bytes=size, reason=reason)
        return True

    def stop_target(self, size, reason, deadline):
        if not self.signal_target(signal.SIGTERM, size, reason):
            return
        kill_at = self.monotonic() + TERM_GRACE_SECONDS
        stop_at = min(kill_at, deadline)
        while self.monotonic() < stop_at:
            self.sleep(min(POLL_SECONDS, stop_at - self.monotonic()))
            if not self.still_matches():
                self.event("target_exited_or_changed", size_bytes=size)
                return
        if self.monotonic() < kill_at:
            self.event("guard_deadline_before_kill", size_bytes=size)
            return
        self.signal_target(signal.SIGKILL, size, reason)

    def run(self):
        deadline = self.monotonic() + MAX_SECONDS
        size = None
        try:
            self.event("guard_started", limit_bytes=LIMIT_BYTES,
                       max_seconds=MAX_SECONDS, wrapper_sid=self.wrapper_sid)
            if self.platform != "linux" or not callable(self.pidfd_open) or not callable(self.send_signal):
                self.event("guard_preflight_error", error="Linux pidfd APIs required")
                return 2
            try:
                size = self.byte_count(self.base)
                if size >= LIMIT_BYTES:
                    self.event("byte_limit_before_ready", size_bytes=size)
                    return 1
                with (self.base / "output-byte-guard.ready").open("x") as stream:
                    json.dump({"utc": self.utc(), "initial_size_bytes": size,
                               "limit_bytes": LIMIT_BYTES}, stream)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception as exc:
                self.event("guard_preflight_error", error=repr(exc), size_bytes=size)
                return 2
            while self.monotonic() < deadline:
                if self.target is None:
                    self.discover()
                elif not self.still_matches():
                    if not self.pidfd_exited(self.pidfd):
                        raise RuntimeError("live model identity could not be verified")
                    self.event("target_exited_or_changed", size_bytes=size)
                    # The supervisor owns cleanup. Staying alive makes an
                    # unexpected early guard exit unambiguously a failure.
                    while self.monotonic() < deadline:
                        remaining = deadline - self.monotonic()
                        if remaining > 0:
                            self.sleep(min(POLL_SECONDS, remaining))
                    return 0
                try:
                    size = self.byte_count(self.base)
                except Exception as exc:
                    self.event("size_check_error", error=repr(exc), size_bytes=size)
                    self.stop_target(size, "size_check_error", deadline)
                    return 2
                if size >= LIMIT_BYTES:
                    self.event("byte_limit_reached", size_bytes=size)
                    if self.target is not None:
                        self.stop_target(size, "byte_limit", deadline)
                        return 1
                    # The supervisor stops its owned timeout if discovery did
                    # not find a safe model target. Never keep running over
                    # budget without protection, and never signal a guess.
                    return 1
                remaining = deadline - self.monotonic()
                if remaining > 0:
                    self.sleep(min(POLL_SECONDS, remaining))
            self.event("guard_deadline_reached", size_bytes=size)
            return 0
        finally:
            if self.pidfd is not None:
                self.close(self.pidfd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("wrapper_session_id", type=int)
    args = parser.parse_args()
    return Guard(args.base, args.wrapper_session_id).run()


if __name__ == "__main__":
    raise SystemExit(main())
