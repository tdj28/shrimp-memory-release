#!/usr/bin/env python3
"""Own the output guard and bounded timeout; fail closed if guarding stops.

Usage: python3 tools/supervise_readback_output.py BASE WRAPPER_SID --
       timeout --signal=TERM --kill-after=30s 2760s COMMAND ...

Return the timeout command's shell status on normal completion, or 74 on a
guard/readiness/supervision failure. Both children remain in the wrapper's
session. Only retained pidfds are used for routine signals; GNU timeout
forwards TERM and performs its configured escalation for its command.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import guard_readback_output as byte_guard

FAILURE = 74
READY_SECONDS = 5
TIMEOUT_STOP_SECONDS = 35
GUARD_STOP_SECONDS = 5


def validate_command(command):
    if (len(command) < 5 or Path(command[0]).name != "timeout"
            or command[1:3] != ["--signal=TERM", "--kill-after=30s"]):
        raise ValueError("expected timeout --signal=TERM --kill-after=30s DURATION COMMAND")
    if not command[3].endswith("s") or not 0 < float(command[3][:-1]) <= 2760:
        raise ValueError("timeout duration must be positive, in seconds, and at most 2760s")


class Supervisor:
    def __init__(self, base, wrapper_sid, command, *, popen=subprocess.Popen,
                 pidfd_open=None, send_signal=None,
                 close=os.close, monotonic=time.monotonic, sleep=time.sleep,
                 getsid=os.getsid, getpid=os.getpid, guard_factory=byte_guard.Guard):
        self.base, self.sid, self.command = Path(base), wrapper_sid, list(command)
        if not self.base.is_absolute() or not self.base.is_dir() or wrapper_sid <= 1:
            raise ValueError("BASE must be an existing absolute directory; SID must exceed 1")
        validate_command(self.command)
        if getsid(getpid()) != wrapper_sid:
            raise ValueError("supervisor is outside the supplied wrapper session")
        self.popen = popen
        self.pidfd_open = pidfd_open if pidfd_open is not None else getattr(os, "pidfd_open", None)
        self.send_signal = send_signal if send_signal is not None else getattr(signal, "pidfd_send_signal", None)
        if not callable(self.pidfd_open) or not callable(self.send_signal):
            raise ValueError("Linux pidfd APIs are required")
        self.close, self.monotonic, self.sleep = close, monotonic, sleep
        self.guard_factory = guard_factory
        self.children = {}

    def event(self, event, **fields):
        line = json.dumps({"utc": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
                           "event": event, **fields}, sort_keys=True) + "\n"
        print(line, end="", file=sys.stderr, flush=True)
        try:
            with (self.base / "output-supervisor.jsonl").open("a") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            print(f"supervisor event append failed: {exc!r}", file=sys.stderr, flush=True)

    def spawn(self, name, command, log_name):
        with (self.base / log_name).open("xb") as stream:
            child = self.popen(command, stdin=subprocess.DEVNULL,
                               stdout=stream, stderr=subprocess.STDOUT)
        try:
            handle = self.pidfd_open(child.pid, 0)
        except BaseException:
            # No poll/wait has happened: this direct child cannot have its PID
            # reused, even if it has already become a zombie. Popen owns it;
            # no other thread or SIGCHLD handler reaps children in this program.
            child.kill()
            child.wait(timeout=GUARD_STOP_SECONDS)
            raise
        self.children[name] = (child, handle)
        self.event("child_started", child=name, pid=child.pid)
        return child

    def send(self, name, number):
        child, handle = self.children[name]
        try:
            self.send_signal(handle, number, None, 0)
        except ProcessLookupError:
            return
        self.event("signal_sent", child=name, pid=child.pid, signal=number)

    def stop_timeout(self):
        child = self.children["timeout"][0]
        if child.poll() is not None:
            return child.returncode
        self.send("timeout", signal.SIGTERM)
        try:
            status = child.wait(timeout=TIMEOUT_STOP_SECONDS)
        except subprocess.TimeoutExpired:
            self.event("timeout_escalation_failed")
            # This is a last resort for a broken timeout contract. The existing
            # guard matcher binds only the exact model command in this session.
            emergency = self.guard_factory(self.base, self.sid)
            try:
                emergency.discover()
                emergency.signal_target(signal.SIGKILL, None, "supervisor_emergency")
            finally:
                if emergency.pidfd is not None:
                    emergency.close(emergency.pidfd)
            self.send("timeout", signal.SIGKILL)
            status = child.wait(timeout=GUARD_STOP_SECONDS)
        self.event("timeout_stopped", status=status)
        return status

    def cleanup(self):
        errors = []
        if "timeout" in self.children:
            try:
                self.stop_timeout()
            except BaseException as exc:
                errors.append(repr(exc))
        if "guard" in self.children:
            child = self.children["guard"][0]
            try:
                if child.poll() is None:
                    self.send("guard", signal.SIGTERM)
                    try:
                        child.wait(timeout=GUARD_STOP_SECONDS)
                    except subprocess.TimeoutExpired:
                        self.send("guard", signal.SIGKILL)
                        child.wait(timeout=GUARD_STOP_SECONDS)
            except BaseException as exc:
                errors.append(repr(exc))
        for _, handle in self.children.values():
            try:
                self.close(handle)
            except OSError as exc:
                errors.append(repr(exc))
        if errors:
            self.event("cleanup_error", errors=errors)
        return not errors

    def ready(self, child):
        deadline = self.monotonic() + READY_SECONDS
        marker = self.base / "output-byte-guard.ready"
        while self.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError(f"guard exited before readiness: {child.returncode}")
            if marker.is_file() and marker.stat().st_size:
                record = json.loads(marker.read_text())
                if (record.get("limit_bytes") != byte_guard.LIMIT_BYTES
                        or not 0 <= record.get("initial_size_bytes", -1) < byte_guard.LIMIT_BYTES):
                    raise RuntimeError("invalid guard readiness marker")
                return
            self.sleep(min(0.1, deadline - self.monotonic()))
        raise RuntimeError("guard did not become ready within five seconds")

    def run(self):
        status = FAILURE
        try:
            if (self.base / "output-byte-guard.ready").exists():
                raise RuntimeError("pre-existing guard readiness marker")
            guard = self.spawn("guard", ["python3", "tools/guard_readback_output.py",
                                        str(self.base), str(self.sid)],
                               "output-byte-guard-console.log")
            self.ready(guard)
            if guard.poll() is not None:
                raise RuntimeError("guard exited at readiness")
            timeout = self.spawn("timeout", self.command, "console.log")
            while True:
                guard_result = guard.poll()
                if guard_result is not None:
                    raise RuntimeError(f"guard exited while timeout was active: {guard_result}")
                result = timeout.poll()
                if result is not None:
                    self.event("timeout_completed", status=result)
                    status = result if result >= 0 else 128 - result
                    break
                self.sleep(0.1)
        except (Exception, KeyboardInterrupt) as exc:
            self.event("supervision_failure", error=repr(exc))
        finally:
            if not self.cleanup():
                status = FAILURE
        self.event("supervisor_finished", status=status)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("wrapper_session_id", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command

    def interrupted(signum, frame):
        raise InterruptedError(f"supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        return Supervisor(args.base, args.wrapper_session_id, command).run()
    except (ValueError, OSError) as exc:
        print(f"supervisor preflight failed: {exc}", file=sys.stderr, flush=True)
        return FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
