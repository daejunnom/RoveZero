"""Owned local Linux process/cgroup capture shared by finite adapter checks."""
from __future__ import annotations
import collections
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time
import uuid

GIB = 1024**3
LOG_CAP = 2 * 1024**2

def put(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")

def outside_git(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("absolute output root required")
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError("linked output root is unsupported")
        if (part / ".git").exists():
            raise ValueError("generated evidence stays outside Git")
    return path

class OwnedRun:
    """Only this fresh cgroup and exact child group are signaled or removed."""
    def __init__(self, directory, wall=180, address_space=2**40):
        if os.name != "posix" or not Path("/sys/fs/cgroup/cgroup.controllers").exists():
            raise ValueError("local cgroup-v2 Linux execution required")
        self.started = time.monotonic()
        self.deadline = self.started + wall
        self.directory = outside_git(directory)
        self.directory.mkdir()
        self.group = Path("/sys/fs/cgroup") / ("rovezero-adapter-" + uuid.uuid4().hex)
        self.group.mkdir()
        self.child = None
        self.selector = selectors.DefaultSelector()
        self.logs = {}
        self.lines = collections.deque()
        self.pending = bytearray()
        self.total = 0
        self.forced = False
        self.affinity = sorted(os.sched_getaffinity(0))[:2]
        self.address_space = address_space
        try:
            if len(self.affinity) != 2:
                raise ValueError("two allowed CPUs required")
            for key, value in {"memory.high": 6*GIB, "memory.max": 12*GIB, "memory.swap.max": 0,
                               "pids.max": 128, "cpu.max": "200000 100000"}.items():
                (self.group/key).write_text(str(value))
        except BaseException:
            if not (self.group/"cgroup.procs").read_text().split():
                self.group.rmdir()
            raise

    def spawn(self, argv):
        import resource
        if self.child is not None or not argv:
            raise ValueError("one finite child per fresh capture required")
        def before():
            (self.group/"cgroup.procs").write_text(str(os.getpid()))
            os.sched_setaffinity(0, self.affinity)
            resource.setrlimit(resource.RLIMIT_AS, (self.address_space, self.address_space))
        temporary = self.directory/"tmp"
        temporary.mkdir()
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
               "XDG_CACHE_HOME": str(Path.home()/".cache"), "TMPDIR": str(temporary),
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "CUDA_CACHE_PATH": str(temporary/"cuda-cache"), "CUDA_CACHE_MAXSIZE": str(64*1024**2)}
        self.child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      cwd=self.directory, env=env, start_new_session=True, preexec_fn=before)
        for name, stream in [("stdout", self.child.stdout), ("stderr", self.child.stderr)]:
            os.set_blocking(stream.fileno(), False)
            self.selector.register(stream, selectors.EVENT_READ, name)
            self.logs[name] = (self.directory/(name+".log")).open("xb")
        os.set_blocking(self.child.stdin.fileno(), False)
        if sorted(os.sched_getaffinity(self.child.pid)) != self.affinity:
            raise ValueError("actual child affinity differs")
        return self.child

    def send(self, command):
        data = memoryview((command+"\n").encode("utf-8"))
        if len(data)>16384 or '\n' in command or '\r' in command or '\0' in command:
            raise ValueError("invalid finite protocol command")
        while data:
            self.check_deadline()
            try:
                count = os.write(self.child.stdin.fileno(), data)
                if not count:
                    raise ValueError("protocol write returned zero")
                data=data[count:]
            except BlockingIOError:
                self.pump()

    def check_deadline(self):
        if time.monotonic()>=self.deadline:
            raise TimeoutError("registered process wall limit exceeded")

    def pump(self):
        for key, _ in self.selector.select(timeout=.01):
            chunk = os.read(key.fileobj.fileno(), 65536)
            if not chunk:
                self.selector.unregister(key.fileobj)
                key.fileobj.close()
                continue
            self.total += len(chunk)
            if self.total>LOG_CAP:
                raise ValueError("combined log exceeds 2MiB")
            self.logs[key.data].write(chunk)
            if key.data=="stdout":
                self.pending.extend(chunk)
                while b"\n" in self.pending:
                    line, _, tail=self.pending.partition(b"\n")
                    self.pending=bytearray(tail)
                    if len(line)>65536 or len(self.lines)>=256:
                        raise ValueError("bounded protocol queue exceeded")
                    self.lines.append(line.decode("utf-8").rstrip("\r"))
                if len(self.pending)>65536:
                    raise ValueError("protocol line exceeds 64KiB")

    def until(self, predicate, observe=lambda line: None):
        while True:
            self.check_deadline()
            self.pump()
            while self.lines:
                line=self.lines.popleft()
                observe(line)
                if predicate(line):
                    return line
            if self.child.poll() is not None and not self.selector.get_map():
                raise ValueError("engine closed protocol before required response")

    def wait_exit(self, observe=lambda line: None):
        """Normal quit/drain is charged to the registered run, before cleanup."""
        if self.child.stdin and not self.child.stdin.closed:
            self.child.stdin.close()
        while self.child.poll() is None or self.selector.get_map():
            self.check_deadline()
            self.pump()
            while self.lines:
                observe(self.lines.popleft())
        if self.pending:
            raise ValueError("unterminated final protocol line")
        return self.child.wait(timeout=.1)

    def finish(self):
        error = None
        # A single 30-second cleanup budget covers TERM then KILL and drain.
        cleanup_deadline=time.monotonic()+30
        try:
            if self.child is not None:
                if self.child.poll() is None:
                    self.forced=True
                    os.killpg(self.child.pid, signal.SIGTERM)
                term_deadline=min(cleanup_deadline,time.monotonic()+15)
                while self.child.poll() is None and time.monotonic()<term_deadline:
                    self.pump()
                    self.lines.clear()
                if self.child.poll() is None:
                    (self.group/"cgroup.kill").write_text("1")
                    self.forced=True
                while (self.child.poll() is None or self.selector.get_map()) and time.monotonic()<cleanup_deadline:
                    self.pump()
                    self.lines.clear()
                if self.child.poll() is None:
                    raise TimeoutError("owned child cleanup unconfirmed")
                self.child.wait(timeout=max(.01,cleanup_deadline-time.monotonic()))
                if self.pending or self.selector.get_map():
                    raise ValueError("incomplete pipe drain")
                if self.child.stdin and not self.child.stdin.closed:
                    self.child.stdin.close()
        except BaseException as exc:
            error=f"{type(exc).__name__}: {exc}"
        remaining=(self.group/"cgroup.procs").read_text().split()
        if remaining:
            (self.group/"cgroup.kill").write_text("1")
            self.forced=True
            while (self.group/"cgroup.procs").read_text().split() and time.monotonic()<cleanup_deadline:
                time.sleep(.01)
            remaining=(self.group/"cgroup.procs").read_text().split()
        # Reap the exact child even if a log error interrupted the normal drain.
        if self.child is not None and not remaining:
            try:
                self.child.wait(timeout=max(.01, cleanup_deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                error=error or "owned child reap unconfirmed"
        for file in self.logs.values():
            file.close()
        if self.child is not None:
            for stream in (self.child.stdin, self.child.stdout, self.child.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
        self.selector.close()
        resources={key:(self.group/key).read_text() for key in
                   ("memory.peak","memory.events","memory.stat","cpu.stat","pids.peak")}
        if not remaining:
            self.group.rmdir()
            import shutil
            # Only this owned transient directory, after the whole cgroup is empty.
            temporary=self.directory/"tmp"
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=False)
        return dict(whole_wall_seconds=time.monotonic()-self.started, resources=resources,
                    affinity=self.affinity,memory_high=6*GIB,memory_max=12*GIB,swap_max=0,
                    address_space_limit_bytes=self.address_space,
                    remaining_owned_processes=remaining,forced_cleanup=self.forced,
                    cleanup_error=error,exit_code=self.child.returncode if self.child else None,
                    vram_peak="unknown",windows_commit_peak="unknown",temporary_removed=not remaining)

