"""OS-independent locking, worker launch and CPU execution; no data or API keys."""

import errno
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from fieldshift import locking
from fieldshift.agent import farms
from fieldshift.app import advisor_core as core
from fieldshift.app import worker_process


def test_lock_blocks_another_process_and_releases(tmp_path):
    path = tmp_path / "lock with spaces"
    probe = (
        "from fieldshift.locking import file_lock; import sys;"
        "\ntry:\n with file_lock(sys.argv[1], blocking=False): pass"
        "\nexcept BlockingIOError: sys.exit(3)"
    )
    with locking.file_lock(path):
        p = subprocess.run([sys.executable, "-c", probe, str(path)], check=False,
                           capture_output=True, text=True)
        assert p.returncode == 3, p.stderr
    p = subprocess.run([sys.executable, "-c", probe, str(path)], check=False,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def test_lock_released_on_process_death(tmp_path):
    path = tmp_path / "job.lock"
    child = subprocess.Popen(
        [sys.executable, "-u", "-c",
         ("import sys,time; from fieldshift.locking import file_lock;"
          "\nwith file_lock(sys.argv[1]):\n print('locked',flush=True); time.sleep(60)"), str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout.readline().strip() == "locked"
        with pytest.raises(BlockingIOError), locking.file_lock(path, blocking=False):
            pass
    finally:
        child.terminate()
        child.communicate(timeout=10)
    with locking.file_lock(path, blocking=False):
        pass


def test_windows_lock_uses_first_byte_and_handles_contention(tmp_path, monkeypatch):
    calls = []
    fake = SimpleNamespace(LK_NBLCK=2, LK_UNLCK=0,
                           locking=lambda fd, mode, size: calls.append((fd, mode, size)))
    monkeypatch.setattr(locking, "WINDOWS", True)
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    with (tmp_path / "lock").open("a+b") as f:
        locking.acquire(f)
        assert f.tell() == 0
        locking.release(f)
        assert (tmp_path / "lock").read_bytes() == b"\0"
    assert [c[1:] for c in calls] == [(2, 1), (0, 1)]

    def busy(*args):
        raise OSError(errno.EACCES, "locked")

    fake.locking = busy
    with (tmp_path / "lock").open("a+b") as f, pytest.raises(BlockingIOError):
        locking.acquire(f, blocking=False)


def test_windows_blocking_lock_retries_only_contention(tmp_path, monkeypatch):
    attempts = []

    def lock(fd, mode, size):
        attempts.append(mode)
        if len(attempts) == 1:
            raise OSError(errno.EACCES, "locked")

    monkeypatch.setattr(locking, "WINDOWS", True)
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(LK_NBLCK=2, locking=lock))
    monkeypatch.setattr(locking.time, "sleep", lambda t: None)
    with (tmp_path / "lock").open("a+b") as f:
        locking.acquire(f)
    assert attempts == [2, 2]


def test_farm_registry_concurrent_writers(tmp_path, monkeypatch):
    monkeypatch.setattr(farms, "validate", lambda record: record)
    with ThreadPoolExecutor(4) as pool:
        ids = list(pool.map(lambda _: farms.add({"adm_id": "US-19-169", "irrigated": False},
                                              root=tmp_path), range(12)))
    assert len(set(ids)) == 12
    assert len(json.loads((tmp_path / "farms.json").read_text())) == 12


def test_job_lock_probe(tmp_path):
    assert not core._lock_held(tmp_path)
    with core.job_lock(tmp_path):
        assert core._lock_held(tmp_path)
        with pytest.raises(core.JobBusy), core.job_lock(tmp_path):
            pass
    assert not core._lock_held(tmp_path)


def test_shell_free_worker_launch(tmp_path, monkeypatch):
    invoked = []
    monkeypatch.setattr(core.subprocess, "Popen",
                        lambda args, **kw: (invoked.append((args, kw)) or SimpleNamespace(pid=123)))
    core.spawn(tmp_path, "ask", env={"FAKE_WORKER_KEY": "secret-not-on-argv"})
    args, kw = invoked[0]
    assert args == [sys.executable, "-u", "-m", "fieldshift.app.worker_process", "ask", str(tmp_path)]
    assert kw["env"]["FAKE_WORKER_KEY"] == "secret-not-on-argv"
    assert "secret-not-on-argv" not in " ".join(args)
    assert "secret-not-on-argv" not in (tmp_path / "worker.log").read_text()
    assert not kw.get("shell")
    assert worker_process.detached_options("nt") == {"creationflags": 0x208}
    assert worker_process.detached_options("posix") == {"start_new_session": True}


def test_worker_supervisor_records_exit(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(worker_process.subprocess, "run",
                        lambda args, **kw: (called.append(args) or SimpleNamespace(returncode=7)))
    assert worker_process.main(["ask", "C:\\Farm Sessions\\one"]) == 7
    assert called[0][-1] == "C:\\Farm Sessions\\one"
    assert "worker exited with status 7" in capsys.readouterr().out


def test_imports_and_cpu_execution_in_clean_process():
    env = {**os.environ, "FIELDSHIFT_DEVICE": "cpu"}
    p = subprocess.run(
        [sys.executable, "-c",
         ("import torch; from fieldshift.sim.components import accelerator;"
         "from fieldshift.model.neural import accelerator_device;"
         "from fieldshift.agent import farms; from fieldshift.app import advisor_worker;"
         "assert accelerator().type == accelerator_device().type == 'cpu';"
         "assert torch.isfinite(torch.ones(3, device=accelerator()) * 2).all()")],
        env=env, capture_output=True, text=True, check=False,
    )
    assert p.returncode == 0, p.stderr


@pytest.mark.parametrize("name", [
    "data/processed/weather/power_county_daily.json",
    r"data\processed\weather\power_county_daily.json",
    r"C:\checkout\data\processed\weather\power_county_daily.json",
])
def test_missing_private_data_paths_are_recognized_on_each_os(name):
    from conftest import _missing_data_error
    exc = FileNotFoundError(errno.ENOENT, "No such file or directory", name)
    assert _missing_data_error(SimpleNamespace(value=exc))
    wrapped = RuntimeError(str(exc))
    wrapped.__cause__ = exc
    assert _missing_data_error(SimpleNamespace(value=wrapped))
    assert not _missing_data_error(SimpleNamespace(
        value=FileNotFoundError(errno.ENOENT, "No such file or directory", "src/missing.py")))
