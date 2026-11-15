"""Kernel-released file locks on Windows and Unix, including nonblocking probes."""

import contextlib
import errno
import os
import time

WINDOWS = os.name == "nt"


def acquire(stream, *, blocking=True):
    if not WINDOWS:
        import fcntl

        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        return
    import msvcrt

    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b"\0")
        stream.flush()
    while True:
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise
            if not blocking:
                raise BlockingIOError(exc.errno, "Another process holds this lock") from exc
            time.sleep(0.05)


def release(stream):
    if WINDOWS:
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream, fcntl.LOCK_UN)


@contextlib.contextmanager
def file_lock(path, *, blocking=True):
    with open(path, "a+b") as stream:
        acquire(stream, blocking=blocking)
        try:
            yield stream
        finally:
            release(stream)
