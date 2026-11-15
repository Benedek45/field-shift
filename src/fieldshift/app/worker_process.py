"""Shell-free background worker supervisor; safe to import before spawning."""

import os
import subprocess
import sys


def detached_options(platform=None):
    if (platform or os.name) == "nt":
        # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS; no POSIX-only setsid.
        return {"creationflags": 0x00000200 | 0x00000008}
    return {"start_new_session": True}


def main(args=None):
    args = sys.argv[1:] if args is None else args
    if len(args) != 2:
        print("Usage: worker_process <solve|ask> <session-directory>", file=sys.stderr)
        return 2
    try:
        code = subprocess.run([sys.executable, "-u", "-m", "fieldshift.app.advisor_worker", *args],
                              check=False).returncode
    except OSError as exc:
        print(f"Could not start the advisor worker: {exc}", file=sys.stderr)
        code = 1
    print(f"worker exited with status {code}", flush=True)
    return code if code >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
