"""Drive one interactive lerobot-rollout run unattended.

Usage: drive_run.py <log file> <run seconds> -- <command...>
Starts the command with stdin piped, mirrors its output to <log file>, sends /start 3 s after the rollout reports
"robot idle until /start", sends /reset after <run seconds>, then /stop 25 s later (the guarded return move can take 20 s), and exits with the command's code.
Kill this driver (or the run) at any time: the rollout's own shutdown then disables torque.
"""
import os, subprocess, sys, threading, time

log_path, run_s = sys.argv[1], float(sys.argv[2])
cmd = sys.argv[sys.argv.index("--") + 1:]
env = {**os.environ, "PYTHONUNBUFFERED": "1"}
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
ready = threading.Event()
log = open(log_path, "w")

def pump():
    for line in p.stdout:
        log.write(line); log.flush()
        if "robot idle until /start" in line or "Interactive rollout session" in line:
            ready.set()
    log.close()

threading.Thread(target=pump, daemon=True).start()

def send(s):
    try:
        p.stdin.write(s + "\n"); p.stdin.flush()
        print(f"[drive_run] {time.strftime('%H:%M:%S')} sent {s}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[drive_run] could not send {s}: {e}", flush=True)

if not ready.wait(timeout=120):
    print("[drive_run] rollout never became ready; stopping", flush=True)
    send("/stop"); p.wait(timeout=30); sys.exit(p.returncode or 1)
time.sleep(3); send("/start")
t_end = time.time() + run_s
while time.time() < t_end and p.poll() is None:
    time.sleep(0.5)
if p.poll() is None:
    send("/reset"); time.sleep(25); send("/stop")  # the return move may take up to ~20 s
try:
    p.wait(timeout=60)
except subprocess.TimeoutExpired:
    p.terminate(); p.wait(timeout=15)
print(f"[drive_run] rollout exited with {p.returncode}", flush=True)
sys.exit(p.returncode or 0)
