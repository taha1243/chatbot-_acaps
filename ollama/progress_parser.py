import sys
import json
import time

start_time = time.time()
last_pct = -1

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        continue

    status = data.get("status", "")
    total = data.get("total", 0)
    completed = data.get("completed", 0)

    if total > 0 and completed >= 0:
        pct = int(completed * 100 / total)
        mb_done = completed / 1048576
        mb_total = total / 1048576
        elapsed = time.time() - start_time

        if pct != last_pct and pct % 5 == 0:
            if elapsed > 3 and completed > 0:
                speed = completed / elapsed
                remaining = (total - completed) / speed if speed > 0 else 0
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                mb_s = speed / 1048576
                print(f">>> [{pct:3d}%] {mb_done:.0f} MB / {mb_total:.0f} MB  —  {mb_s:.1f} MB/s  ETA: {mins}m{secs:02d}s", flush=True)
            else:
                print(f">>> [{pct:3d}%] {mb_done:.0f} MB / {mb_total:.0f} MB", flush=True)
            last_pct = pct
    elif status and total == 0:
        print(f">>> {status}", flush=True)
