import subprocess
import time
while True:
    result = subprocess.run(["python3", "scripts/wordpress/normal-performance.py"])
    if result.returncode != 0:
        print("Python script exited unexpectedly. Restarting...")
    else:
        print("Python script exited normally. No restart needed.")
        break
    time.sleep(0.5)

