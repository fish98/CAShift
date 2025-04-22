import subprocess
import time
import sys

# check args
if len(sys.argv) != 2:
    print("Usage: python daemon.py [service url e.g, http://192.168.49.2:xxxx/]")
    sys.exit(1)

ip = sys.argv[1]

while True:
    result = subprocess.run(["python3", "./normal-performance.py", ip])
    if result.returncode != 0:
        print("Python script exited unexpectedly. Restarting...")
    else:
        print("Python script exited normally. No restart needed.")
        break
    time.sleep(0.5)

