import time, subprocess, sys
result = subprocess.run([sys.executable, 'train_lenet.py'], capture_output=True, text=True, timeout=600)
with open('train_log.txt', 'w') as f:
    f.write(result.stdout)
    if result.stderr:
        f.write('\nSTDERR:\n' + result.stderr)
# Print just last few lines to console
lines = result.stdout.strip().split('\n')
for line in lines[-5:]:
    print(line)
