import json
import os
import sys
import threading
import time
import urllib.request

MODEL = "qwen3:4b"
PROMPT = "Write a short story about a robot learning to cook."
TOKENS = 200
URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def ask(results, latencies, i):
    sent = time.time()
    body = json.dumps({
        "model": MODEL,
        "prompt": PROMPT,
        "stream": False,
        "think": False,
        "options": {"num_predict": TOKENS, "temperature": 0},
    }).encode()
    req = urllib.request.Request(f"{URL}/api/generate", body,
                                 {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req).read())
    results[i] = r["eval_count"] / (r["eval_duration"] / 1e9)
    latencies[i] = time.time() - sent


def run(users, label=""):
    results = [0] * users
    latencies = [0] * users
    threads = [threading.Thread(target=ask, args=(results, latencies, i)) for i in range(users)]
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - start
    total = users * TOKENS / wall
    print(f"{label}{users} user(s): gen speed {sum(results)/users:6.1f} tok/s | "
          f"total {total:6.1f} tok/s | user waits avg {sum(latencies)/users:5.1f}s "
          f"worst {max(latencies):5.1f}s")


run(1, "[warm-up] ")
for n in [int(x) for x in sys.argv[1:]] or [1, 2, 4]:
    run(n)
