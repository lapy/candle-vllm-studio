import paramiko
import json

HOST = "10.0.0.129"
USER = "root"
KEY_PATH = r"C:\\Users\\vlapy\\.ssh\\id_rsa"
PASSPHRASE = "150156"

payload = json.dumps({
    "model": "default",
    "messages": [{"role": "user", "content": "Hello manual"}],
    "max_tokens": 32,
    "stream": False,
})
payload_escaped = payload.replace("\"", "\\\"")

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
key = paramiko.RSAKey.from_private_key_file(KEY_PATH, password=PASSPHRASE)
client.connect(hostname=HOST, username=USER, pkey=key)

client.exec_command("docker exec candle-vllm-studio bash -lc 'curl -s -X POST http://127.0.0.1:8080/api/models/1/stop'", timeout=30)

run_cmd = (
    "docker exec candle-vllm-studio bash -lc 'set -euo pipefail; LOG=/tmp/candle_manual.log; "
    "RESP=/tmp/candle_resp.txt; rm -f $LOG $RESP; "
    "(/app/data/candle-builds/cuda-rel-master-cuda-flash-attn-graph-nccl/candle-vllm "
    "--w /app/data/models --f /app/data/models/Qwen3-4B.Q2_K.gguf --h 0.0.0.0 --p 4100 "
    "--mem 6881 --max-num-seqs 2 --block-size 205 --temperature 0.32 --top-p 0.74 "
    "--min-p 0.032 --top-k 59 --frequency-penalty 0.01 --presence-penalty 0.004 "
    "--prefill-chunk-size 1024 --d 0) > $LOG 2>&1 & pid=$!; "
    "for i in $(seq 1 60); do if curl -sf http://127.0.0.1:4100/v1/models >/dev/null 2>&1; then break; fi; sleep 1; done; "
    "curl -sS -o $RESP -w \"HTTP:%{http_code}\" -X POST http://127.0.0.1:4100/v1/chat/completions "
    "-H \"Content-Type: application/json\" -d \"" + payload_escaped + "\" || true; "
    "sleep 2; kill $pid 2>/dev/null || true; wait $pid 2>/dev/null || true'"
)

stdin, stdout, stderr = client.exec_command(run_cmd)
stdout.channel.settimeout(600)
stderr.channel.settimeout(600)
print(stdout.read().decode(errors='ignore'))
err = stderr.read().decode(errors='ignore')
if err:
    print('err:', err)

# fetch outputs
for path in ["/tmp/candle_manual.log", "/tmp/candle_resp.txt"]:
    cmd = f"docker exec candle-vllm-studio bash -lc 'cat {path} 2>/dev/null'"
    stdin, stdout, stderr = client.exec_command(cmd)
    data = stdout.read().decode(errors='ignore')
    if data:
        print(f"=== {path} ===\n{data}")

client.close()
