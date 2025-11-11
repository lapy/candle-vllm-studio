import paramiko
import base64

HOST = "10.0.0.129"
USER = "root"
KEY_PATH = r"C:\\Users\\vlapy\\.ssh\\id_rsa"
PASSPHRASE = "150156"

script = '''import requests
url = "http://127.0.0.1:4000/v1/chat/completions"
payload = {
    "model": "default",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one word."},
    ],
    "stream": False,
    "max_tokens": 16,
    "temperature": 0.7,
}
resp = requests.post(url, json=payload, timeout=60)
print(resp.status_code)
print(resp.text[:500])
'''
encoded = base64.b64encode(script.encode()).decode()

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
key = paramiko.RSAKey.from_private_key_file(KEY_PATH, password=PASSPHRASE)
client.connect(hostname=HOST, username=USER, pkey=key, timeout=20)

cmd = (
    "docker exec candle-vllm-studio bash -lc \"python - <<'PY'\n"
    "import base64\n"
    f"code = base64.b64decode('{encoded}')\n"
    "namespace = {}\n"
    "exec(code.decode('utf-8'), namespace)\n"
    "PY\""
)
stdin, stdout, stderr = client.exec_command(cmd, timeout=180)
print(stdout.read().decode(errors='ignore'))
err = stderr.read().decode(errors='ignore')
if err:
    print('err:', err)

client.close()
