import paramiko
host = '10.0.0.129'
username = 'root'
key_path = r'C:\Users\vlapy\.ssh\id_rsa'
passphrase = '150156'
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
key = paramiko.RSAKey.from_private_key_file(key_path, password=passphrase)
client.connect(hostname=host, username=username, pkey=key)
client.exec_command("docker exec candle-vllm-studio bash -lc 'curl -s -X POST http://127.0.0.1:8080/api/models/1/start'")
payload = '{"model":"Qwen3-4B.Q2_K","messages":[{"role":"user","content":"Hello"}]}'
# write payload into container temp file
client.exec_command("docker exec candle-vllm-studio bash -lc 'cat > /tmp/payload.json <<\'EOF\'\n" + payload + "\nEOF'")
cmd = "docker exec candle-vllm-studio bash -lc 'curl -s -X POST http://127.0.0.1:3000/v1/chat/completions -H \"Content-Type: application/json\" --data-binary @/tmp/payload.json'"
stdin, stdout, stderr = client.exec_command(cmd)
print(stdout.read().decode(errors='ignore')[:2000])
err = stderr.read().decode(errors='ignore')
if err:
    print('err:', err)
client.close()
