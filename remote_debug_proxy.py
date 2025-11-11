import paramiko
import time
host='10.0.0.129'
username='root'
key_path=r'C:\\Users\\vlapy\\.ssh\\id_rsa'
passphrase='150156'
client=paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
key=paramiko.RSAKey.from_private_key_file(key_path,password=passphrase)
client.connect(hostname=host,username=username,pkey=key)
# ensure model stopped
stdin,stdout,stderr=client.exec_command("docker exec candle-vllm-studio bash -lc 'curl -s -X POST http://127.0.0.1:8080/api/models/1/stop'")
stdout.read(); stderr.read()
# start model
stdin,stdout,stderr=client.exec_command("docker exec candle-vllm-studio bash -lc 'curl -s -X POST http://127.0.0.1:8080/api/models/1/start'")
start_out=stdout.read().decode(); start_err=stderr.read().decode()
print('start:', start_out)
if start_err:
    print('start err:', start_err)
client.exec_command("docker exec candle-vllm-studio bash -lc 'sleep 8'")
# list models
stdin,stdout,stderr=client.exec_command("docker exec candle-vllm-studio bash -lc 'curl -s http://127.0.0.1:3000/v1/models'")
print('models:', stdout.read().decode())
err=stderr.read().decode()
if err:
    print('models err:', err)
# streaming chat completion
payload='{"model":"Qwen3-4B.Q2_K","messages":[{"role":"user","content":"Hello"}]}'
cmd=f"docker exec candle-vllm-studio bash -lc 'cat <<\'EOF\' | curl -s -N -X POST http://127.0.0.1:3000/v1/chat/completions -H \"Content-Type: application/json\" --data-binary @-\n{payload}\nEOF'"
stdin,stdout,stderr=client.exec_command(cmd)
print('completion sample:', stdout.read().decode(errors='ignore')[:2000])
comp_err=stderr.read().decode(errors='ignore')
if comp_err:
    print('completion err:', comp_err)
client.close()
