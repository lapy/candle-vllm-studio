import paramiko
host='10.0.0.129'
username='root'
key_path=r'C:\\Users\\vlapy\\.ssh\\id_rsa'
passphrase='150156'
payload='{"model":"Qwen3-4B.Q2_K","messages":[{"role":"user","content":"Hello"}]}'
cmd=f"docker exec candle-vllm-studio bash -lc 'cat <<\'EOF\' | curl -s -N -X POST http://127.0.0.1:3000/v1/chat/completions -H \"Content-Type: application/json\" --data-binary @-\n{payload}\nEOF'"
client=paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
key=paramiko.RSAKey.from_private_key_file(key_path,password=passphrase)
client.connect(hostname=host,username=username,pkey=key)
stdin,stdout,stderr=client.exec_command(cmd)
print(stdout.read().decode(errors='ignore')[:2000])
err=stderr.read().decode(errors='ignore')
if err:
    print(err)
client.close()
