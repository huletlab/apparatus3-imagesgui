import paramiko
import getpass
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy( paramiko.AutoAddPolicy() )
pwd=getpass.getpass('password for pedro')
ssh.connect('128.42.181.230', username='pedro', password=pwd)
stdin, stdout, stderr = ssh.exec_command("uptime")
print stdin
print stdout
print stderr
ssh.close()


