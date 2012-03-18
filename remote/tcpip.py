#!/lab/software/epd-7.0-1-rh5-x86/bin/python
import socket

HOST = ''       # Symbolic name meaning the local host
PORT = 50010    # Arbitrary non-privileged port

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
conn, addr = s.accept()
print 'Connected by', addr
while 1:
   data = conn.recv(1024)
   if not data: break
   conn.send(data)
conn.close()



