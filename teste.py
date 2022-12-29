import socket
mysock =  socket.socket(socket.AF_INET, socket.SOCK_STREAM)

mysock.connect(('192.168.0.121', 81))
cmd = 'GET http://192.168.0.121/stream HTTP/1.0\r\n\r\n'.encode("ISO-8859-1")
mysock.send(cmd)

while True :
    data = mysock.recv(1084)
    if (len(data) < 1):
        break
    print(data.decode("ISO-8859-1"))
mysock.close()