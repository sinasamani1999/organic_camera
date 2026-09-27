import socket 
sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
sock.bind(("127.0.0.1",8000))
print("receiver is listening")
while True :
    data,addr = sock.recvfrom (1024)
    print(f"Received message from {addr}: {data}")