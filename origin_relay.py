import os
import select
import socket
import socketserver


LISTEN_HOST = os.getenv("ORIGIN_RELAY_LISTEN_HOST", "127.0.0.1")
LISTEN_PORT = int(os.getenv("ORIGIN_RELAY_LISTEN_PORT", "8001"))
TARGET_HOST = os.getenv("ORIGIN_RELAY_TARGET_HOST", "app")
TARGET_PORT = int(os.getenv("ORIGIN_RELAY_TARGET_PORT", "8000"))
BUFFER_SIZE = 64 * 1024


class ProxyHandler(socketserver.BaseRequestHandler):
    def handle(self):
        with socket.create_connection((TARGET_HOST, TARGET_PORT), timeout=10) as upstream:
            sockets = [self.request, upstream]
            while True:
                readable, _, _ = select.select(sockets, [], [], 60)
                if not readable:
                    return
                for sock in readable:
                    data = sock.recv(BUFFER_SIZE)
                    if not data:
                        return
                    (upstream if sock is self.request else self.request).sendall(data)


class ThreadedTcpServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    if ":" in LISTEN_HOST:
        ThreadedTcpServer.address_family = socket.AF_INET6

    with ThreadedTcpServer((LISTEN_HOST, LISTEN_PORT), ProxyHandler) as server:
        print(
            f"Origin relay listening on {LISTEN_HOST}:{LISTEN_PORT} -> {TARGET_HOST}:{TARGET_PORT}",
            flush=True,
        )
        server.serve_forever()
