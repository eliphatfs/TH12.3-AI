# -*- coding: utf-8 -*-

import socket
import random
import traceback

ip_port = ('127.0.0.1', 5211)

web = socket.socket()
web.bind(ip_port)
web.listen(10)

while True:
    try:
        conn, addr = web.accept()
        while True:
            try:
                conn.send(("%d %d" % (random.randint(0, 45),
                                      random.randint(0, 45)))
                          .encode())
                print(conn.recv(255).decode())
            except KeyboardInterrupt:
                break
            except Exception:
                traceback.print_exc()
                break
    except KeyboardInterrupt:
        break
    except Exception:
        traceback.print_exc()
