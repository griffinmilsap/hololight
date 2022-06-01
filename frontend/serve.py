import http.server, ssl, os

server_address = ('0.0.0.0', 443)
httpd = http.server.HTTPServer(server_address, http.server.SimpleHTTPRequestHandler)
httpd.socket = ssl.wrap_socket(httpd.socket,
                               server_side=True,
                               certfile=os.environ.get('LOCAL_CERT'),
                               ssl_version=ssl.PROTOCOL_TLS)
httpd.serve_forever()
