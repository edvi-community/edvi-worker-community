"""Packaged browser assets and same-origin protection for local services."""
from importlib.resources import files


def allow_browser_request(handler):
    port = handler.server.server_port
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"{handler.server.server_address[0]}:{port}"}
    host = handler.headers.get("Host", "")
    origin = handler.headers.get("Origin")
    return (host in hosts and (origin is None or origin == f"http://{host}")
            and handler.headers.get("Sec-Fetch-Site") != "cross-site")


def serve_asset(handler, page):
    routes = {"/": (page + ".html", "text/html"),
              "/app.css": ("app.css", "text/css"),
              "/dashboard.js": ("dashboard.js", "text/javascript")}
    if handler.command != "GET" or handler.path not in routes:
        return False
    name, content_type = routes[handler.path]
    raw = files("edvi_worker").joinpath("static", name).read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", content_type + "; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.send_header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
    handler.end_headers()
    handler.wfile.write(raw)
    return True
