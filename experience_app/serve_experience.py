from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    port = 8714
    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(*args, directory=str(root), **kwargs)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"DOMCS-EEG experience server running at http://127.0.0.1:{port}/experience_app/index.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
