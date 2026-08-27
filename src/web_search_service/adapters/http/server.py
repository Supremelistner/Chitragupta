# HTTP Server
from __future__ import annotations
import json, logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from web_search_service.application.retrieval import RetrievalService
from web_search_service.domain.models import ContentFormat, CrawlRequest, DownloadRequest, FetchRequest, ScrapeRequest, SearchQuery

logger = logging.getLogger("web_search_service.http")

class _Handler(BaseHTTPRequestHandler):
    svc: RetrievalService
    def do_GET(self):
        p = urlparse(self.path).path
        if p in ("/healthz","/readyz","/"):
            self._json(HTTPStatus.OK, self.svc.health())
        else: self._json(HTTPStatus.NOT_FOUND, {"error":"Not found"})
    def do_POST(self):
        p = urlparse(self.path).path
        try:
            b = self._body()
            if p=="/search":
                q=b.get("query","")
                if not q: self._json(HTTPStatus.BAD_REQUEST,{"error":"query required"}); return
                r=self.svc.search(SearchQuery(query=q,limit=int(b.get("limit",10))))
                self._json(HTTPStatus.OK,{"query":r.query,"provider":r.provider.value,"results":[{"url":x.url,"title":x.title,"snippet":x.snippet} for x in r.results]})
            elif p=="/scrape":
                u=b.get("url","")
                if not u: self._json(HTTPStatus.BAD_REQUEST,{"error":"url required"}); return
                r=self.svc.scrape(ScrapeRequest(url=u,format=ContentFormat(b.get("format","markdown"))))
                self._json(HTTPStatus.OK,{"url":r.url,"title":r.title,"content":r.content[:5000],"provider":r.provider.value})
            elif p=="/fetch":
                u=b.get("url","")
                if not u: self._json(HTTPStatus.BAD_REQUEST,{"error":"url required"}); return
                r=self.svc.fetch(FetchRequest(url=u,format=ContentFormat(b.get("format","markdown"))))
                self._json(HTTPStatus.OK,{"url":r.url,"content":r.content[:5000],"status":r.status_code})
            elif p=="/download":
                u=b.get("url","")
                if not u: self._json(HTTPStatus.BAD_REQUEST,{"error":"url required"}); return
                r=self.svc.download(DownloadRequest(
                    url=u,
                    filename=b.get("filename"),
                    timeout_ms=int(b.get("timeout_ms", 30000)),
                    max_size_bytes=int(b.get("max_size_bytes", 50*1024*1024)),
                ))
                self._json(HTTPStatus.OK,{"url":r.url,"status":r.status.value,"file_path":r.file_path,"filename":r.filename,"content_type":r.content_type,"content_length":r.content_length,"error":r.error})
            else: self._json(HTTPStatus.NOT_FOUND,{"error":"Not found"})
        except Exception as e: logger.exception("Error"); self._json(HTTPStatus.INTERNAL_SERVER_ERROR,{"error":str(e)})
    def log_message(self, f, *a): logger.info(f, *a)
    def _body(self):
        n=int(self.headers.get("Content-Length","0"))
        return json.loads(self.rfile.read(n).decode()) if n>0 else {}
    def _json(self, s, p):
        b=json.dumps(p,default=str,separators=(",",":")).encode()
        self.send_response(s); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)

class WebSearchHTTPServer:
    def __init__(self, *, host, port, retrieval_service: RetrievalService):
        h = type("H",(_Handler,),{"svc":retrieval_service})
        self._server = ThreadingHTTPServer((host,port),h)
    def serve_forever(self): self._server.serve_forever(poll_interval=0.5)
    def stop(self): self._server.shutdown(); self._server.server_close()
