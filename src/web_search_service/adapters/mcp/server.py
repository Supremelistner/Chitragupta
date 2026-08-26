# MCP Server
from __future__ import annotations
import json, logging, sys
from web_search_service.application.retrieval import RetrievalService
from web_search_service.domain.models import ContentFormat, CrawlRequest, FetchRequest, ScrapeRequest, SearchQuery

logger = logging.getLogger("web_search_service.mcp")

class MCPServer:
    def __init__(self, *, config, retrieval_service: RetrievalService):
        self._config = config
        self._retrieval = retrieval_service

    def serve(self):
        reader = sys.stdin.buffer
        writer = sys.stdout.buffer
        while True:
            msg = self._read(reader)
            if msg is None: return
            resp = self._dispatch(msg)
            if resp: self._write(writer, resp)

    def _dispatch(self, msg):
        m, rid = msg.get("method"), msg.get("id")
        try:
            if m == "initialize": return self._ok(rid, {"protocolVersion":"2024-11-05","serverInfo":{"name":"web-search-service","version":"0.1.0"},"capabilities":{"tools":{}}})
            if m == "tools/list": return self._ok(rid, {"tools": self._tools()})
            if m == "tools/call": return self._call(rid, msg.get("params",{}))
            if m == "ping": return self._ok(rid, {"pong": True})
            return self._err(rid, -32601, f"Not found: {m}") if rid else None
        except Exception as e:
            logger.exception("MCP error")
            return self._err(rid, -32603, str(e)) if rid else None

    def _call(self, rid, p):
        n, a = p.get("name"), p.get("arguments") or {}
        if n == "web_search":
            q = a.get("query","")
            if not q: raise ValueError("query required")
            r = self._retrieval.search(SearchQuery(query=q, limit=int(a.get("limit",10))))
            return self._ok(rid, {"query":r.query,"provider":r.provider.value,"latency_ms":r.latency_ms,"results":[{"url":x.url,"title":x.title,"snippet":x.snippet} for x in r.results]})
        if n == "web_scrape":
            u = a.get("url","")
            if not u: raise ValueError("url required")
            r = self._retrieval.scrape(ScrapeRequest(url=u, format=ContentFormat(a.get("format","markdown"))))
            return self._ok(rid, {"url":r.url,"title":r.title,"content":r.content[:5000],"provider":r.provider.value,"latency_ms":r.latency_ms})
        if n == "web_crawl":
            u = a.get("url","")
            if not u: raise ValueError("url required")
            r = self._retrieval.crawl(CrawlRequest(url=u, limit=int(a.get("limit",10)), depth=int(a.get("depth",2))))
            return self._ok(rid, {"url":r.url,"provider":r.provider.value,"page_count":len(r.pages),"pages":[{"url":p.url,"title":p.title,"preview":p.content[:500]} for p in r.pages]})
        if n == "fetch_url":
            u = a.get("url","")
            if not u: raise ValueError("url required")
            r = self._retrieval.fetch(FetchRequest(url=u, format=ContentFormat(a.get("format","markdown"))))
            return self._ok(rid, {"url":r.url,"content":r.content[:5000],"status":r.status_code,"content_type":r.content_type,"latency_ms":r.latency_ms})
        if n == "health": return self._ok(rid, self._retrieval.health())
        return self._err(rid, -32602, f"Unknown: {n}")

    def _tools(self):
        return [
            {"name":"web_search","description":"Search the web.","inputSchema":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"}},"required":["query"]}},
            {"name":"web_scrape","description":"Scrape URL to markdown.","inputSchema":{"type":"object","properties":{"url":{"type":"string"},"format":{"type":"string"}},"required":["url"]}},
            {"name":"web_crawl","description":"Crawl a website.","inputSchema":{"type":"object","properties":{"url":{"type":"string"},"limit":{"type":"integer"},"depth":{"type":"integer"}},"required":["url"]}},
            {"name":"fetch_url","description":"Fetch URL content.","inputSchema":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}},
            {"name":"health","description":"Health check.","inputSchema":{"type":"object","properties":{}}},
        ]

    def _ok(self, rid, payload): return {"jsonrpc":"2.0","id":rid,"result":{"content":[{"type":"text","text":json.dumps(payload,default=str,separators=(",",":"))}],"isError":False}}
    def _err(self, rid, code, msg): return {"jsonrpc":"2.0","id":rid,"error":{"code":code,"message":msg}}
    def _read(self, r):
        h = {}
        while True:
            l = r.readline()
            if not l: return None
            l = l.decode().rstrip("
")
            if not l: break
            if ":" in l:
                k,v = l.split(":",1)
                h[k.strip().lower()] = v.strip()
        cl = int(h.get("content-length","0"))
        return json.loads(r.read(cl).decode()) if cl > 0 else None
    def _write(self, w, msg):
        p = json.dumps(msg,separators=(",",":")).encode()
        w.write(f"Content-Length: {len(p)}

".encode())
        w.write(p)
        w.flush()
