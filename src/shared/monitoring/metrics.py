"""Prometheus-compatible metrics for microservices."""
from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Generator


class Counter:
    def __init__(self, name, description, labels=None):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._value = 0.0
    def inc(self, amount=1.0):
        self._value += amount
    def get(self):
        return self._value
    def to_prometheus(self):
        lbl = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        lbl = "{" + lbl + "}" if lbl else ""
        nl = chr(10)
        return f"# HELP {self.name} {self.description}{nl}# TYPE {self.name} counter{nl}{self.name}{lbl} {self._value}"


class Histogram:
    def __init__(self, name, description, labels=None):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._buckets = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self._counts = {b: 0 for b in self._buckets}
        self._counts["inf"] = 0
        self._sum = 0.0
        self._count = 0
    def observe(self, value):
        self._sum += value
        self._count += 1
        for b in self._buckets:
            if value <= b:
                self._counts[b] += 1
        self._counts["inf"] += 1
    def to_prometheus(self):
        lbl = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        inner = "," + lbl if lbl else ""
        parts = [f"# HELP {self.name} {self.description}", f"# TYPE {self.name} histogram"]
        for b in self._buckets:
            parts.append(f'{self.name}_bucket{{{inner},le="{b}"}} {self._counts[b]}')
        parts.append(f'{self.name}_bucket{{{inner},le="+Inf"}} {self._counts["inf"]}')
        if lbl:
            parts.append(f'{self.name}_sum{{{lbl}}} {self._sum}')
            parts.append(f'{self.name}_count{{{lbl}}} {self._count}')
        else:
            parts.append(f'{self.name}_sum {self._sum}')
            parts.append(f'{self.name}_count {self._count}')
        nl = chr(10)
        return nl.join(parts)


class Gauge:
    def __init__(self, name, description, labels=None):
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._value = 0.0
    def set(self, v):
        self._value = v
    def inc(self, amount=1.0):
        self._value += amount
    def dec(self, amount=1.0):
        self._value -= amount
    def get(self):
        return self._value
    def to_prometheus(self):
        lbl = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        lbl = "{" + lbl + "}" if lbl else ""
        nl = chr(10)
        return f"# HELP {self.name} {self.description}{nl}# TYPE {self.name} gauge{nl}{self.name}{lbl} {self._value}"


class ServiceMetrics:
    def __init__(self, service_name):
        self.service_name = service_name
        L = {"service": service_name}
        self.request_counter = Counter(f"{service_name}_requests_total", "Total requests", L)
        self.request_errors = Counter(f"{service_name}_request_errors_total", "Total request errors", L)
        self.request_latency = Histogram(f"{service_name}_request_duration_seconds", "Request latency", L)
        self.documents_ingested = Counter(f"{service_name}_documents_ingested_total", "Documents ingested", L)
        self.validations_run = Counter(f"{service_name}_validations_total", "Validations performed", L)
        self.validation_fakes = Counter(f"{service_name}_validation_fakes_total", "Fake documents detected", L)
        self.model_inferences = Counter(f"{service_name}_model_inferences_total", "Model inference calls", L)
        self.searches = Counter(f"{service_name}_searches_total", "Search operations", L)
        self.active_connections = Gauge(f"{service_name}_active_connections", "Active connections", L)
        self.storage_used_bytes = Gauge(f"{service_name}_storage_used_bytes", "Storage used bytes", L)
    @contextmanager
    def track_request(self):
        self.request_counter.inc()
        start = time.monotonic()
        try:
            yield
        except Exception:
            self.request_errors.inc()
            raise
        finally:
            self.request_latency.observe(time.monotonic() - start)
    def to_prometheus(self):
        parts = []
        for m in [self.request_counter, self.request_errors, self.request_latency,
                  self.documents_ingested, self.validations_run, self.validation_fakes,
                  self.model_inferences, self.searches, self.active_connections, self.storage_used_bytes]:
            parts.append(m.to_prometheus())
        nl2 = chr(10) + chr(10)
        return nl2.join(parts)


_REGISTRY = {}


def get_metrics(service_name):
    if service_name not in _REGISTRY:
        _REGISTRY[service_name] = ServiceMetrics(service_name)
    return _REGISTRY[service_name]


def export_all_metrics():
    nl2 = chr(10) + chr(10)
    return nl2.join(m.to_prometheus() for m in _REGISTRY.values())
