"""
RequestLogger middleware
Logs every request + response status to Flask's logger.
"""

import time
from flask import Flask, request, g


class RequestLogger:
    def __init__(self, app: Flask):
        app.before_request(self._before)
        app.after_request(self._after)

    @staticmethod
    def _before():
        g._start_time = time.perf_counter()

    @staticmethod
    def _after(response):
        elapsed_ms = (time.perf_counter() - g._start_time) * 1000
        import flask
        flask.current_app.logger.info(
            "%s %s → %d  (%.1f ms)",
            request.method,
            request.path,
            response.status_code,
            elapsed_ms,
        )
        return response
