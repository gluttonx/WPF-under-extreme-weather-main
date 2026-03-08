#!/usr/bin/env python3
import os
import re
import subprocess
import sys
import threading
from pathlib import Path


HEADER_RE = re.compile(br"Content-Length:\s*(\d+)", re.IGNORECASE)


def log(message):
    log_path = None
    if "MCP_BRIDGE_LOG" in os.environ:
        log_path = Path(os.environ["MCP_BRIDGE_LOG"])
    if not log_path:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def read_content_length_message(buffer, source):
    while True:
        header_end = buffer.find(b"\r\n\r\n")
        delimiter_size = 4
        if header_end == -1:
            header_end = buffer.find(b"\n\n")
            delimiter_size = 2
        if header_end == -1:
            chunk = source.read(4096)
            if not chunk:
                return None, buffer
            buffer += chunk
            continue

        headers = buffer[:header_end]
        match = HEADER_RE.search(headers)
        if not match:
            raise ValueError(f"Missing Content-Length header: {headers!r}")

        content_length = int(match.group(1))
        message_start = header_end + delimiter_size
        while len(buffer) < message_start + content_length:
            chunk = source.read(4096)
            if not chunk:
                return None, buffer
            buffer += chunk

        payload = buffer[message_start:message_start + content_length]
        buffer = buffer[message_start + content_length:]
        return payload, buffer


def read_line_message(buffer, source):
    while True:
        newline_index = buffer.find(b"\n")
        if newline_index != -1:
            payload = buffer[:newline_index].rstrip(b"\r")
            buffer = buffer[newline_index + 1:]
            if payload:
                return payload, buffer
            continue

        chunk = source.read(4096)
        if not chunk:
            return None, buffer
        buffer += chunk


def client_to_server(source, target):
    buffer = b""
    mode = None
    try:
        while True:
            if mode is None:
                while not buffer:
                    chunk = source.read(4096)
                    if not chunk:
                        target.close()
                        return
                    buffer += chunk
                mode = "content-length" if buffer.startswith(b"Content-Length:") else "line"
                log(f"client_mode={mode}")

            if mode == "content-length":
                payload, buffer = read_content_length_message(buffer, source)
            else:
                payload, buffer = read_line_message(buffer, source)

            if payload is None:
                target.close()
                return

            target.write(payload + b"\n")
            target.flush()
    except BrokenPipeError:
        return


def server_to_client(source, target):
    buffer = b""
    try:
        while True:
            payload, buffer = read_line_message(buffer, source)
            if payload is None:
                return

            frame = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii") + payload
            target.write(frame)
            target.flush()
    except BrokenPipeError:
        return


def pipe_stderr(source, target):
    while True:
        chunk = source.read(4096)
        if not chunk:
            return
        target.write(chunk)
        target.flush()


def main():
    if len(sys.argv) < 2:
        print("Usage: mcp_stdio_bridge.py <command> [args...]", file=sys.stderr)
        return 2

    process = subprocess.Popen(
        sys.argv[1:],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    log(f"spawned={' '.join(sys.argv[1:])}")

    threads = [
        threading.Thread(
            target=client_to_server,
            args=(sys.stdin.buffer, process.stdin),
            daemon=True,
        ),
        threading.Thread(
            target=server_to_client,
            args=(process.stdout, sys.stdout.buffer),
            daemon=True,
        ),
        threading.Thread(
            target=pipe_stderr,
            args=(process.stderr, sys.stderr.buffer),
            daemon=True,
        ),
    ]

    for thread in threads:
        thread.start()

    return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
