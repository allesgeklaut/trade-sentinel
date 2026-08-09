#!/usr/bin/env python3
"""Generate a 180×180 pixel-art PNG icon for iOS homescreen / PWA.

Uses only the Python standard library (struct + zlib) so it can run
during the Docker build with no extra dependencies.

Design: two candlesticks (green bullish + red bearish) on a dark
background — the Trade Sentinel signature look.
"""

import struct
import zlib
from pathlib import Path

W = H = 180

# Palette — matches the app's terminal-green theme
BG     = (0x05, 0x0a, 0x07)   # #050a07  dark background
GREEN  = (0x42, 0xff, 0x72)   # #42ff72  bullish candle body
RED    = (0xff, 0x5c, 0x70)   # #ff5c70  bearish candle body
GWICK  = (0x2d, 0xbb, 0x54)   # #2dbb54  green wick (darker)
RWICK  = (0xcc, 0x47, 0x59)   # #cc4759  red wick (darker)

# Create pixel buffer
pixels = [[BG] * W for _ in range(H)]


def fill(x0: int, y0: int, x1: int, y1: int, color: tuple) -> None:
    """Fill a rectangle (x0,y0)-(x1,y1) exclusive with the given RGB color."""
    for y in range(max(0, y0), min(H, y1)):
        for x in range(max(0, x0), min(W, x1)):
            pixels[y][x] = color


# --- Green candlestick (left, taller — bullish) ---
fill(54, 20, 58, 155, GWICK)     # wick: 4px wide, y 20-154
fill(38, 44, 74, 126, GREEN)     # body: 36px wide, y 44-125

# --- Red candlestick (right, shorter — bearish) ---
fill(124, 38, 128, 165, RWICK)   # wick: 4px wide, y 38-164
fill(108, 64, 144, 141, RED)     # body: 36px wide, y 64-140

# --- Build raw scanlines (filter byte 0 + RGB triplets) ---
raw = bytearray()
for y in range(H):
    raw.append(0)               # filter: None
    for x in range(W):
        r, g, b = pixels[y][x]
        raw += struct.pack("BBB", r, g, b)

compressed = zlib.compress(bytes(raw), 9)


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


png = b"\x89PNG\r\n\x1a\n"
png += _chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))  # 8-bit RGB
png += _chunk(b"IDAT", compressed)
png += _chunk(b"IEND", b"")

out = Path(__file__).resolve().parent / "static" / "icon-180.png"
out.write_bytes(png)
print(f"Generated {out} ({len(png)} bytes)")