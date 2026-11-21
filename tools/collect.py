#!/usr/bin/env python3
"""Talk to the ESP32 over USB serial: preview framing, collect training crops,
and measure live accuracy.

  pip install pyserial
  python tools/collect.py preview                    # live 28x28 view in the terminal
  python tools/collect.py collect --signer alice     # 26 letters x 30 crops -> data/crops.csv
  python tools/collect.py collect --signer bob --letters JZ --n 40
  python tools/collect.py sheet                      # data/sheet.pgm contact sheet (open in Preview)
  python tools/collect.py quiz --n 26                # random letters, live accuracy on a new hand

Crops are exactly what the model sees (the firmware preprocesses them), so
training on them removes the camera/dataset mismatch.
"""
import argparse
import collections
import csv
import glob
import pathlib
import random
import string
import sys
import time

try:
    import serial  # pyserial
except ImportError:
    serial = None

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CSV = DATA / "crops.csv"
ALPHA = string.ascii_uppercase
SHADES = " .:-=+*#%@"


def find_port():
    pats = ["/dev/cu.usbserial*", "/dev/cu.wchusbserial*", "/dev/cu.SLAB*",
            "/dev/cu.usbmodem*", "/dev/ttyUSB*", "/dev/ttyACM*"]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    raise SystemExit("no serial port found; pass --port (Windows: COM5 etc.)")


def open_port(port, baud):
    if serial is None:
        raise SystemExit("pip install pyserial")
    s = serial.Serial()
    s.port = port or find_port()
    s.baudrate = baud
    s.timeout = 1
    # ESP32-CAM-MB wires RTS->EN and DTR->IO0: keep both released or the
    # board sits in reset / bootloader.
    s.dtr = False
    s.rts = False
    s.open()
    print(f"[serial] {s.port} - waiting for boot...")
    time.sleep(2.5)
    s.reset_input_buffer()
    return s


def cmd(s, line):
    s.write((line + "\n").encode())


def read_lines(s, until, timeout=15):
    t = time.time()
    while time.time() - t < timeout:
        raw = s.readline()
        if not raw:
            continue
        line = raw.decode(errors="replace").strip()
        yield line
        if line.startswith(until):
            return


def snap(s, n):
    cmd(s, f"SNAP {n}")
    imgs = []
    for line in read_lines(s, "END", timeout=5 + n * 0.5):
        if line.startswith("IMG ") and len(line) == 4 + 784 * 2:
            imgs.append(bytes.fromhex(line[4:]))
        elif line.startswith("ERR"):
            print("[device]", line)
    return imgs


def ascii_img(px):
    rows = []
    for y in range(28):
        rows.append("".join(SHADES[px[y * 28 + x] * len(SHADES) // 256] * 2 for x in range(28)))
    return "\n".join(rows)


# ------------------------------------------------------------------ commands
def do_preview(a):
    s = open_port(a.port, a.baud)
    cmd(s, "LIVE 0")
    print("Ctrl-C to stop. Aim: hand fills the square, plain backdrop, even light.")
    try:
        while True:
            imgs = snap(s, 1)
            if imgs:
                sys.stdout.write("\x1b[H\x1b[2J" + ascii_img(imgs[0]) + "\n")
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass


def do_collect(a):
    s = open_port(a.port, a.baud)
    DATA.mkdir(exist_ok=True)
    new = not CSV.exists()
    letters = a.letters.upper()
    print(f"Signer '{a.signer}': {len(letters)} letters x {a.n} crops.")
    print("For each letter: hold the sign in the frame and press Enter.")
    print("While it records, move a little: tilt, shift, closer/further. Variety = accuracy.")
    print("J and Z: hold the hand shape from the MIDDLE of the motion.\n")
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["signer", "label", "letter"] + [f"pixel{i}" for i in range(784)])
        for i, L in enumerate(letters):
            ans = input(f"[{i + 1}/{len(letters)}] sign '{L}'  (Enter=record, s=skip, q=quit) ").strip().lower()
            if ans == "q":
                break
            if ans == "s":
                continue
            for c in "321":
                print(c, end=" ", flush=True)
                time.sleep(0.4)
            print("recording...", end=" ", flush=True)
            imgs = snap(s, a.n)
            for px in imgs:
                w.writerow([a.signer, ALPHA.index(L), L] + list(px))
            f.flush()
            print(f"got {len(imgs)}")
            if imgs:
                print(ascii_img(imgs[len(imgs) // 2]))
    print(f"saved to {CSV}")


def load_rows():
    with open(CSV, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            yield row[0], row[2], bytes(map(int, row[3:]))


def do_sheet(a):
    rows = list(load_rows())
    by = collections.defaultdict(list)
    for signer, L, px in rows:
        by[(L, signer)].append(px)
    keys = sorted(by)
    cols = a.per
    H, W = len(keys) * 29, cols * 29
    img = bytearray([255]) * (H * W)
    for r, k in enumerate(keys):
        for c, px in enumerate(by[k][:cols]):
            for y in range(28):
                o = (r * 29 + y) * W + c * 29
                img[o:o + 28] = px[y * 28:(y + 1) * 28]
    out = DATA / "sheet.pgm"
    out.write_bytes(f"P5 {W} {H} 255\n".encode() + bytes(img))
    print(f"wrote {out} (rows: {', '.join(f'{l}/{s}' for l, s in keys[:6])} ...)")
    print("crops per signer:", dict(collections.Counter(s for s, _, _ in rows)))


def do_quiz(a):
    s = open_port(a.port, a.baud)
    cmd(s, "INFO")
    labels = ALPHA
    for line in read_lines(s, "INFO", timeout=5):
        if line.startswith("INFO labels="):
            labels = line.split()[1].split("=")[1]
    order = [random.choice(labels) for _ in range(a.n)]
    hits, wrong = 0, []
    for i, L in enumerate(order):
        input(f"[{i + 1}/{a.n}] sign '{L}' and press Enter ")
        cmd(s, "PRED")
        got = "?"
        for line in read_lines(s, "PRED", timeout=10):
            if line.startswith("PRED "):
                got = line.split()[1]
                print("   ", line)
        if got == L:
            hits += 1
        else:
            wrong.append(f"{L}->{got}")
    print(f"\naccuracy: {hits}/{a.n} = {100 * hits / a.n:.0f}%")
    if wrong:
        print("misses:", " ".join(wrong))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=115200)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preview")
    c = sub.add_parser("collect")
    c.add_argument("--signer", required=True)
    c.add_argument("--letters", default=ALPHA)
    c.add_argument("--n", type=int, default=30)
    sh = sub.add_parser("sheet")
    sh.add_argument("--per", type=int, default=15)
    q = sub.add_parser("quiz")
    q.add_argument("--n", type=int, default=26)
    a = ap.parse_args()
    {"preview": do_preview, "collect": do_collect, "sheet": do_sheet, "quiz": do_quiz}[a.cmd](a)


if __name__ == "__main__":
    main()
