"""Download only the before-meal photos referenced by meal rows (resumable, parallel), then pack them.
Usage (from the repo root, on a machine that can reach PhysioNet):  python3 scripts/00_fetch_photos.py"""
import os, sys, tarfile, threading
from concurrent.futures import ThreadPoolExecutor
from remotezip import RemoteZip

URL = "https://physionet-open.s3.amazonaws.com/cgmacros/1.0.0/CGMacros_dateshifted365.zip"
OUT = "data/photos"
want = [l.strip() for l in open("splits/photo_list.txt") if l.strip()]
local = threading.local()

def zipobj():
    if not hasattr(local, "z"):
        local.z = RemoteZip(URL)
        local.names = {n.split("CGMacros/", 1)[-1]: n for n in local.z.namelist()}
    return local.z, local.names

def fetch(rel):
    dst = os.path.join(OUT, rel)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return "skip"
    z, names = zipobj()
    key = rel.split("CGMacros/", 1)[-1]
    if key not in names:
        return "missing"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    data = z.read(names[key])
    open(dst + ".part", "wb").write(data); os.replace(dst + ".part", dst)
    return "ok"

res = {"ok": 0, "skip": 0, "missing": 0}
with ThreadPoolExecutor(8) as ex:
    for i, r in enumerate(ex.map(fetch, want), 1):
        res[r] += 1
        if i % 50 == 0 or i == len(want):
            print(f"{i}/{len(want)} {res}", flush=True)
with tarfile.open("data/photos.tar", "w") as t:
    t.add(OUT, arcname="photos")
print("packed data/photos.tar", os.path.getsize("data/photos.tar") // 1_000_000, "MB")
