"""Reduce the BIG IDEAs glycemic-wearable files (PhysioNet v1.1.2) to the three small tables
`glimpse external-bigideas` reads. Run inside data/bigideas/ after the download lines in the README:
    python ../../scripts/prepare_bigideas.py
Person 003's food log has no header and 11 columns (no time_end, fibre or fat); its columns are named here."""

import glob

import pandas as pd

cg, hr, fd = [], [], []
for d in sorted(glob.glob("0??")):
    i = int(d)
    x = pd.read_csv(f"{d}/Dexcom_{d}.csv", low_memory=False)
    x = x[x["Event Type"] == "EGV"]
    cg.append(pd.DataFrame({"id": i, "t": pd.to_datetime(x.iloc[:, 1], errors="coerce"),
                            "glucose": pd.to_numeric(x["Glucose Value (mg/dL)"], errors="coerce")}).dropna())  # fmt: skip
    h = pd.read_csv(f"{d}/HR_{d}.csv", encoding="utf-8-sig")
    h.columns = [k.strip() for k in h.columns]
    h["t"] = pd.to_datetime(h["datetime"], errors="coerce", format="mixed")
    h["hr"] = pd.to_numeric(h["hr"], errors="coerce")
    h = h.dropna(subset=["t", "hr"]).set_index("t")["hr"].resample("5min").mean().dropna().reset_index()
    h.insert(0, "id", i)
    hr.append(h)
    f = pd.read_csv(f"{d}/Food_Log_{d}.csv", encoding="utf-8-sig", on_bad_lines="skip")
    f.columns = [k.strip() for k in f.columns]
    if "time_begin" not in f.columns:
        f = pd.read_csv(f"{d}/Food_Log_{d}.csv", encoding="utf-8-sig", on_bad_lines="skip", header=None)
        f = pd.DataFrame({"time_begin": f[2], "logged_food": f[3], "calorie": pd.to_numeric(f[7], errors="coerce"),
                          "total_carb": pd.to_numeric(f[8], errors="coerce")})  # fmt: skip
    f.insert(0, "id", i)
    fd.append(f)
pd.concat(cg).to_csv("bigideas_cgm.csv", index=False)
pd.concat(hr).to_csv("bigideas_hr5.csv", index=False)
pd.concat(fd).sort_values(["id", "time_begin"]).to_csv("bigideas_food.csv", index=False)
