from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

MPS_TO_KNOTS = 1.9438444924574


def _latlon_at(v, k: int) -> tuple[float, float]:
    ll = np.asarray(v.latlon, float)  # (2,N)
    a, b = float(ll[0, k]), float(ll[1, k])
    # lat i [-90,90]
    return (a, b) if (-90 <= a <= 90) else (b, a)


def _cog_deg_0_360_from_sim_cog_rad(cog_rad: float) -> float:
    # Simulator-cog er rad, NAV-konvensjon (0=nord, +/-90 = øst/vest)
    # Evaluator forventer cog i grader (AIS), og den konverterer selv til rad.
    return float(np.degrees(cog_rad) % 360.0)


def export_episode_to_evaluator_csv(
    ep0: dict,
    out_csv: Path,
    start_dt: datetime | None = None,
    nav_status_default: int = 0,
) -> Path:
    """
    Lager ; -separert CSV som matcher colav_evaluation_tool VesselData.create_from_ais_data.

    Påkrevde kolonner:
      mmsi;date_time_utc;sog;cog;true_heading;nav_status;calc_speed;lon;lat

    Enheter:
      sog = knots
      cog/true_heading = degrees

    Viktig:
      - 'date_time_utc' skrives uten 'Z' og med mellomrom (pandas-vennlig).
    """
    if start_dt is None:
        start_dt = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    rows: list[dict] = []

    for v in ep0["vessel_data"]:
        ts = np.asarray(v.timestamps, float)  # sek
        sog_mps = np.asarray(v.sog, float)    # m/s
        cog_rad = np.asarray(v.cog, float)    # rad
        mmsi = int(getattr(v, "mmsi", 0) or 0)

        # Bygg alle rader i tidsrekkefølge (k)
        for k, t in enumerate(ts):
            lat, lon = _latlon_at(v, k)

            # Lag "naiv" datetime (uten tz) men i samme tidsskala
            dt = (start_dt + timedelta(seconds=float(t))).replace(tzinfo=None)

            sog_kn = float(sog_mps[k] * MPS_TO_KNOTS)
            cog_deg = _cog_deg_0_360_from_sim_cog_rad(float(cog_rad[k]))

            rows.append(
                {
                    "mmsi": mmsi,
                    "date_time_utc": dt,                 # <- datetime objekt i DF
                    "sog": sog_kn,
                    "cog": cog_deg,
                    "true_heading": cog_deg,             # best effort
                    "nav_status": nav_status_default,    # må være ikke-NaN
                    "calc_speed": sog_kn,                # best effort
                    "lon": float(lon),
                    "lat": float(lat),
                    "_k": k,                              # <- kun for stabil sortering
                }
            )

    df = pd.DataFrame(rows)

    # Stabil sort: først mmsi, så sample-indeks
    df = df.sort_values(["mmsi", "_k"]).drop(columns=["_k"])

    # Skriv til tekstformatet evaluator/pandas parse_dates liker:
    # "YYYY-mm-dd HH:MM:SS.ffffff"
    df["date_time_utc"] = df["date_time_utc"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, sep=";", index=False)
    print(f"Wrote evaluator CSV: {out_csv} (rows={len(df)})")
    return out_csv
