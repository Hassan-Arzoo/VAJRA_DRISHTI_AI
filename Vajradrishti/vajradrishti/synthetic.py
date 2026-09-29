"""Deterministic synthetic storm data generator."""
from datetime import datetime, timedelta, timezone
import numpy as np

SOURCES = ("radar", "satellite", "lightning", "atmosphere")

def make_event(seed=42, step=0, grid_size=24, missing=()):
    rng = np.random.default_rng(seed + step * 17)
    y, x = np.mgrid[-1:1:complex(grid_size), -1:1:complex(grid_size)]
    cx, cy = -0.35 + 0.65 * min(step / 12, 1), 0.1 * np.sin(step / 3)
    radius = np.sqrt((x-cx)**2 + (y-cy)**2)
    core = np.exp(-(radius**2)/(0.13 + 0.01*step))
    radar = np.clip(10 + 55*core + rng.normal(0,3,core.shape), 0, 75).round(1)
    cloud = np.clip(0.2 + .72*core + rng.normal(0,.035,core.shape), 0, 1)
    flashes = rng.poisson(.3 + 7*core).astype(float)
    if "radar" in missing: radar[:] = np.nan
    if "satellite" in missing: cloud[:] = np.nan
    if "lightning" in missing: flashes[:] = np.nan
    bad = "atmosphere" in missing
    stamp = (datetime.now(timezone.utc)-timedelta(minutes=60-step*5)).isoformat()
    brightness=280-25*core+rng.normal(0,1,core.shape)
    if "satellite" in missing: brightness[:]=np.nan
    return {"event_id":"synthetic-cell-01", "timestamp":stamp,
      "radar":{"reflectivity_dbz":radar.tolist()},
      "satellite":{"cloud_index":cloud.tolist(),"brightness_temperature_k":brightness.tolist()},
      "lightning":{"flash_counts":flashes.tolist(),"recent_flash_count":None if "lightning" in missing else float(np.nansum(flashes))},
      "atmosphere":{"temperature_c":None if bad else 29-4*float(core.mean()),"humidity_pct":None if bad else 70+24*float(core.mean()),"wind_speed_mps":None if bad else 6+5*float(core.mean()),"pressure_hpa":None if bad else 1008-5*float(core.mean())}}

def demo_replay(seed=42, grid_size=24, missing=()):
    return [make_event(seed, i, grid_size, missing) for i in range(13)]
