"""Finite-value preprocessing and feature-level multimodal fusion."""
import numpy as np

FEATURE_NAMES = ["radar_mean","radar_max","radar_spread","radar_core_fraction","cloud_mean","cloud_max","brightness_temp_mean","cloud_trend_proxy","flash_count","flash_density","flash_active_fraction","temperature_c","humidity_pct","wind_speed_mps","pressure_hpa","radar_available","satellite_available","lightning_available","atmosphere_available"]

def _num(v):
    try:
        n=float(v)
        return n if np.isfinite(n) else None
    except (TypeError,ValueError): return None

def _values(obj,key):
    try: a=np.asarray(obj.get(key),dtype=float).reshape(-1)
    except (TypeError,ValueError): return np.array([])
    return a[np.isfinite(a)]

def extract_features(payload):
    rad=_values(payload.get("radar",{}),"reflectivity_dbz")
    cloud=_values(payload.get("satellite",{}),"cloud_index")
    bt=_values(payload.get("satellite",{}),"brightness_temperature_k")
    light=payload.get("lightning",{}) or {}
    flash=_values(light,"flash_counts")
    atm=payload.get("atmosphere",{}) or {}
    avail={"radar":bool(rad.size),"satellite":bool(cloud.size or bt.size),"lightning":bool(flash.size or _num(light.get("recent_flash_count")) is not None),"atmosphere":any(_num(atm.get(k)) is not None for k in ("temperature_c","humidity_pct","wind_speed_mps","pressure_hpa"))}
    total=_num(light.get("recent_flash_count")); total=total if total is not None else (float(flash.sum()) if flash.size else 0.)
    f={"radar_mean":float(rad.mean()) if rad.size else 0.,"radar_max":float(rad.max()) if rad.size else 0.,"radar_spread":float(rad.std()) if rad.size else 0.,"radar_core_fraction":float((rad>=40).mean()) if rad.size else 0.,"cloud_mean":float(cloud.mean()) if cloud.size else 0.,"cloud_max":float(cloud.max()) if cloud.size else 0.,"brightness_temp_mean":float(bt.mean()) if bt.size else 280.,"cloud_trend_proxy":float(cloud.mean()-.5) if cloud.size else 0.,"flash_count":total,"flash_density":float(flash.mean()) if flash.size else total/576,"flash_active_fraction":float((flash>0).mean()) if flash.size else min(total/100,1.),"temperature_c":_num(atm.get("temperature_c")) or 29.,"humidity_pct":_num(atm.get("humidity_pct")) or 70.,"wind_speed_mps":_num(atm.get("wind_speed_mps")) or 6.,"pressure_hpa":_num(atm.get("pressure_hpa")) or 1008.}
    f.update({k+"_available":float(v) for k,v in avail.items()})
    return f,avail
