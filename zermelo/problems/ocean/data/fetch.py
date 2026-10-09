"""Fetch one month of the Gulf Stream box into `gulf_stream_<yyyy><mon>.npz`: `python -m zermelo.problems.ocean.data.fetch 2019 06`

HYCOM GOFS 3.1 surface current (1/12 degree, 3-hourly) through its NetCDF subset service, and ERA5 10 m wind (1/4 degree)
through the Copernicus climate data store, interpolated onto the HYCOM grid points and frames.

keys: latitude (n_lat), longitude (n_lon), hours (frames, since the month's first 00Z),
      u, v, wind_u, wind_v (frames, n_lat, n_lon) in m/s, land (n_lat, n_lon); land carries zero current and zero wind
"""

import calendar
import subprocess
import sys
import tempfile
from pathlib import Path

import cdsapi
import numpy as np
import xarray as xr

BOX = dict(north=42, south=35, west=-75, east=-55)
"""The Gulf Stream after Cape Hatteras, degrees"""

HYCOM = "https://ncss.hycom.org/thredds/ncss/GLBy0.08/expt_93.0/uv3z?var=water_u&var=water_v&vertCoord=0&accept=netcdf4"


def hycom_day(day: str, into: Path) -> Path:
    """One day of surface current, 3-hourly, fetched until the file opens"""
    out = into / f"hycom_{day}.nc"
    box = "&".join(f"{side}={degrees}" for side, degrees in BOX.items())
    for _ in range(4):
        subprocess.run(
            ["curl", "-s", "-m", "600", "-o", str(out), f"{HYCOM}&{box}&time_start={day}T00:00:00Z&time_end={day}T21:00:00Z"], check=False
        )
        try:
            xr.open_dataset(out).close()
            return out
        except (OSError, ValueError):
            continue
    raise RuntimeError(f"HYCOM returned no readable file for {day}")


def era5_month(year: int, month: int, into: Path) -> Path:
    """The month's 10 m wind at every third hour, from the climate data store"""
    out = into / "era5.nc"
    cdsapi.Client().retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)],
            "time": [f"{h:02d}:00" for h in range(0, 24, 3)],
            "area": [BOX["north"], BOX["west"], BOX["south"], BOX["east"]],
        },
        str(out),
    )
    return out


def assemble(days: list[Path], era5: Path, start: np.datetime64, out: Path) -> None:
    """The daily current files and the wind file into one npz at `out`"""
    current = xr.concat([xr.open_dataset(day) for day in days], dim="time").isel(depth=0).load()
    current = current.assign_coords(lon=((current.lon + 180) % 360) - 180)
    wind = xr.open_dataset(era5)
    wind = wind.rename({"valid_time": "time"} if "valid_time" in wind.dims else {}).sortby("latitude")
    wind = wind.interp(latitude=current.lat, longitude=current.lon, time=current.time)
    u, v = current.water_u.values.astype(np.float32), current.water_v.values.astype(np.float32)
    wind_u, wind_v = wind.u10.values.astype(np.float32), wind.v10.values.astype(np.float32)
    land = np.isnan(u).all(axis=0)
    for field in (u, v, wind_u, wind_v):
        field[np.isnan(field)] = 0.0
    np.savez(
        out,
        latitude=current.lat.values.astype(np.float32),
        longitude=current.lon.values.astype(np.float32),
        hours=((current.time.values - start) / np.timedelta64(1, "h")).astype(np.float32),
        u=u,
        v=v,
        wind_u=wind_u,
        wind_v=wind_v,
        land=land,
    )


if __name__ == "__main__":
    year, month = int(sys.argv[1]), int(sys.argv[2])
    with tempfile.TemporaryDirectory() as scratch:
        into = Path(scratch)
        days = [hycom_day(f"{year}-{month:02d}-{d:02d}", into) for d in range(1, calendar.monthrange(year, month)[1] + 1)]
        era5 = era5_month(year, month, into)
        target = Path(__file__).parent / f"gulf_stream_{year}{calendar.month_abbr[month].lower()}.npz"
        assemble(days, era5, np.datetime64(f"{year}-{month:02d}-01T00:00:00"), target)
        print(f"wrote {target}")
