"""Download and parse Department of Census and Statistics (DCS) paddy tables.

The DCS site serves its tables as binary files (PDF or Excel) behind
"View" links. This module:
  1. downloads each district series (yield, production, sown, harvested;
     Maha and Yala, 1979-2025) into data/raw/dcs/,
  2. detects the file type from its first bytes,
  3. parses the wide table (districts x seasons, either orientation) into a
     tidy long table: district, season, season_year, variable, value.

Run from the project root:
    python -m src.dcs download       # fetch the 8 district series
    python -m src.dcs parse          # -> data/interim/paddy_district_long.csv
    python -m src.dcs seasons        # optional: Major/Minor/Rainfed split 2004+
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config as C

HEADERS = {"User-Agent": "Mozilla/5.0 (MSc AppliedML coursework; paddy-yield-sl)"}
YEAR_RE = re.compile(r"(19[5-9]\d|20[0-4]\d)\s*(?:[/\-]\s*(\d{2,4}))?")


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def _sniff(content: bytes) -> str:
    head = content[:8]
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK"):
        return "xlsx"
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        return "xls"
    txt = content[:2000].lower()
    if b"<html" in txt or b"<!doctype" in txt:
        return "html"
    return "bin"


def _get(url: str, retries: int = 3) -> requests.Response:
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r
        except requests.RequestException as e:  # noqa: PERF203
            if i == retries - 1:
                raise
            print(f"  retry {i + 1} after error: {e}")
            time.sleep(3)
    raise RuntimeError("unreachable")


def download_file(url: str, stem: str, out_dir: Path = C.RAW_DCS, force: bool = False) -> Path:
    """Download url, follow an embedded PDF/XLSX link if an HTML page comes back."""
    existing = list(out_dir.glob(stem + ".*"))
    if existing and not force:
        print(f"  cached  {existing[0].name}")
        return existing[0]
    r = _get(url)
    kind = _sniff(r.content)
    if kind == "html":
        m = re.search(r'(?:href|src|data)=["\']([^"\']+\.(?:pdf|xlsx?|PDF|XLSX?))["\']', r.text)
        if not m:
            path = out_dir / f"{stem}.html"
            path.write_bytes(r.content)
            print(f"  WARNING {stem}: got an HTML page without a file link -> saved {path.name}")
            return path
        link = requests.compat.urljoin(url, m.group(1))
        r = _get(link)
        kind = _sniff(r.content)
    path = out_dir / f"{stem}.{kind}"
    path.write_bytes(r.content)
    print(f"  saved   {path.name}  ({len(r.content) / 1024:.0f} KB)")
    return path


def download_series(force: bool = False) -> None:
    print("Downloading DCS district series ...")
    for key, (slug, _, _) in C.DCS_SERIES.items():
        download_file(f"{C.DCS_BASE}/{slug}", key, force=force)
        time.sleep(1)


# --------------------------------------------------------------------------
# Reading raw files into a list of 2-D string grids (one per sheet/page)
# --------------------------------------------------------------------------
def _grids_from_file(path: Path) -> tuple[list[list[list[str]]], str]:
    """Return (grids, full_text). Each grid is a list of rows of cell strings."""
    suffix = path.suffix.lower()
    grids: list[list[list[str]]] = []
    text = ""
    if suffix in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
        for df in sheets.values():
            grid = df.fillna("").astype(str).values.tolist()
            grids.append(grid)
            text += "\n".join(" ".join(r) for r in grid)
    elif suffix == ".pdf":
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
                tables = page.extract_tables()
                if tables:
                    for t in tables:
                        grids.append([[c or "" for c in row] for row in t])
                # Text fallback: split each line into tokens (district names may
                # contain a space, so join leading alphabetic tokens).
                rows = []
                for line in page_text.splitlines():
                    toks = line.split()
                    lead = []
                    while toks and re.match(r"^[A-Za-z][A-Za-z'.\-]*$", toks[0]):
                        lead.append(toks.pop(0))
                    rows.append([" ".join(lead)] + toks if lead else toks)
                grids.append(rows)
    else:
        raise ValueError(f"Unsupported file type: {path.name}. Open it and check what it is.")
    return grids, text


def _canon_district(cell: str) -> str | None:
    key = re.sub(r"[^a-z']", "", str(cell).lower())
    if not key:
        return None
    if key in C.DISTRICT_ALIASES:
        return C.DISTRICT_ALIASES[key]
    for alias, name in C.DISTRICT_ALIASES.items():
        if len(alias) >= 5 and key.startswith(alias):
            return name
    return None


def _season_year(cell: str) -> int | None:
    m = YEAR_RE.search(str(cell))
    return int(m.group(1)) if m else None


def _num(cell: str) -> float:
    s = str(cell).strip().replace(",", "")
    if s in ("", "-", "--", "…", "...", "n.a", "na", "N.A", "nan", "None"):
        return np.nan
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_grid(grid: list[list[str]]) -> list[tuple[str, int, float]]:
    """Return (district, season_year, value) triples from one grid."""
    out: list[tuple[str, int, float]] = []
    if not grid:
        return out
    # Some sheets carry a stray, fully-blank leading column (e.g. an unlabelled
    # row-number spacer); drop any column that is blank in every row, or the
    # whole grid would be shifted one column off from the district/year labels.
    width = max(len(r) for r in grid)
    grid = [list(r) + [""] * (width - len(r)) for r in grid]
    blank_cols = {j for j in range(width) if all(str(r[j]).strip() == "" for r in grid)}
    if blank_cols:
        keep = [j for j in range(width) if j not in blank_cols]
        grid = [[r[j] for j in keep] for r in grid]
    # Orientation A: header rows with >=3 year tokens; districts down column 0.
    # A grid (PDF page) can hold several stacked blocks, each with its own header.
    cols, header = None, None
    for i, row in enumerate(grid):
        ycols = {j: _season_year(c) for j, c in enumerate(row) if _season_year(c)}
        nonblank = sum(1 for c in row if str(c).strip() not in ("", "-"))
        # A real header row is ALMOST ALL year labels; a data row for a district
        # the alias table doesn't recognise (a mis-spelling, an irrigation
        # scheme, a "SRI LANKA" total row) is mostly numbers that only look
        # like years by coincidence - require a high year-match ratio too, or
        # a handful of wide rows like that corrupt every row parsed after them.
        looks_like_header = nonblank > 0 and len(ycols) / nonblank >= 0.6
        if len(ycols) >= 3 and looks_like_header and _canon_district(row[0] if row else "") is None:
            cols, header = dict(ycols), row
            # Some tables break each year into Major / Minor / Rainfed / Total
            # sub-columns (next row); keep only the "Total" sub-column per year.
            nxt = grid[i + 1] if i + 1 < len(grid) else []
            labels = [str(c).strip().lower() for c in nxt]
            if "total" in labels:
                starts = sorted(cols)
                new_cols = {}
                for k, j0 in enumerate(starts):
                    j_end = starts[k + 1] if k + 1 < len(starts) else len(labels)
                    tot = next((j for j in range(j0, min(j_end, len(labels)))
                                if labels[j] == "total"), None)
                    if tot is not None:
                        new_cols[tot] = cols[j0]
                if new_cols:
                    cols, header = new_cols, nxt
            continue
        if cols is None or not row:
            continue
        d = _canon_district(row[0])
        if d is None:
            continue
        ys = [cols[j] for j in sorted(cols)]
        if len(row) == len(header):
            for j, y in cols.items():
                out.append((d, y, _num(row[j])))
        else:  # text-fallback row: align values from the right
            vals = [_num(c) for c in row[1:]]
            if len(vals) >= len(ys):
                for y, v in zip(ys, vals[-len(ys):]):
                    out.append((d, y, v))
    if out:
        return out
    # Orientation B: a header row with >=5 district names; years down column 0
    for hi, row in enumerate(grid):
        dcols = {j: _canon_district(c) for j, c in enumerate(row) if _canon_district(c)}
        if len(dcols) >= 5:
            for r in grid[hi + 1:]:
                if not r:
                    continue
                y = _season_year(r[0])
                if y is None:
                    continue
                for j, d in dcols.items():
                    if j < len(r):
                        out.append((d, y, _num(r[j])))
            return out
    return out


def parse_series_file(path: Path) -> tuple[pd.DataFrame, str]:
    grids, text = _grids_from_file(path)
    triples: list[tuple[str, int, float]] = []
    # Parse each sheet/page, then all pages stacked (tables that continue onto
    # the next page without repeating the header row).
    for g in grids + [[row for g in grids for row in g]]:
        triples.extend(_parse_grid(g))
    df = pd.DataFrame(triples, columns=["district", "season_year", "value"])
    df = (df.dropna(subset=["value"])
            .drop_duplicates(["district", "season_year"], keep="first"))
    return df, text


def _detect_unit(variable: str, text: str, values: pd.Series) -> str:
    t = text.lower()
    if variable == "yield":
        if "bushel" in t or "bu." in t:
            return "bu_per_acre"
        if "kg" in t:
            return "kg_per_ha"
        return "bu_per_acre" if values.median() < 300 else "kg_per_ha"
    if variable in ("sown", "harvested"):
        if "acre" in t and "hectare" not in t:
            return "acres"
        return "ha"
    if variable == "production":
        if "bushel" in t:
            return "bushels"
        if "'000" in t or "000 mt" in t:
            return "thousand_mt"
        return "mt"
    return "unknown"


def _to_metric(variable: str, unit: str, s: pd.Series) -> pd.Series:
    if unit == "bu_per_acre":
        return s * C.BU_PER_AC_TO_KG_PER_HA
    if unit == "acres":
        return s * C.ACRE_HA
    if unit == "bushels":
        return s * C.BUSHEL_PADDY_KG / 1000.0
    if unit == "thousand_mt":
        return s * 1000.0
    return s


def parse_all() -> pd.DataFrame:
    frames, report = [], []
    for key, (_, variable, season) in C.DCS_SERIES.items():
        files = [p for p in C.RAW_DCS.glob(key + ".*")]
        if not files:
            report.append(f"{key:18s} MISSING - run `python -m src.dcs download`")
            continue
        df, text = parse_series_file(files[0])
        if df.empty:
            report.append(f"{key:18s} PARSE FAILED ({files[0].name}). See diagnostics below.")
            _diagnose(files[0])
            continue
        unit = _detect_unit(variable, text, df["value"])
        df["value"] = _to_metric(variable, unit, df["value"])
        df["variable"], df["season"] = variable, season
        frames.append(df)
        report.append(f"{key:18s} {len(df):5d} values, {df.district.nunique():2d} districts, "
                      f"{df.season_year.min()}-{df.season_year.max()}, unit detected: {unit}")
    print("\n".join(report))
    if not frames:
        raise SystemExit("Nothing parsed.")
    long = pd.concat(frames, ignore_index=True)
    wide = (long.pivot_table(index=["district", "season", "season_year"],
                             columns="variable", values="value", aggfunc="first")
                .reset_index())
    wide.columns.name = None
    wide = wide.rename(columns={"yield": "yield_kg_ha", "production": "production_mt",
                                "sown": "sown_ha", "harvested": "harvested_ha"})
    out = C.INTERIM / "paddy_district_long.csv"
    wide.to_csv(out, index=False)
    print(f"\nWrote {out.relative_to(C.ROOT)}  ({len(wide)} district-season rows)")
    _sanity(wide)
    return wide


def _sanity(df: pd.DataFrame) -> None:
    print("\nSanity checks")
    if "yield_kg_ha" in df:
        y = df["yield_kg_ha"]
        print(f"  yield kg/ha: median {y.median():.0f}, min {y.min():.0f}, max {y.max():.0f}"
              "  (expect roughly 2,000-6,000)")
    if {"production_mt", "harvested_ha", "yield_kg_ha"} <= set(df.columns):
        chk = df["harvested_ha"] * df["yield_kg_ha"] / 1000 / df["production_mt"]
        print(f"  harvested x yield / production: median ratio {chk.median():.2f} (expect ~1.0;"
              " DCS yield is per *nett* hectare so 0.9-1.1 is fine)")
    missing = set(C.DISTRICTS) - set(df["district"])
    if missing:
        print(f"  districts not found: {sorted(missing)}")


def _diagnose(path: Path) -> None:
    grids, text = _grids_from_file(path)
    print(f"\n--- Diagnostics for {path.name} ---")
    print(text[:1500])
    if grids:
        print("First grid, first 6 rows:")
        for r in grids[0][:6]:
            print("   ", r[:10])
    print("--- end ---\n Adjust _parse_grid() for this layout, or check the downloaded file.\n")


# --------------------------------------------------------------------------
# Optional: per-season tables with Major / Minor / Rainfed sown extent
# --------------------------------------------------------------------------
def season_labels(first: int = 2004, last: int = 2025) -> list[tuple[str, str, int]]:
    labels = []
    for y in range(first, last + 1):
        labels.append((f"{y}-{y + 1}Maha", "Maha", y))
        if y + 1 <= last:
            labels.append((f"{y + 1}Yala", "Yala", y + 1))
    return labels


def download_seasons(force: bool = False) -> None:
    out_dir = C.RAW_DCS / "seasons"
    out_dir.mkdir(exist_ok=True)
    for label, _, _ in season_labels():
        try:
            download_file(C.DCS_SEASON_URL.format(label=label), label, out_dir, force)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {label}: {e}")
        time.sleep(1)


def parse_seasons() -> pd.DataFrame:
    """Take the first four numbers after each district name as sown extent
    Major / Minor / Rainfed / All and keep rows where the parts add up."""
    rows = []
    for label, season, year in season_labels():
        files = list((C.RAW_DCS / "seasons").glob(label + ".*"))
        if not files:
            continue
        try:
            grids, _ = _grids_from_file(files[0])
        except ValueError:
            continue
        seen = set()
        for g in grids:
            for r in g:
                if not r:
                    continue
                d = _canon_district(r[0])
                if d is None or d in seen:
                    continue
                nums = [v for v in (_num(c) for c in r[1:]) if not np.isnan(v)]
                if len(nums) < 4:
                    continue
                major, minor, rainfed, total = nums[:4]
                if total > 0 and abs(major + minor + rainfed - total) / total < 0.03:
                    rows.append((d, season, year, major, minor, rainfed, total))
                    seen.add(d)
    df = pd.DataFrame(rows, columns=["district", "season", "season_year", "sown_major_ha",
                                     "sown_minor_ha", "sown_rainfed_ha", "sown_all_ha"])
    out = C.INTERIM / "paddy_irrigation_split.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out.relative_to(C.ROOT)}: {len(df)} rows, "
          f"{df[['season', 'season_year']].drop_duplicates().shape[0] if len(df) else 0} seasons")
    return df


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    force = "--force" in sys.argv
    if cmd in ("download", "all"):
        download_series(force)
    if cmd in ("parse", "all"):
        parse_all()
    if cmd == "seasons":
        download_seasons(force)
        parse_seasons()
