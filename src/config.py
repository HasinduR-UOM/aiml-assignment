"""Project-wide settings: paths, districts, seasons, URLs and split years."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_DCS = DATA / "raw" / "dcs"
RAW_WEATHER = DATA / "raw" / "weather"
RAW_ONI = DATA / "raw" / "oni"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
MODELS = ARTIFACTS / "models"

for _p in (RAW_DCS, RAW_WEATHER, RAW_ONI, INTERIM, PROCESSED, FIGURES, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Districts: one representative point per district, placed over the main
# paddy-growing area (not always the district capital). Used to pull ERA5
# weather from Open-Meteo. Coordinates are approximate (0.05 deg is enough
# because ERA5 is ~0.25 deg).
# ---------------------------------------------------------------------------
DISTRICTS = {
    "Colombo":      (6.85, 80.05),
    "Gampaha":      (7.10, 80.00),
    "Kalutara":     (6.60, 80.10),
    "Kandy":        (7.30, 80.65),
    "Matale":       (7.55, 80.65),
    "Nuwara Eliya": (7.00, 80.80),
    "Galle":        (6.20, 80.30),
    "Matara":       (6.05, 80.55),
    "Hambantota":   (6.25, 81.15),
    "Jaffna":       (9.65, 80.05),
    "Kilinochchi":  (9.40, 80.40),
    "Mannar":       (8.85, 80.05),
    "Vavuniya":     (8.75, 80.50),
    "Mullaitivu":   (9.15, 80.65),
    "Batticaloa":   (7.70, 81.60),
    "Ampara":       (7.25, 81.70),
    "Trincomalee":  (8.40, 81.10),
    "Kurunegala":   (7.50, 80.35),
    "Puttalam":     (8.00, 80.00),
    "Anuradhapura": (8.30, 80.40),
    "Polonnaruwa":  (7.95, 81.00),
    "Badulla":      (7.10, 81.10),
    "Monaragala":   (6.80, 81.30),
    "Ratnapura":    (6.70, 80.40),
    "Kegalle":      (7.25, 80.35),
}

# Spellings seen in DCS tables -> canonical name (lower-case keys, no spaces)
DISTRICT_ALIASES = {
    "nuwaraeliya": "Nuwara Eliya", "n'eliya": "Nuwara Eliya", "n.eliya": "Nuwara Eliya",
    "moneragala": "Monaragala", "monaragala": "Monaragala",
    "trincomalee": "Trincomalee", "trinco": "Trincomalee",
    "anuradhapura": "Anuradhapura", "polonnaruwa": "Polonnaruwa",
    "mullaitivu": "Mullaitivu", "mullaithivu": "Mullaitivu", "mullativu": "Mullaitivu",
    "kilinochchi": "Kilinochchi", "killinochchi": "Kilinochchi",
    "batticaloa": "Batticaloa", "kegalle": "Kegalle", "kegalla": "Kegalle",
    "ratnapura": "Ratnapura", "rathnapura": "Ratnapura",
    "hambantota": "Hambantota", "puttalam": "Puttalam", "kurunegala": "Kurunegala",
    "digamadulla": "Ampara", "amparai": "Ampara", "ampara": "Ampara",
}
for _d in DISTRICTS:
    DISTRICT_ALIASES.setdefault(_d.lower().replace(" ", ""), _d)

# Districts the analyst may want to drop (very small, urban paddy area).
LOW_PADDY_DISTRICTS = ["Colombo", "Nuwara Eliya"]

# ---------------------------------------------------------------------------
# Seasons. season_year = calendar year in which the season STARTS.
#   Maha 1979/80 -> season_year 1979, window Oct 1979 - Mar 1980
#   Yala 1980    -> season_year 1980, window Apr 1980 - Sep 1980
# ---------------------------------------------------------------------------
SEASONS = {
    "Maha": {"start_month": 10, "n_months": 6, "t_offset": 0.75},
    "Yala": {"start_month": 4, "n_months": 6, "t_offset": 0.25},
}

# WMO-style climatological base period (fixed, before the test period, so
# anomaly features do not leak information from the future).
CLIMATE_BASE = (1981, 2010)

# Temporal split. Everything with season_year <= TRAIN_END is used for model
# selection (with expanding-window CV); later seasons are a held-out test.
TRAIN_END = 2015
CV_VAL_BLOCKS = [(2001, 2003), (2004, 2006), (2007, 2009), (2010, 2012), (2013, 2015)]

# Known structural shock: chemical fertilizer import ban (May 2021) and the
# shortage that followed. Flags Maha 2021/22 and Yala 2022.
FERTILIZER_CRISIS = {("Maha", 2021), ("Yala", 2022)}

# Shortfall rule used for the early-warning view and classification metrics.
SHORTFALL_THRESHOLD = 0.85   # yield < 85 % of the district's previous 5 same-season mean

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
DCS_BASE = "https://www.statistics.gov.lk/Agriculture/StaticalInformation/PaddyStatistics"
DCS_SERIES = {
    # key: (url slug, variable, season)
    "yield_maha":      ("Average_Yeild-Maha_Season-1979-2025", "yield", "Maha"),
    "yield_yala":      ("Average_Yeild-Yala_Season-1979-2025", "yield", "Yala"),
    "production_maha": ("Production-Maha_Season-1979-2025", "production", "Maha"),
    "production_yala": ("Production-Yala_Season-1979-2025", "production", "Yala"),
    "sown_maha":       ("Sown_Extent-Maha_Seasons-1979-2025", "sown", "Maha"),
    "sown_yala":       ("Sown_Extent-Yala_Seasons-1979-2025", "sown", "Yala"),
    "harvested_maha":  ("Harvested_Extent-Maha_Seasons-1979-2025", "harvested", "Maha"),
    "harvested_yala":  ("Harvested_Extent-Yala_Seasons-1979-2025", "harvested", "Yala"),
}
# Per-season district tables (Major / Minor / Rainfed split), 2004/05 onward
DCS_SEASON_URL = DCS_BASE + "/MetricUnits/IncludingMahaweli/{label}"

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_VARS = ["precipitation_sum", "temperature_2m_max", "temperature_2m_min",
                "et0_fao_evapotranspiration"]
WEATHER_START = "1979-01-01"

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

# Unit conversions
BUSHEL_PADDY_KG = 20.87          # 1 bushel of paddy (46 lb) in kg
ACRE_HA = 0.404686
BU_PER_AC_TO_KG_PER_HA = BUSHEL_PADDY_KG / ACRE_HA   # ~51.57
