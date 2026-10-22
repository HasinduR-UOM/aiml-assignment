import pandas as pd
import pandera.pandas as pa
import pytest

from src.contract import SCHEMA


def row(**kw):
    base = {"district": "Ampara", "season": "Maha", "season_year": 2010, "yield_kg_ha": 4000.0, "sown_ha": 1.0,
            "rain_total": 900.0, "longest_dry_spell": 20.0, "tmax_mean": 31.0, "fertilizer_crisis": 0}
    return pd.DataFrame([{**base, **kw}])


def test_valid_row_passes():
    SCHEMA.validate(row())


@pytest.mark.parametrize("bad", [{"yield_kg_ha": 0.0}, {"season": "Spring"}, {"district": "Atlantis"}])
def test_bad_rows_fail(bad):
    with pytest.raises(pa.errors.SchemaError):
        SCHEMA.validate(row(**bad))
