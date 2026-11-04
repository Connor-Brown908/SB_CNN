import math

from seqparam.data.loaders import load_cambray, load_lafleur


def test_load_cambray_maps_columns_and_skips_nan(cambray_zip):
    recs = load_cambray(cambray_zip)
    # NaN-value row is skipped at load; the >100 and duplicate rows survive to cleaning
    vals = sorted(r.value for r in recs)
    assert vals == [40.0, 55.0, 70.0, 250.0]
    r0 = next(r for r in recs if r.value == 40.0)
    assert r0.element_type == "rbs" and r0.unit == "PNI_1_100"
    assert math.isclose(r0.value_sd, 2.0)         # sqrt(var=4)
    assert len(r0.sequence) == 96
    assert r0.meta["group"] == 1


def test_load_cambray_limit(cambray_zip):
    assert len(load_cambray(cambray_zip, limit=1)) <= 1


def test_load_lafleur_three_sources_and_concatenation(lafleur_xlsx):
    recs = load_lafleur(lafleur_xlsx)
    by_source = {}
    for r in recs:
        by_source.setdefault(r.source, []).append(r)

    assert set(by_source) == {"lafleur2022", "urtecho2019", "hossain2020"}
    assert all(r.element_type == "promoter" and r.unit == "TX_au" for r in recs)

    # Fig 1d sequence is UP+h35+spacs+h10 concatenated
    f = by_source["lafleur2022"][0]
    assert f.sequence == "ACGTACGTACGTACGT" + "TTGACA" + "CTATTTCCTATTTCTCT" + "TATAAT"
    assert math.isclose(f.value, 2.5) and math.isclose(f.value_sd, 0.3)

    # Urtecho NaN-value row is skipped
    assert len(by_source["urtecho2019"]) == 1

    # Hossain concatenates upstream+promoter+downstream
    h = by_source["hossain2020"][0]
    assert h.sequence.startswith("ACGTACGTAC") and h.sequence.endswith("TTTTGGGG")
