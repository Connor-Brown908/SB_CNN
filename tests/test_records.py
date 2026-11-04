from seqparam.data.records import RawRecord


def test_as_row_fields_and_length():
    r = RawRecord(source="cambray2018", element_type="rbs", seq_id="x1",
                  sequence="ACGTACGT", value=42.0, unit="PNI_1_100",
                  value_sd=1.5, meta={"group": 2})
    row = r.as_row()
    assert row["length"] == 8
    assert row["value"] == 42.0 and row["value_sd"] == 1.5
    assert row["element_type"] == "rbs" and row["unit"] == "PNI_1_100"
    assert row["meta"] == {"group": 2}


def test_as_row_handles_missing_sd():
    r = RawRecord("s", "promoter", "p1", "ACGT", 1.0, "TX_au")
    assert r.as_row()["value_sd"] is None


def test_meta_is_copied_not_shared():
    meta = {"k": 1}
    r = RawRecord("s", "rbs", "i", "ACGT", 1.0, "u", meta=meta)
    r.as_row()["meta"]["k"] = 999
    assert meta["k"] == 1
