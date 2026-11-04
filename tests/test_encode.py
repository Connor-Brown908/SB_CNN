import numpy as np

from seqparam.data.encode import gc_content, is_valid_sequence, one_hot, one_hot_batch


def test_one_hot_shape_and_onehot_columns():
    x = one_hot("ACGT", 4)
    assert x.shape == (4, 4)
    assert np.array_equal(x, np.eye(4, dtype=np.float32))  # A,C,G,T on the diagonal
    assert np.all(x.sum(axis=0) == 1)                      # exactly one base per column


def test_one_hot_pads_and_truncates():
    assert one_hot("ACGT", 6)[:, 4:].sum() == 0            # padded columns all zero
    assert one_hot("ACGTAC", 3).shape == (4, 3)            # truncated to length


def test_one_hot_invalid_char_is_zero_column():
    x = one_hot("ANGT", 4)                                 # N is not in alphabet
    assert x[:, 1].sum() == 0                              # that column is all zeros
    assert x[:, 0].sum() == 1 and x[:, 3].sum() == 1


def test_lowercase_handled():
    assert np.array_equal(one_hot("acgt", 4), one_hot("ACGT", 4))


def test_batch_shape():
    assert one_hot_batch(["AC", "GT", "AA"], 2).shape == (3, 4, 2)


def test_is_valid_sequence():
    assert is_valid_sequence("ACGT")
    assert not is_valid_sequence("ACGN")
    assert not is_valid_sequence("")


def test_gc_content():
    assert gc_content("GGCC") == 1.0
    assert gc_content("ATAT") == 0.0
    assert gc_content("") == 0.0
    assert abs(gc_content("ACGT") - 0.5) < 1e-9
