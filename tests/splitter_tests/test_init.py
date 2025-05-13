# tests/test_init.py

def test_module_imports():
    from nirs4all.data_splitters import (
        CustomSplitter,
        KBinsStratifiedSplitter,
        KMeansSplitter,
        SystematicCircularSplitter,
        KennardStoneSplitter,
        SPXYSplitter,
        get_splitter,
        run_splitter,
    )
    assert CustomSplitter is not None
    assert KBinsStratifiedSplitter is not None
    assert KMeansSplitter is not None
    assert SystematicCircularSplitter is not None
    assert KennardStoneSplitter is not None
    assert SPXYSplitter is not None
    assert get_splitter is not None
    assert run_splitter is not None
