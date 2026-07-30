import impact_embedding_core


def test_package_imports():
    assert impact_embedding_core is not None


def test_public_api_is_importable_from_the_package_root():
    for name in impact_embedding_core.__all__:
        assert hasattr(impact_embedding_core, name), f"missing export: {name}"
