import importlib


def test_public_packages_import_without_source_siblings():
    for name in (
        "vipaneltr.data",
        "vipaneltr.evaluation",
        "vipaneltr.system",
        "vipaneltr.baseline",
        "vipaneltr.cli",
    ):
        assert importlib.import_module(name)
