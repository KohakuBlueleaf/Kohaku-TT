"""Doctor report built from a recorded snapshot and fake sysfs links."""

import pytest

from kohakutt.core.doctor import format_report, run_doctor

SNAPSHOT = {
    "host_info": {"Hostname": "h", "Driver": "TT-KMD 2.12.0"},
    "host_sw_vers": {"tt_smi": "6.0.0", "tt_umd": "0.9.5"},
    "device_info": [
        {
            "board_info": {
                "bus_id": "0000:c1:00.0",
                "board_type": "p150a",
                "dram_status": True,
            },
            "telemetry": {
                "aiclk": " 800",
                "power": " 41.0",
                "asic_temperature": "48.8",
            },
            "smbus_telem": {"ENABLED_TENSIX_COL": "0xfff"},
            "firmwares": {"fw_bundle_version": "19.13.1.0"},
            "limits": {"asic_fmax": "1350"},
        }
    ],
}


def link(speed, width):
    return {
        "0000:c1:00.0": {
            "speed": speed,
            "width": width,
            "max_speed": "32.0 GT/s PCIe",
            "max_width": "16",
        }
    }


def test_healthy_card_has_no_warnings():
    report = run_doctor(SNAPSHOT, link("16.0 GT/s PCIe", "8"))
    card = report["cards"][0]
    assert card["tensix_columns"] == 12 and card["aiclk_mhz"] == 800.0
    assert report["warnings"] == []
    assert "card 0" in format_report(report)


@pytest.mark.parametrize("speed", ["2.5 GT/s PCIe", "8.0 GT/s PCIe"])
def test_downtrained_link_is_warned(speed):
    report = run_doctor(SNAPSHOT, link(speed, "8"))
    assert any("PCIe" in w for w in report["warnings"])


def test_missing_snapshot_is_reported():
    report = run_doctor({"device_info": []}, {})
    assert report["cards"] == []
