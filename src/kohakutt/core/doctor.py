"""`ktt doctor`: a read-only report of the host and every Tenstorrent card.

Sources: a `tt-smi -s` telemetry snapshot (firmware, clocks, power, temperature) and
PCIe link state from sysfs. Nothing is written to any card.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

TT_VENDOR = "0x1e52"
# PCIe link each card should train to on this host (Gen4 x8 per card, bifurcated slot).
EXPECTED_LINK = os.environ.get("KTT_EXPECT_PCIE", "16.0 GT/s PCIe x8")


def find_tt_smi():
    """Path of the tt-smi executable: $KTT_TT_SMI, then PATH."""
    for candidate in (os.environ.get("KTT_TT_SMI"), shutil.which("tt-smi")):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def tt_smi_snapshot(timeout=60):
    """The parsed `tt-smi -s` snapshot, or None when tt-smi is missing or fails."""
    tt_smi = find_tt_smi()
    if tt_smi is None:
        return None
    proc = subprocess.run(
        [tt_smi, "-s", "--snapshot_no_tty", "--offline"],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd="/tmp",
    )
    if proc.returncode != 0:
        return None
    start = proc.stdout.find("{")
    return json.loads(proc.stdout[start:]) if start >= 0 else None


def pcie_links(root="/sys/bus/pci/devices"):
    """{bus_id: link dict} for every Tenstorrent PCIe function, from sysfs."""
    links = {}
    for dev in sorted(Path(root).iterdir()):
        if _read(dev / "vendor") != TT_VENDOR:
            continue
        links[dev.name] = {
            "speed": _read(dev / "current_link_speed"),
            "width": _read(dev / "current_link_width"),
            "max_speed": _read(dev / "max_link_speed"),
            "max_width": _read(dev / "max_link_width"),
        }
    return links


def run_doctor(snapshot=None, links=None):
    """The report: host, software and one entry per card, each with warnings."""
    snapshot = tt_smi_snapshot() if snapshot is None else snapshot
    links = pcie_links() if links is None else links
    report = {"host": {}, "software": {}, "cards": [], "warnings": []}
    if snapshot is None:
        report["warnings"].append("tt-smi snapshot unavailable")
        snapshot = {"device_info": []}
    report["host"] = snapshot.get("host_info", {})
    report["software"] = snapshot.get("host_sw_vers", {})

    for index, dev in enumerate(snapshot.get("device_info", [])):
        card = _card(index, dev, links)
        report["cards"].append(card)
        report["warnings"] += [f"card {index}: {w}" for w in card["warnings"]]
    return report


def _card(index, dev, links):
    """One card's identity, firmware, link, clocks, power and temperature."""
    board = dev.get("board_info", {})
    telem = dev.get("telemetry", {})
    smbus = dev.get("smbus_telem", {})
    fw = dev.get("firmwares", {})
    bus = board.get("bus_id")
    link = links.get(bus, {})
    tensix_cols = bin(int(smbus.get("ENABLED_TENSIX_COL") or "0", 16)).count("1")

    card = {
        "index": index,
        "bus_id": bus,
        "board_type": board.get("board_type"),
        "board_id": board.get("board_id"),
        "firmware": {
            "bundle": fw.get("fw_bundle_version"),
            "cm": fw.get("cm_fw"),
            "eth": fw.get("eth_fw"),
            "dm_app": fw.get("dm_app_fw"),
            "gddr": fw.get("gddr_fw"),
        },
        "pcie": link,
        "aiclk_mhz": _num(telem.get("aiclk")),
        "aiclk_max_mhz": _num(dev.get("limits", {}).get("asic_fmax")),
        "power_w": _num(telem.get("power")),
        "current_a": _num(telem.get("current")),
        "vcore_v": _num(telem.get("voltage")),
        "asic_temp_c": _num(telem.get("asic_temperature")),
        "fan_rpm": _num(telem.get("fan_speed")),
        "tdp_limit_w": _num(dev.get("limits", {}).get("tdp_limit")),
        "tensix_columns": tensix_cols,
        "dram_ok": board.get("dram_status"),
        "dram_speed": board.get("dram_speed"),
        "warnings": [],
    }

    # Warnings: link below the expected training, DRAM down, hot ASIC.
    if link and f"{link['speed']} x{link['width']}" != _expected(EXPECTED_LINK):
        card["warnings"].append(
            f"PCIe {link['speed']} x{link['width']} (expected {EXPECTED_LINK}); "
            "retrain manually after a reset"
        )
    if not link:
        card["warnings"].append("no sysfs PCIe link found for this bus id")
    if board.get("dram_status") is False:
        card["warnings"].append("DRAM training failed")
    if (card["asic_temp_c"] or 0) > 85:
        card["warnings"].append(f"ASIC at {card['asic_temp_c']} C")
    return card


def format_report(report):
    """The report as aligned text lines."""
    host, sw = report["host"], report["software"]
    lines = [
        f"host {host.get('Hostname')}  {host.get('Driver')}  "
        f"tt-smi {sw.get('tt_smi')}  umd {sw.get('tt_umd')}"
    ]
    for c in report["cards"]:
        link = c["pcie"]
        lines.append(
            f"card {c['index']}  {c['board_type']} {c['bus_id']}"
            f"  fw {c['firmware']['bundle']}"
            f"  pcie {link.get('speed')} x{link.get('width')}"
            f"  aiclk {c['aiclk_mhz']}/{c['aiclk_max_mhz']} MHz  {c['power_w']} W"
            f"  {c['asic_temp_c']} C  tensix cols {c['tensix_columns']}"
        )
    lines += [f"warning: {w}" for w in report["warnings"]] or ["no warnings"]
    return "\n".join(lines)


def _expected(text):
    """'16.0 GT/s PCIe x8' -> '16.0 GT/s PCIe x8' as speed + width string."""
    speed, _, width = text.rpartition(" x")
    return f"{speed} x{width}"


def _num(value):
    """A telemetry string as a float, or None."""
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _read(path):
    """A sysfs attribute as stripped text, or None."""
    try:
        return path.read_text().strip()
    except OSError:
        return None
