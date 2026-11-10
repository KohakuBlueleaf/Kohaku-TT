"""Builds a self-contained kohakutt wheel: the package, the native library, the
tt-metal libraries it links and the kernel runtime tree of the pinned tt-metal.
Usage: python packaging/build_wheel.py --tt-metal <tree> [--out dist] [--platform TAG]
"""

import argparse
import base64
import hashlib
import itertools
import re
import shutil
import subprocess
import tempfile
import zipfile
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The kernel runtime tree: what tt-metal's own wheel ships (its setup.py), by root.
RUNTIME = {
    "runtime": ["hw/**/*"],
    "ttnn": ["api/ttnn/tensor/enum_types.hpp"],
    "ttnn/cpp": [
        "ttnn/kernel/**/*",
        "ttnn/operations/**/kernels/**/*",
        "ttnn/operations/**/kernels_ng/**/*",
        "ttnn/operations/**/kernels_dfb/**/*",
        "ttnn/operations/**/shared_with_host/**/*",
        "ttnn/operations/**/hostdevcommon/**/*",
        "ttnn/operations/kernel_helper_functions/*",
        "ttnn/operations/ccl/**/*",
        "ttnn/operations/data_movement/**/*",
        "ttnn/operations/moreh/**/*",
        "ttnn/kernel_lib/**/*.{hpp,inl}",
        "ttnn/operations/normalization/kernel_util/**/*",
    ],
    "tt_metal": [
        "api/tt-metalium/buffer_constants.hpp",
        "api/tt-metalium/buffer_types.hpp",
        "api/tt-metalium/circular_buffer_constants.h",
        "api/tt-metalium/constants.hpp",
        "api/tt-metalium/dev_msgs.h",
        "api/tt-metalium/experimental/fabric/fabric_edm_types.hpp",
        "api/tt-metalium/experimental/fabric/edm_fabric_counters.hpp",
        "core_descriptors/*.yaml",
        "fabric/hw/**/*",
        "fabric/mesh_graph_descriptors/*.{yaml,textproto}",
        "fabric/impl/kernels/**/*",
        "hw/**/*",
        "hostdevcommon/api/hostdevcommon/**/*",
        "impl/buffers/kernels/**/*",
        "impl/buffers/dram_sender_state_block.hpp",
        "impl/buffers/tensor_prefetcher_request.hpp",
        "impl/dispatch/kernels/**/*",
        "impl/streaming_profiler/kernels/**/*",
        "include/**/*",
        "kernels/**/*",
        "tt-llk/**/*",
        "tools/profiler/**/*",
        "soc_descriptors/*.yaml",
        "sfpi-version",
    ],
    # The fused attention kernels of tt-train compiled into the native library.
    "tt-train/sources/ttml": ["metal/**/kernels/**/*", "metal/common/**/*"],
}
# Not needed to run kernels.
SKIP = ("/tt-llk/tests/", "/tt-llk/docs/", "/__pycache__/", "/.git")


def expand(pattern):
    """Glob patterns of `pattern` with its {a,b} groups expanded."""
    groups = re.findall(r"\{([^}]*)\}", pattern)
    if not groups:
        return [pattern]
    template = re.sub(r"\{[^}]*\}", "{}", pattern)
    choices = [g.split(",") for g in groups]
    return [template.format(*c) for c in itertools.product(*choices)]


def run(*command, **kwargs):
    return subprocess.run(command, check=True, capture_output=True, text=True, **kwargs)


def bundled_libraries(native, tt_build):
    """(file name to load as, path) of the libraries `native` needs that live in the
    tt-metal build tree."""
    out = []
    for line in run("ldd", str(native)).stdout.splitlines():
        match = re.match(r"\s*(\S+) => (\S+)", line)
        if match and Path(match[2]).resolve().is_relative_to(tt_build):
            out.append((match[1], Path(match[2]).resolve()))
    return out


def copy_runtime(tt_metal, root):
    count = 0
    for base, patterns in RUNTIME.items():
        for pattern in itertools.chain.from_iterable(expand(p) for p in patterns):
            for path in (tt_metal / base).glob(pattern):
                text = path.as_posix()
                if not path.is_file() or any(s in text for s in SKIP):
                    continue
                target = root / path.relative_to(tt_metal)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                count += 1
    return count


def record_line(stage, path):
    data = path.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"{path.relative_to(stage).as_posix()},sha256={digest.decode()},{len(data)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tt-metal", required=True, type=Path)
    parser.add_argument("--tt-build", type=Path, help="default: <tt-metal>/build")
    parser.add_argument("--out", type=Path, default=REPO / "dist")
    parser.add_argument("--platform", default="linux_x86_64")
    parser.add_argument("--patchelf", default="patchelf")
    args = parser.parse_args()
    tt_metal = args.tt_metal.resolve()
    tt_build = (args.tt_build or tt_metal / "build").resolve()
    native = REPO / "src" / "kohakutt_native" / "_ktt_native.so"
    if not native.exists():
        raise SystemExit(f"build the native library first: {native} is missing")

    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        tmp = Path(tmp)
        run("uv", "build", "--wheel", "--out-dir", str(tmp / "base"), str(REPO))
        (base,) = (tmp / "base").glob("*.whl")
        stage = tmp / "stage"
        with zipfile.ZipFile(base) as archive:
            archive.extractall(stage)
        package = stage / "kohakutt_native"
        bundle = package / "_tt"
        (bundle / "lib").mkdir(parents=True)

        shutil.copyfile(native, package / native.name)
        run(args.patchelf, "--set-rpath", "$ORIGIN/_tt/lib", str(package / native.name))
        for name, path in bundled_libraries(native, tt_build):
            shutil.copyfile(path, bundle / "lib" / name)
            run(args.patchelf, "--set-rpath", "$ORIGIN", str(bundle / "lib" / name))
        files = copy_runtime(tt_metal, bundle / "root")
        revision = run("git", "-C", str(tt_metal), "describe", "--tags", "--always")
        (bundle / "tt-metal-version").write_text(revision.stdout)

        (info,) = stage.glob("*.dist-info")
        name, version = info.name.removesuffix(".dist-info").split("-")
        tag = f"py3-none-{args.platform}"
        wheel = (info / "WHEEL").read_text()
        wheel = re.sub(r"Root-Is-Purelib: \w+", "Root-Is-Purelib: false", wheel)
        wheel = re.sub(r"(?m)^Tag: .*\n", "", wheel) + f"Tag: {tag}\n"
        (info / "WHEEL").write_text(wheel)
        # The native library is built against one torch release.
        series = ".".join(metadata.version("torch").split(".")[:2])
        text = (info / "METADATA").read_text()
        text = re.sub(
            r"(?m)^Requires-Dist: torch$", f"Requires-Dist: torch=={series}.*", text
        )
        (info / "METADATA").write_text(text)
        record = info / "RECORD"
        record.unlink()
        paths = sorted(p for p in stage.rglob("*") if p.is_file())
        lines = [record_line(stage, p) for p in paths]
        record.write_text("\n".join([*lines, f"{info.name}/RECORD,,"]) + "\n")

        args.out.mkdir(parents=True, exist_ok=True)
        target = args.out / f"{name}-{version}-{tag}.whl"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in [*paths, record]:
                archive.write(path, path.relative_to(stage).as_posix())
    size = target.stat().st_size / 1e6
    pinned = revision.stdout.strip()
    print(f"{target} ({size:.0f} MB, {files} runtime files, tt-metal {pinned})")


if __name__ == "__main__":
    main()
