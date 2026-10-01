# FASTER Fusion (app)

A standalone app for calibrating and reconstructing mirror-swept ultrasound in 3-D:
B-mode cines, colour flow with the SIVV volumetric flow rate, and shear-wave elastography.
It's the MATLAB **FASTER_Fusion** ported to Python, with the same modes, layouts and
numbers, and it doesn't need MATLAB or Python to run.

| rail | does what the MATLAB GUI did |
|---|---|
| **Calibration → Stepped** | `Calibration_CF`: index tabs on a stepped position set (for Color Flow, SWE) |
| **Calibration → Cine** | `Calibration_GUI_v1`: index tabs on a continuous cine (for B-Mode) |
| **B-Mode** | `Recon_v1`: Siemens DICOM cine or Verasonics IQ |
| **Color Flow** | `Recon_CF_v2`: velocity/speed volume and SIVV flow rate |
| **SWE** | `Recon_SWE`: velocity/quality volume |

## Install and run

**macOS:** open `FASTER-Fusion-1.2.2-macOS.dmg` and drag **FASTER Fusion** to Applications.
The app isn't code-signed, so the first time you open it, **right-click → Open → Open**.
After that it opens normally.

**Windows:** run `FASTER-Fusion-1.2.2-Setup.exe` (adds Start Menu and optional Desktop shortcuts), or unzip `FASTER-Fusion-1.2.2-Windows.zip` anywhere and run
`FASTER Fusion\FASTER Fusion.exe`. See *Building* for how the zip is made.

**Data:** the file dialogs start in your **Data** folder. It's found automatically next to
the app's source tree, or you can set it with **Data folder…** on the rail (remembered).
`Data/README.md` says which calibration goes with which scan.

**Saving:** calibrations are saved by default into the session folder they came from, as
`CAL_<session>_<direction>.mat`, so they sit next to the scans they're for. `.mat`
calibrations are read and written in MATLAB's format, so the app and the MATLAB GUIs can
use each other's files.

**Checking an install:** this loads a colour-flow session, reconstructs it, computes SIVV,
writes a report and quits:

```bash
"/Applications/FASTER Fusion.app/Contents/MacOS/FASTER Fusion" --selftest ".../Data" report.txt
```

On the phantom sets the report ends `SIVV Q = 148.10 mL/min at 69.24 mm`, `RESULT OK`.

## How it's built

```
faster_fusion/
  core/          the algorithms, with no GUI (numpy/scipy)
    mlcompat.py    MATLAB built-ins numpy doesn't reproduce: findpeaks, round,
                   movmean/movmedian, imresize3, imgaussfilt, interp1, max(A(:))
    io.py          stepped sets (loadPositionStack/loadCFStack), cines and IQ (loadScanStack)
    sweep.py       findSweepBlocks, beamscopeSweep
    sector.py      SectorRecon, with the weights built once for all lateral lines
    tabs.py        tab snapping, spacing, the angle fit (both calibrations)
    decode.py      colour-flow bar + decode, SWE bars + blend inversion
    recon.py       calibrations, angles, NaN-aware sweeps, volumes, SIVV
  ui/            PySide6 (Qt) pages; 2-D in pyqtgraph, 3-D in VTK (pyvista)
tests/
  test_core_vs_matlab.py   the port against MATLAB outputs on the real data
  test_core_synthetic.py   MATLAB semantics; SIVV against a closed-form flow rate
  make_reference.m         writes tests/ref/*.mat from the MATLAB originals
packaging/                 PyInstaller spec, icon, build scripts
```

## Validation against MATLAB

`tests/test_core_vs_matlab.py` runs every stage on the real data and compares it with the
MATLAB output (`tests/ref`, written by `make_reference.m`):

| stage | agreement |
|---|---|
| stepped-set loading (every pixel of every position) | identical |
| colour-flow bar, decode, box, misfit | identical (velocities to 1e-6) |
| SWE bars, opacity fit, decode | identical |
| cine JPEG decode → grey | identical (max difference 0) |
| Verasonics IQ log compression | to 1e-4 grey levels |
| turnarounds, sweep blocks, direction classes | identical |
| B-mode 2-D frame | to 8e-12 |
| tab snapping + angle fit from the same clicks | to 4e-15 |
| colour-flow 2-D frame, SIVV Q profile | to 1e-9 |
| 3-D volume; isotropic resample | identical; to 2.4e-7 (7 of 11 M voxels sit on the empty/non-empty threshold) |

In the packaged app, `--selftest` gives Q = 148.10 mL/min, the same as MATLAB `Recon_CF_v2`.

## Developing

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m faster_fusion colorflow      # run from source
.venv/bin/python -m pytest -q                     # tests (the MATLAB ones need ../Data)
```

## Building

**macOS** (on a Mac): `bash packaging/build_mac.sh` → `dist/FASTER Fusion.app` and
`dist/FASTER-Fusion-<version>-macOS.dmg`. The app is built for the Mac's own architecture
(Apple silicon here).

**Windows** (on a Windows PC, Python 3.12):

```bat
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
packaging\build_windows.bat
```

→ `dist\FASTER Fusion\FASTER Fusion.exe` and `dist\FASTER-Fusion-<version>-Windows.zip`.
A Windows app has to be built on Windows. If the folder is on GitHub,
`.github/workflows/build.yml` builds both platforms on GitHub's machines and attaches the
dmg and zip to each run.

**Signing:** to get rid of the "unidentified developer" step on macOS, the app needs signing
and notarizing with an Apple Developer ID. On Windows, a code-signing certificate avoids the
SmartScreen prompt. Neither is needed to use the app.
