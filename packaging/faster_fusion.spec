# PyInstaller spec for FASTER Fusion - macOS .app and Windows folder build.
#   macOS:    pyinstaller packaging/faster_fusion.spec   -> dist/FASTER Fusion.app
#   Windows:  pyinstaller packaging\faster_fusion.spec   -> dist\FASTER Fusion\FASTER Fusion.exe
import os
import sys
from PyInstaller.utils.hooks import (collect_data_files, collect_dynamic_libs, collect_submodules,
                                    copy_metadata)

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
sys.path.insert(0, ROOT)
from faster_fusion import __version__

datas = [(os.path.join(ROOT, "faster_fusion", "resources"), "faster_fusion/resources")]
datas += collect_data_files("pyvista")
datas += collect_data_files("imageio_ffmpeg")
datas += collect_data_files("pydicom")
# Package metadata (the *.dist-info): imageio reads its own version from it
# when it starts, so without it Save MP4 fails with "No package metadata was
# found for imageio". The others are included for the same reason.
for pkg in ("imageio", "imageio-ffmpeg", "pydicom", "pyvista", "pyvistaqt", "numpy", "scipy",
            "pyqtgraph", "Pillow"):
    datas += copy_metadata(pkg)
binaries = collect_dynamic_libs("imageio_ffmpeg")
hidden = (collect_submodules("pydicom.pixels") + collect_submodules("vtkmodules")
          + ["pyvistaqt", "PIL._imaging"])

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    excludes=["PyQt5", "PyQt6", "PySide2", "tkinter", "IPython", "jupyter", "notebook", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
icon = os.path.join(SPECPATH, "icon.icns" if sys.platform == "darwin" else "icon.ico")
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="FASTER Fusion", console=False,
          icon=icon, argv_emulation=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="FASTER Fusion", upx=False)
if sys.platform == "darwin":
    app = BUNDLE(coll, name="FASTER Fusion.app", icon=icon, bundle_identifier="org.faster.fusion",
                 version=__version__,
                 info_plist={"NSHighResolutionCapable": True, "CFBundleShortVersionString": __version__,
                             "LSMinimumSystemVersion": "11.0",
                             "NSHumanReadableCopyright": "FASTER Fusion"})
