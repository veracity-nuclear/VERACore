# -*- mode: python ; coding: utf-8 -*-
import sys

is_mac = sys.platform == 'darwin'

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all

datas, binaries, hiddenimports = [], [], []

datas += collect_data_files("vera_core")
hiddenimports += collect_submodules("vera_core")

for pkg in ["trame", "trame_client", "trame_server", "trame_vuetify", "trame_vtk"]:
    datas += collect_data_files(pkg)
    hiddenimports += collect_submodules(pkg)

for pkg in ["vtkmodules", "vtk"]:
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

d, b, h = collect_all("pyvera")
datas += d; binaries += b; hiddenimports += h


a = Analysis(
    ['VeraCore.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VeraCore',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='VeraCoreIcon.icns' if is_mac else "VeraCoreIco.ico",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='VeraCore',
)

if is_mac:
    app = BUNDLE(
        coll,
        name='VeraCore.app',
        icon='VeraCoreIcon.icns',
        bundle_identifier='veracore',
        info_plist={'NSHighResolutionCapable': True},
    )