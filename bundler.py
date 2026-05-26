# myapp.spec
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all

datas, binaries, hiddenimports = [], [], []

# Problem 1: Trame's frontend is shipped as DATA files inside these packages.
# collect_data_files grabs the served JS/CSS; collect_submodules covers any
# widget modules Trame registers dynamically. List only packages you import.
for pkg in ["trame", "trame_client", "trame_server",
            "trame_vuetify", "trame_vtk"]:
    datas += collect_data_files(pkg)
    hiddenimports += collect_submodules(pkg)

# Problem 2: VTK loads vtkmodules dynamically and ships compiled libs + data.
# collect_all returns all three buckets. The contrib hook may already cover
# most of this; this is the explicit fallback.
vtk_datas, vtk_binaries, vtk_hidden = collect_all("vtkmodules")
datas += vtk_datas
binaries += vtk_binaries
hiddenimports += vtk_hidden

# PyWebView picks its window backend at runtime, so force its submodules in.
hiddenimports += collect_submodules("webview")

a = Analysis(
    ["app.py"],              # your entry point — the script Stage 1 ran
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,   # onedir: binaries go in COLLECT below
    name="MyApp",
    console=True,            # keep True while debugging; flip to False for release
)

coll = COLLECT(exe, a.binaries, a.datas, name="MyApp")