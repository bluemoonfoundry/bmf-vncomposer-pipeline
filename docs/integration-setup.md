# Blender 5.1 + DAZ Studio 4 setup

The installed executables are recorded in [`config/toolchain.json`](../config/toolchain.json). Both paths were verified on the development machine:

```text
C:\Program Files\Blender Foundation\Blender 5.1\blender.exe
X:\DAZNext\Applications\64-bit\DAZ 3D\DAZStudio4\DAZStudio.exe
```

## Versions to install

Use Diffeomorphic **5.1.0** for Blender 5.1. The maintainer’s 5.1.0 release was specifically tested on Blender 5.1. Do not install the repository `master` snapshot on this machine: its current manifest advertises Blender 5.3.

Use the **DAZ Studio 4** build of DazScriptServer matching the installed plugin. This machine currently has `DazScriptServer.dll` under `...\\plugins\\DazScriptServer\\Release` and `dazpy` **2.9.2**. Keep those versions paired; do not replace the working 2.9.2 install with the older public v2.9.0 tag unless we intentionally create a clean rollback.

## Diffeomorphic installation

1. Download `import_daz-5.1.0.zip` from the Diffeomorphic downloads page.
2. In Blender 5.1, install the zip through Preferences → Extensions/Add-ons → Install.
3. Ensure the installed package directory is named `import_daz`.
4. Enable the importer and its required feature modules.
5. Copy the Diffeomorphic DAZ-side scripts from the archive’s `to_daz_studio/Scripts` directory into a DAZ content-library `Scripts` directory. Do not put these scripts in the DAZ executable folder.
6. In DAZ Studio, configure the importer’s Blender path to the executable in `config/toolchain.json`.

For an upgrade, disable the old importer, close Blender, remove the old `import_daz` directory from Blender’s user add-on directory, then install the new archive. Mixing files from different Diffeomorphic releases is a known source of import failures.

## DazScriptServer installation

1. If a clean reinstall is needed, obtain the DS4 Windows DLL from the matching daz-script-server release/build. The DS4 DLL is distinct from the DS6 DLL.
2. Close DAZ Studio.
3. Remove/rename any older `DazScriptServer*.dll` in the DAZ plugins directory, then copy the DS4 DLL into:

   ```text
   X:\DAZNext\Applications\64-bit\DAZ 3D\DAZStudio4\plugins
   ```

4. Start DAZ Studio, open Window → Panes → Daz Script Server, and start the server. Keep it bound to `127.0.0.1` for local automation and copy the API token if authentication is enabled.
5. Install the matching Python SDK:

   ```powershell
   python -m pip install dazpy==2.9.2
   ```

The default server endpoint is `http://127.0.0.1:18811`.

## First validation sequence

Run these checks before attempting a full character bake:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' --background --python-expr "import bpy; print(bpy.app.version_string)"
python -c "import dazpy; print(dazpy.__version__)"
```

Then use the DazScriptServer health endpoint or `dazpy` health call, open a tiny Genesis test scene in DAZ Studio, and verify that Diffeomorphic produces both the exported `.dbz` and `.duf`-side data before wiring it into the batch worker.

## Sources

- Diffeomorphic importer: https://github.com/Diffeomorphic/import_daz
- Diffeomorphic Blender 5.1.0 release announcement: https://www.daz3d.com/forums/discussion/689606/diffeomorphic-add-ons-version-4-5-0-released
- DazScriptServer repository: https://github.com/bluemoonfoundry/daz-script-server
- DazScriptServer public release page: https://github.com/bluemoonfoundry/daz-script-server/releases/latest
