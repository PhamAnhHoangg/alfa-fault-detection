# Plugin template for the Claude desktop app

This folder is a plugin that starts `src/mcp_server.py` on your own computer, so the Claude desktop app can use its tools.

1. Edit `.mcp.json`: replace the three `X:\your-folder` paths with your Python (inside the project's virtual environment), your copy of `src\mcp_server.py` and your ALFA data folder.
2. Zip the contents of this folder (the `.claude-plugin` folder, `.mcp.json` and this file, not the folder itself) and rename the zip to `alfa-fault-detection.plugin`.
3. In the Claude desktop app: Settings, Plugins, Add, Upload plugin, and pick the file. Restart the app.

Step 2 with Python, run inside this folder (works on Windows, macOS and Linux):

```
python -c "import shutil; shutil.make_archive('../alfa-fault-detection', 'zip', '.')"
```

Then rename `../alfa-fault-detection.zip` to `alfa-fault-detection.plugin`.
