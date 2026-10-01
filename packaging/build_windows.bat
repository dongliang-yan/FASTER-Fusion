@echo off
REM Builds dist\FASTER Fusion\FASTER Fusion.exe and a zip of it, on Windows.
REM Run from the FASTER_Fusion_App folder:
REM   py -3.12 -m venv .venv
REM   .venv\Scripts\pip install -r requirements.txt
REM   packaging\build_windows.bat
cd /d "%~dp0\.."
set PY=.venv\Scripts\python.exe
%PY% packaging\make_icon.py || exit /b 1
rmdir /s /q build dist 2>nul
%PY% -m PyInstaller --noconfirm --clean --distpath dist --workpath build packaging\faster_fusion.spec || exit /b 1
for /f %%v in ('%PY% -c "import faster_fusion; print(faster_fusion.__version__)"') do set VER=%%v
powershell -NoProfile -Command "Compress-Archive -Force -Path 'dist\FASTER Fusion' -DestinationPath 'dist\FASTER-Fusion-%VER%-Windows.zip'"
echo Built dist\FASTER Fusion\FASTER Fusion.exe and dist\FASTER-Fusion-%VER%-Windows.zip
REM Optional: one-file installer with Desktop/Start Menu shortcuts (needs Inno Setup 6)
set ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe
if not exist "%ISCC%" set ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe
if exist "%ISCC%" ("%ISCC%" /DAppVersion=%VER% packaging\installer.iss && echo Built dist\FASTER-Fusion-%VER%-Setup.exe) else echo Inno Setup not found - skipped the installer.
