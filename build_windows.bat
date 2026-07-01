@echo off
REM ============================================================
REM  build_windows.bat — Build iColoriT LoRA Windows executable
REM  Run this script from the repo root with the venv activated.
REM
REM  Prerequisites (run once):
REM    pip install pyinstaller
REM    pip install -r requirements.txt
REM
REM  Usage:
REM    build_windows.bat
REM
REM  Output:
REM    dist\iColoriT_LoRA\iColoriT_LoRA.exe
REM ============================================================

setlocal EnableDelayedExpansion

echo ============================================================
echo  iColoriT LoRA — Windows build
echo ============================================================
echo.

REM ── 1. Verify Python & PyInstaller ───────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Activate your virtual environment first.
    pause & exit /b 1
)

pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] PyInstaller not found — installing...
    pip install pyinstaller
    if errorlevel 1 ( echo [ERROR] pip install failed. & pause & exit /b 1 )
)

REM ── 2. Convert icon (optional, requires Pillow) ───────────────────────────
if exist "gui\icon.png" (
    if not exist "gui\icon.ico" (
        echo [INFO] Converting icon.png to icon.ico...
        python -c "from PIL import Image; img=Image.open('gui/icon.png'); img.save('gui/icon.ico')" 2>nul
        if not errorlevel 1 (
            echo [INFO] Icon converted: gui\icon.ico
            REM Patch spec to use the .ico
            powershell -Command "(Get-Content icolorit_v2.spec) -replace '# icon=''gui/icon.ico''', 'icon=''gui/icon.ico''' | Set-Content icolorit_v2.spec"
        ) else (
            echo [WARN] Icon conversion failed — building without custom icon.
        )
    )
)

REM ── 3. Clean previous build ───────────────────────────────────────────────
echo [INFO] Cleaning previous build artifacts...
if exist "build\iColoriT_LoRA" rmdir /s /q "build\iColoriT_LoRA"
if exist "dist\iColoriT_LoRA"  rmdir /s /q "dist\iColoriT_LoRA"

REM ── 4. Run PyInstaller ────────────────────────────────────────────────────
echo [INFO] Running PyInstaller...
pyinstaller icolorit_v2.spec --noconfirm --clean
if errorlevel 1 (
    echo.
    echo [ERROR] PyInstaller failed. Check output above.
    pause & exit /b 1
)

REM ── 5. Copy checkpoint + sample ──────────────────────────────────────────
echo [INFO] Copying model checkpoint and samples...
set DIST=dist\iColoriT_LoRA

if exist "checks" (
    if not exist "%DIST%\checks" mkdir "%DIST%\checks"
    echo [INFO]  Copying checks\...
    xcopy /e /i /y "checks" "%DIST%\checks" >nul
)

if not exist "%DIST%\samples" mkdir "%DIST%\samples"
if exist "samples" (
    xcopy /e /i /y "samples" "%DIST%\samples" >nul
)

REM ── 6. Create launcher batch ──────────────────────────────────────────────
echo [INFO] Creating launcher script...
(
    echo @echo off
    echo REM  iColoriT LoRA Launcher
    echo REM  Edit --model_path and --target_image as needed.
    echo.
    echo set MODEL=checks\Lora_r32_Dataset_4_epochs_1000_wES_lrf_1e-3.pth
    echo set IMAGE=samples\11.jpeg
    echo.
    echo if not exist "%%MODEL%%" (
    echo     echo [WARN] Model file not found: %%MODEL%%
    echo     echo        Please copy your .pth checkpoint to the checks\ folder.
    echo     pause
    echo ^)
    echo.
    echo iColoriT_LoRA.exe --model_path "%%MODEL%%" --target_image "%%IMAGE%%"
) > "%DIST%\run_icolorit.bat"

REM ── 7. Done ───────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo  BUILD COMPLETE
echo  Executable: %DIST%\iColoriT_LoRA.exe
echo  Launcher:   %DIST%\run_icolorit.bat
echo ============================================================
echo.
echo  To distribute:
echo    Zip the entire  dist\iColoriT_LoRA\  folder and share it.
echo    Users only need to copy their .pth file into checks\
echo    and double-click run_icolorit.bat.
echo.
pause
