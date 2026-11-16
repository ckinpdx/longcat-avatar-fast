@echo off
rem Windows: run run_fast.py inside the MSVC x64 environment (torch.compile's inductor needs cl.exe).
rem   run_fast.bat -- <their demo flags>
rem Uses the Python on PATH (activate your venv first) unless LONGCAT_PYTHON points at one.
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
  echo MSVC build tools not found. Install "Desktop development with C++" or set LONGCAT_COMPILE=0.
  exit /b 1
)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
if not defined LONGCAT_PYTHON set "LONGCAT_PYTHON=python"
"%LONGCAT_PYTHON%" "%~dp0run_fast.py" %*
