@echo off
REM Precompile all HLSL compute shaders to .cso bytecode
REM Uses fxc.exe from Windows SDK (sets up VS environment first)
REM Output goes to shaders/ directory

call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

if not exist shaders mkdir shaders

set FLAGS=/T cs_5_0 /E CSMain /O3 /nologo

set ERRORS=0
setlocal enabledelayedexpansion

for %%f in (nn_*.hlsl) do (
    echo Compiling %%f ...
    fxc.exe %FLAGS% /Fo shaders/%%~nf.cso %%f
    if errorlevel 1 (
        echo   FAILED: %%f
        set /a ERRORS+=1
    )
)

REM Also compile non-nn shaders if needed
for %%f in (matmul_coarsened.hlsl matmul_tiled.hlsl matmul_tiled_unroll.hlsl matmul.hlsl matmul_coarsened_2d.hlsl) do (
    if exist %%f (
        echo Compiling %%f ...
        fxc.exe %FLAGS% /Fo shaders/%%~nf.cso %%f
        if errorlevel 1 (
            echo   FAILED: %%f
            set /a ERRORS+=1
        )
    )
)

if !ERRORS! == 0 (
    echo.
    echo All shaders compiled successfully to shaders\*.cso
) else (
    echo.
    echo !ERRORS! shader(s) failed to compile!
)
endlocal
