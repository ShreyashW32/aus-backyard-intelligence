@echo off
if "%QGIS_ROOT%"=="" set "QGIS_ROOT=%ProgramFiles%\QGIS"
if not exist "%QGIS_ROOT%\bin\o4w_env.bat" (
  echo Set QGIS_ROOT to your QGIS install directory, for example C:\Program Files\QGIS 3.44.9
  exit /b 1
)
call "%QGIS_ROOT%\bin\o4w_env.bat"
path %OSGEO4W_ROOT%\apps\qgis-ltr\bin;%PATH%
set QGIS_PREFIX_PATH=%OSGEO4W_ROOT:\=/%/apps/qgis-ltr
set QT_PLUGIN_PATH=%OSGEO4W_ROOT%\apps\qgis-ltr\qtplugins;%OSGEO4W_ROOT%\apps\qt5\plugins
"%OSGEO4W_ROOT%\bin\qgis-ltr-bin.exe" --help
