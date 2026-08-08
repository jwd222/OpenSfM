@echo off
rem Minimal pipeline: SfM -> depthmaps -> dense cloud -> DEM + ortho.
rem Skips `mesh` (web-viewer-only; not needed for the dense products).
rem dense_merging is georeferenced (UTM zone auto-selected from reference_lla.json).

call %~dp0opensfm.bat extract_metadata %1
call %~dp0opensfm.bat detect_features   %1
call %~dp0opensfm.bat match_features    %1
call %~dp0opensfm.bat create_tracks     %1
call %~dp0opensfm.bat reconstruct       %1
python "%~dp0export_camera_poses.py" %1 --proj EPSG:32645 --geoid "C:\Program Files\Agisoft\Metashape Pro\geoids\us_nga_egm2008_1.tif"
call %~dp0opensfm.bat undistort         %1
call %~dp0opensfm.bat dense_equalize    %1
call %~dp0opensfm.bat dense_clustering  %1
call %~dp0opensfm.bat compute_depthmaps %1
call %~dp0opensfm.bat fuse_depthmaps    %1
call %~dp0opensfm.bat dense_merging     %1 --georeferenced
