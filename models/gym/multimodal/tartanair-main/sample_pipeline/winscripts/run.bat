@REM 1. assume the environment names are consistent, change them manually if they aren't 
        @REM - packaged folder name
        @REM - the exe file name
        @REM - trajs folder name

@echo off
@REM CHANGE THE FOLLOWING PARAMETERS
set env=AbandonedSchool
set gamma=1.0
set seglabel=C:\tartanair-v2\Package\AbandonedSchool\AbandonedSchool\Content\buffered_id_map_DemoMap_main.json

@REM DOUBLE CHECK THE FOLLOWING PARAMETERS
set codedir=C:\programs\tartanair\sample_pipeline\src
set envdir=C:\tartanair-v2\Package\%env%
set posedir=C:\tartanair-v2\trajs\%env%
set outputdir_root=C:\TartanAir_v2
set outputdir=%outputdir_root%\%env%
set envexe=%env%.exe

@REM set gamma value
cd /d %codedir%
start /wait python misc\AirsimSetting.py --data-collection --gamma %gamma%

cd /d %envdir%
start %envexe%

cd /d %outputdir_root%
if not exist "%env%" mkdir %env%
cd %env%

cd /d %codedir%
@REM start the data collection
start /wait python -m collection.collect_images --environment-dir %outputdir% --posefile-folder %posedir%\pose_easy --data-folder Data_easy --cam-list 0_1_2_3_4_5_6_7_8_9_10_11
start /wait python -m collection.collect_images --environment-dir %outputdir% --posefile-folder  %posedir%\pose_hard --data-folder Data_hard --cam-list 0_1_2_3_4_5_6_7_8_9_10_11
TASKKILL /IM %envexe% /F /T

@REM data verification
copy %seglabel% %outputdir%\seg_label.json
start /wait python -m postprocessing.data_validation --data-root %outputdir_root% --env-folders %env% --data-folders Data_easy,Data_hard --analyze-depth --analyze-rgb  --analyze-seg --rgb-depth-filter --create-video
