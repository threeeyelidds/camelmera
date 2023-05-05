@REM 1. assume the environment names are consistent, change them manually if they aren't 
        @REM - packaged folder name
        @REM - the exe file name
        @REM - trajs folder name

@echo off
@REM CHANGE THE FOLLOWING PARAMETERS
set env=CastleFortressExposure
set gamma=1.0
set seglabel=C:\tartanair-v2\Package\CastleFortressExposure\CastleFortress\Content\buffered_id_map_Demonstration.json

@REM DOUBLE CHECK THE FOLLOWING PARAMETERS
set codedir=C:\programs\tartanair\sample_pipeline\src
set envdir=C:\tartanair-v2\Package
set posedir=C:\tartanair-v2\trajs
set outputdir=C:\TartanAir_v2

@REM @REM set gamma value
@REM cd /d %codedir%
@REM start /wait python misc\AirsimSetting.py --data-collection --gamma %gamma%  --max-exposure 0.7 --min-exposure 0.7

@REM cd /d %envdir%\%env%\
@REM start %env%.exe

@REM cd /d %outputdir%
@REM if not exist "%env%" mkdir %env%
@REM cd %env%

@REM @REM start the data collection
@REM cd /d %codedir%
@REM @REM start /wait python -m collection.collect_images --environment-dir %outputdir%\%env% --posefile-folder %posedir%\%env%\pose_easy --data-folder Data_easy --cam-list 0_1_2_3_4_5_6_7_8_9_10_11
@REM start /wait python -m collection.collect_images --environment-dir %outputdir%\%env% --posefile-folder %posedir%\%env%\pose_hard --data-folder Data_hard --cam-list 0_1_2_3_4_5_6_7_8_9_10_11
@REM TASKKILL /IM %env%.exe /F /T

@REM @REM data verification
@REM copy %seglabel% %outputdir%\%env%\seg_label.json
@REM start /wait python -m postprocessing.data_validation --data-root %outputdir% --env-folders %env% --data-folders Data_easy,Data_hard --analyze-depth --analyze-rgb  --analyze-seg --rgb-depth-filter --create-video --np 12

@REM PostProcessing
set processnum=12

@REM generate optical flow
cd /d %codedir%

@REM generate fisheye panorama
start /wait python -m postprocessing.fish_and_pano --data-root %outputdir% --env-folders %env% --data-folders Data_easy,Data_hard --modalities image,depth,seg --new-cam-models fish,equirect --np %processnum%

start /wait python -m postprocessing.flow_and_warping_error --data-root %outputdir% --env-folders %env% --data-folders Data_easy,Data_hard --np %processnum%

@REM generate imu
start /wait python -m postprocessing.imu_generator --data-root %outputdir% --env-folders %env% --data-folders Data_easy,Data_hard 

@REM generate lidar
start /wait python -m postprocessing.lidar --data-root %outputdir% --env-folders %env% --data-folders Data_easy,Data_hard --np %processnum%
