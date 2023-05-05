@echo off
@REM CHANGE THE FOLLOWING PARAMETERS
set env=AbandonedSchool

@REM DOUBLE CHECK THE FOLLOWING PARAMETERS
set codedir=C:\programs\tartanair\sample_pipeline\src
set outputdir_root=C:\tartanair-v2\data
set processnum=12

@REM generate optical flow
cd /d %codedir%
start /wait python -m postprocessing.flow_and_warping_error --data-root %outputdir_root% --env-folders %env% --data-folders Data_easy,Data_hard --np %processnum%

@REM generate imu
start /wait python -m postprocessing.imu_generator --data-root %outputdir_root% --env-folders %env% --data-folders Data_easy,Data_hard 

@REM generate fisheye panorama
start /wait python -m postprocessing.fish_and_pano --data-root %outputdir_root% --env-folders %env% --data-folders Data_easy,Data_hard --modalities image,depth,seg --new-cam-models fish,equirect --np %processnum%

@REM generate lidar
start /wait python -m postprocessing.lidar --data-root %outputdir_root% --env-folders %env% --data-folders Data_easy,Data_hard --np %processnum%
