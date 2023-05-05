MAPNAME=tiny_house
ENVDIR=/home/sakura/tmp/maps/${MAPNAME}
SCRIPT=../src/sample_pipeline.py

# step 0 mapping
python $SCRIPT \
--mapping \
--map-dir ${ENVDIR} \
--map-filename ${MAPNAME} \
--path-skip 10

# # step 1
# python $SCRIPT \
# --sample-graph \
# --environment-dir $ENVDIR \
# --node-num 80 \
# --edge-num 10 \
# --min-dist-thresh 10 \
# --max-dist-thresh 30 \
# --node-range-xmin -20.0 \
# --node-range-xmax 20.0 \
# --node-range-ymin -20.0 \
# --node-range-ymax 20.0 \
# --node-range-zmin -4 \
# --node-range-zmax 2 

# # step 2
# python $SCRIPT \
# --sample-path \
# --graph-filename node80_edge10_len2_8.graph \
# --environment-dir $ENVDIR \
# --sample-cycle-mode 0 \
# --ros-path-dir ros_path_v3 \
# --cycle-min-nodes 5
# # --interactive \

# # step 3 - easy
# python $SCRIPT \
# --sample-position \
# --ros-path-dir ros_path_v3 \
# --environment-dir $ENVDIR \
# --position-path-suffix easy \
# --dist-max 0.2 \
# --dist-min 0.0 \
# --acc-max 0.1 \
# --step-max 20 

# # step 3 - mid
# python $SCRIPT \
# --sample-position \
# --ros-path-dir ros_path \
# --environment-dir $ENVDIR \
# --position-path-suffix mid \
# --dist-max 0.3 \
# --dist-min 0.0 \
# --acc-max 0.1 \
# --step-max 20 

# # step 3 - hard
# python $SCRIPT \
# --sample-position \
# --ros-path-dir ros_path \
# --environment-dir $ENVDIR \
# --position-path-suffix hard \
# --dist-max 0.5 \
# --dist-min 0.0 \
# --acc-max 0.1 \
# --step-max 20 

# python $SCRIPT \
# --data-collection \
# --environment-dir $ENVDIR \
# --position-folder position_easy \
# --data-folder-perfix easy \
# --rand-degree 60 \
# --smooth-count 20 \
# --max-pitch 0 \
# --min-pitch 0 \
# --max-roll 0 \
# --min-roll 0 \
# --img-type "Scene" 

# python $SCRIPT \
# --data-collection \
# --environment-dir $ENVDIR \
# --position-folder position_mid \
# --data-folder-perfix mid \
# --rand-degree 50 \
# --smooth-count 10 \
# --max-pitch 30 \
# --min-pitch -45 \
# --max-roll 20 \
# --min-roll -20 \
# --img-type "Scene" 

# python $SCRIPT \
# --data-collection \
# --environment-dir $ENVDIR \
# --position-folder position_hard \
# --data-folder-perfix hard \
# --rand-degree 50 \
# --smooth-count 5 \
# --max-pitch 30 \
# --min-pitch -45 \
# --max-roll 20 \
# --min-roll -20 \
# --img-type "Scene" 