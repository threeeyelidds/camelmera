#!/bin/bash

for (( i = 1; i < 6; i++ )); do
    rosrun ompl random_corridors_benchmark _num:=$i
    ./scripts/generatePlot.sh log/random_corridors_$i.log plot/random_corridors_$i.pdf
done