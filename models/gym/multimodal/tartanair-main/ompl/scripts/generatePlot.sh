#!/bin/bash

python scripts/ompl_benchmark_statistics.py $1 -d temp.db
python scripts/ompl_benchmark_statistics.py -d temp.db -p $2
rm -rf temp.db
rm -rf *.console
rm -rf $1