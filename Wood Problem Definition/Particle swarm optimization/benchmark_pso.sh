#!/bin/bash
for i in {1..10}
do
	echo "Running experiment no$i"
	python main_PSO.py
	echo "End of experiment no$i"
done
