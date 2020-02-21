#!/bin/bash
cd ..
for i in {1..10}
do
	echo "Running experiment no$i"
	python main_minimizers.py "L-BFGS-B"
	echo "End of experiment no$i"
done
