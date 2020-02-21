#!/bin/bash
for i in {1..10}
do
	echo "Running experiment no$i"
	python main_minimizers.py "SLSQP"
	echo "End of experiment no$i"
done
