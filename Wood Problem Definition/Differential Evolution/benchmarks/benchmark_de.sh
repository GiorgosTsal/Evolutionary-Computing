#!/bin/bash
cd ..
for i in {1..10}
do
	echo "Running experiment no$i"
	python main_DE.py
	echo "End of experiment no$i"
done
