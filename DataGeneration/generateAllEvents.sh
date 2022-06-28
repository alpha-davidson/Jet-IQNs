#!/bin/bash
for i in 0 125 250 375 500 625 750 875
do
	./generateEventsSub.sh $i > /dev/null &
done
