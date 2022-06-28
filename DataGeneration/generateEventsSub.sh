#!/bin/bash
startSeed=$1
for i in {1..125}
do
    	currentSeed=$((${startSeed} + ${i}))
    	./DelphesPythia8 cards/delphes_card_CMS.tcl pythiaFiles/qcd_seed_${currentSeed}.txt qcd${currentSeed}_CMS_10000events.root
    	root <<-EOF
	gSystem->AddIncludePath("-I/home/.../fastjet-3.4.0/fastjet-install/include") # add the correct location
	gSystem->Load("/usr/local/lib/libfastjet.so") # add the correct location
    	gSystem->Load("/home/.../Delphes-3.5.0/libDelphes") # add the correct location
    	.L analyzeDelphes.cc
    	analyzeDelphes("qcd${currentSeed}_CMS_10000events.root", "${currentSeed}")
	EOF
	rm qcd${currentSeed}_CMS_10000events.root
done

