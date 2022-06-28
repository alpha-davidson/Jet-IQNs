# Generate the 1000 Pythia files for data generation

for seed in range(1,1001):
    file=["Main:numberOfEvents   = 10000", "Main:timesAllowErrors =  1000",
        "Init:showChangedSettings = on",
        "Init:showAllSettings = off",
        "Init:showChangedParticleData = on",
        "Init:showAllParticleData = off",
        "Next:numberCount = 1000",
        "Next:numberShowLHA = 1",
        "Next:numberShowInfo = 1",
        "Next:numberShowProcess = 10",
        "Next:numberShowEvent = 1",
        "Stat:showPartonLevel = on",
        "Beams:idA = 2212",
        "Beams:idB = 2212",
        "Beams:eCM = 13000.",
        "HardQCD:all		= on",
        "PhaseSpace:pTHatMin	= 20",
        "PartonLevel:ISR = on",
        "PartonLevel:FSR = on",
        "PartonLevel:MPI = on",
        "HadronLevel:all = on",
        "Random::setSeed = on",
        "Random:seed     = "+str(seed)]
    
    f = open("pythiaFiles/qcd_seed_"+str(seed)+".txt", "a")
    for line in file:
        f.write(line+"\n")
    f.close()


