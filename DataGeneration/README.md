# Data Generation

The code contained here is used to generate the dataset used for paper. This requires an installation of Delphes, FastJet, Pythia, and Root. To use it first adjust generateEventsSub.sh to have the appropriate paths. Next, make the PythiaFiles directory and run writepythiaFile.py as follows:
```
mkdir pythiaFiles
python writePythiaFile.py
```
Then, make the two bash scripts exectuable and run as follows
```
chmod +x generateEventsSub.sh
chmod +x generateAllEvents.sh
./generateAllEvents
```
This code is designed to spawn 8 parallel proccesses in the background which will generate the events in batches. On an 8 core AMD 4700U laptop this took a little under 8 hours to run while nothing else was running. In the end, there will be 1000 files containing 9.4 million sets of jets. The root files are not kept to preserve space. To compact these into one file run
```
python extractData.py
```
This script needs to be in the folder with the generated text files.

The C++ files for this are adpated from https://github.com/alpha-davidson/falcon-cWGAN.
