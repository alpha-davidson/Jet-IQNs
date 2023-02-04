#include "ExRootAnalysis/ExRootTreeReader.h"
#include "helpers.h"

void analyzeDelphes(char const *charInputFile, char const *argv2) {
  
    std::string outputFile(argv2);
    std::string fullOutput = "/home/brkronheim/Delphes-3.5.0/data/processed/" + outputFile + "_seed_data.txt";

    // Create chain of root trees

    std::ofstream write_out_jets(fullOutput);

    TChain chain("Delphes");
    chain.Add(charInputFile);

    // Create object of class ExRootTreeReader
    ExRootTreeReader *treeReader = new ExRootTreeReader(&chain);
    Long64_t numberOfEntries = treeReader->GetEntries();

    // Get pointers to branches used in this analysis
    TClonesArray *branchParticle = treeReader->UseBranch("Particle");
    TClonesArray *branchGenJet = treeReader->UseBranch("GenJet");
    TClonesArray *branchRecoJet = treeReader->UseBranch("Jet");

    //numberOfEntries = 100;
    // Loop over all events
    Long64_t totalJets = 0;
    int numPartonJetsRecoMatchTotal = 0;
    int numPartonJetsGenMatchTotal = 0;
    int numPartonJetsBothMatchTotal = 0;
    int numPartonJetsTotal = 0;
    for(Int_t entry = 0; entry < numberOfEntries; ++entry)
    {
        if((entry+1)%10000==0){
            std::cout << entry+1 << '/' << numberOfEntries << std::endl;
        }
        // Load selected branches with data from specified event
        treeReader->ReadEntry(entry);

        // If event contains at least 1 jet
        Long64_t numberOfParticles = branchParticle->GetEntries();
        Long64_t numberOfGenJets = branchGenJet->GetEntries();
        Long64_t numberOfRecoJets = branchRecoJet->GetEntries();
        for (size_t kr = 0; kr < numberOfRecoJets; kr++){
            Jet *recoJet = (Jet*) branchRecoJet->At(kr);
            if (recoJet->PT >= 0.0){
                bool matched = false;
                float recoPt = recoJet->PT;
                float recoEta = recoJet->Eta;
                float recoPhi = recoJet->Phi;
                float recoMass = recoJet->Mass;
                float recoFlavor = recoJet->Flavor;
                float genPt = -1000;
                float genEta = -1000;
                float genPhi = -1000;
                float genMass = -1000;

                float minDR = 10.0;
                for (size_t k = 0; k < numberOfGenJets; k++){
                    Jet *genJet = (Jet*) branchGenJet->At(k);
                    
                    if (genJet->PT >= 0.0) {
                        float dR = deltaR(genJet->Eta, genJet->Phi, recoEta, recoPhi);
                        if (dR < 0.35) {
                            matched = true;
                        }
                        if (dR < minDR){
                            minDR = dR;
                            genPt = genJet->PT;
                            genEta = genJet->Eta;
                            genPhi = genJet->Phi;
                            genMass = genJet->Mass;
                        }
                    }       
             
                    if (matched){
                         write_out_jets << genPt << " " << genEta << " " <<  genPhi << " " <<  genMass<< " "; 
                         write_out_jets << recoPt << " " << recoEta << " " <<  recoPhi << " " <<  recoMass<< " ";
                         write_out_jets << recoFlavor <<  std::endl;
                    }
                }
            }
        }
    }
    write_out_jets.close();
}
