/**
 * This program clusters partons in dataset into jets, then runs jet matching
 * between the parton, generated, and reco level jets. It records the 4-vectors
 * of the matched jets
**/
#include "ExRootAnalysis/ExRootTreeReader.h"
#include "fastjet/ClusterSequence.hh"
#include "helpers.h"

using namespace fastjet;

void analyzeDelphes(char const *charInputFile, char const *argv2)
{
  
  std::string outputFile(argv2);
  // Save data to the processed directory
  std::string fullOutput = "processed/" + outputFile + "_seed_data.txt";

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
    
    // Get partons
    int partonCount = 0;
    std::vector<PseudoJet> partonJetParticles;
    for(Int_t particleEntry = 0; particleEntry < numberOfParticles; ++particleEntry){
    	GenParticle *particle = (GenParticle*) branchParticle->At(particleEntry);
      	if(particle->Status==71){
      	    partonCount++;
      	    partonJetParticles.push_back(PseudoJet(particle->Px, particle->Py, particle->Pz, particle->E));
        }
      	
    }
    
    double R = 0.4;
    JetDefinition jet_def(antikt_algorithm, R);
    ClusterSequence cs(partonJetParticles, jet_def);

    std::vector<PseudoJet> partonJets = cs.inclusive_jets(); // get new parton jets
    
    int numPartonJetsRecoMatch = 0;
    int numPartonJetsGenMatch = 0;
    int numPartonJetsBothMatch = 0;
    int numPartonJets = 0;
    
    if (partonJets.size() > 0)
        {
            
            for (size_t j = 0; j < partonJets.size(); j++)
            {
                
                if (partonJets[j].pt() >= 20.0)
                {
                    float partonPt = partonJets[j].pt();
                    float partonEta = partonJets[j].rap();
                    float partonPhi = partonJets[j].phi_std();
                    float partonMass = partonJets[j].m();
                    
                    float genPt = -1000;
                    float genEta = -1000;
                    float genPhi = -1000;
                    float genMass = -1000;
                    
                    float recoPt = -1000;
                    float recoEta = -1000;
                    float recoPhi = -1000;
                    float recoMass = -1000;
                    float recoFlavor = -1000;
                    numPartonJets++;

                    int genJetMatches = 0;
                    float minDR = 10.0;
                    // Match parton jets to gen jets via dR
                    for (size_t k = 0; k < numberOfGenJets; k++)
                    {
                        Jet *genJet = (Jet*) branchGenJet->At(k);
                        
                        if (genJet->PT >= 0.0)
                        {
                            float dR = deltaR(genJet->Eta, genJet->Phi, partonJets[j].rap(), partonJets[j].phi_std());
                            if (dR < 0.35)
                                genJetMatches++;
                            if (dR < minDR)
                                minDR = dR;
                                
                                genPt = genJet->PT;
				 genEta = genJet->Eta;
				 genPhi = genJet->Phi;
				 genMass = genJet->Mass;
			   }
                    }
                    if (genJetMatches == 1){
                        numPartonJetsGenMatch++;
                   }
                    int pfJetMatches = 0;
                    minDR = 10.0;
                    // Match parton jets to reco jets via dR
                    for (size_t k = 0; k < numberOfRecoJets; k++)
                    {
                        Jet *recoJet = (Jet*) branchRecoJet->At(k);
                        if (recoJet->PT >= 0.0)
                        {
                            float dR = deltaR(recoJet->Eta, recoJet->Phi, partonJets[j].rap(), partonJets[j].phi_std());
                            if (dR < 0.35){
                                pfJetMatches++;
                            }
                            if (dR < minDR){
                                minDR = dR;
                                
                                recoPt = recoJet->PT;
				 recoEta = recoJet->Eta;
				 recoPhi = recoJet->Phi;
				 recoMass = recoJet->Mass;
				 recoFlavor = recoJet->Flavor;
                            }
                        }
                    }
                    // If there are matches to both reco and gen record the data
                    if (pfJetMatches == 1){
                        numPartonJetsRecoMatch++;
                        if (genJetMatches == 1){
                             numPartonJetsBothMatch++;
                             write_out_jets << partonPt << " " << partonEta << " " <<  partonPhi << " " <<  partonMass << " "; 
                             write_out_jets << genPt << " " << genEta << " " <<  genPhi << " " <<  genMass<< " "; 
                             write_out_jets << recoPt << " " << recoEta << " " <<  recoPhi << " " <<  recoMass<< " ";
                             write_out_jets << recoFlavor <<  std::endl;
                        }
                   }
  
            }
        }
    }
    numPartonJetsRecoMatchTotal += numPartonJetsRecoMatch;
    numPartonJetsGenMatchTotal += numPartonJetsGenMatch;
    numPartonJetsBothMatchTotal += numPartonJetsBothMatch;
    numPartonJetsTotal+= numPartonJets;
  
  }
  
  write_out_jets.close();

}
