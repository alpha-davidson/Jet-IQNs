import numpy as np

def extract(trainFile, validationFile, singleTarget=-1):
    """
    A helper function to process two input file containing the training and 
    validaiton data (or training and test data) and return an input dataset
    and an output dataset applying the log scaling, ratio, and normalization.

    Parameters
    ----------
    trainFile : string
        File name ending in .npy of the training data file
    validationFile : string
        File name ending in .npy of the validaiton (or test) data file
    singleTarget : int
        Indicates which dataset desired. 0 is pt, 1 is eta,
        2 is phi, 3 is mass, and anything else is all 4

    Returns
    -------
    None.

    """

    # Load data
    data = np.load(trainFile).T
    genData = data[0:4,:10000]
    recoData = data[4:8,:10000]

    # Log scale pt, mass
    # 0: pt, 1: eta, 2: phi, 3: mass
    genData[0,:] = np.log(genData[0,:])
    genData[3,:] = np.log(genData[3,:]+2)
    recoData[0,:] = np.log(recoData[0,:])
    recoData[3,:] = np.log(recoData[3,:]+2)

    inputData = genData # Input data is log scaled gen data
    # Output is a ratio of the reco to the gen
    outputData=(recoData+10)/(inputData[:4,:]+10)

    # Transpose to get proper shape
    trainIn = inputData.T
    trainOut = outputData.T

    # Repeat for the validation data
    data = np.load(validationFile).T
    genData = data[0:4,:10000]
    recoData = data[4:8,:10000]
    genData[0,:] = np.log(genData[0,:])
    genData[3,:] = np.log(genData[3,:]+2)
    recoData[0,:] = np.log(recoData[0,:])
    recoData[3,:] = np.log(recoData[3,:]+2)

    inputData = genData

    outputData=(recoData+10)/(inputData[:4,:]+10)

    valIn = inputData.T
    valOut = outputData.T
    
    
    # Normalize the data and save the normalization info.
    normInfoIn = [[0,1],[0,1],[0,1],[0,1]]
    normInfoOut = [[0,1],[0,1],[0,1],[0,1]]

    for x in range(4):
        normInfoIn[x] = [np.mean(trainIn[:,x]), np.std(trainIn[:,x])]
        valIn[:, x] = (valIn[:,x]-np.mean(trainIn[:,x]))/(np.std(trainIn[:,x]))
        trainIn[:, x] = (trainIn[:,x]-np.mean(trainIn[:,x]))/(np.std(trainIn[:,x]))
        
        normInfoOut[x] = [np.mean(trainOut[:,x]), np.std(trainOut[:,x])]
        valOut[:, x] = (valOut[:,x]-np.mean(trainOut[:,x]))/(np.std(trainOut[:,x]))
        trainOut[:, x] = (trainOut[:,x]-np.mean(trainOut[:,x]))/(np.std(trainOut[:,x]))
    
    # Adjust the dataset for single variables. Essentially, targets for earlier
    # variables become part of the input dataset and there is only ever 1 output
    if(singleTarget in [0,1,2,3]):
        
        trainIn = np.concatenate([trainIn, trainOut[:,0:singleTarget]], axis=1)
        trainOut = trainOut[:,singleTarget:singleTarget+1]
        
        valIn = np.concatenate([valIn, valIn[:,0:singleTarget]], axis=1)
        valOut = valOut[:,singleTarget:singleTarget+1]
        
        normInfoIn = normInfoIn + normInfoOut[:singleTarget]
        normInfoOut = normInfoOut[singleTarget:singleTarget+1]
        
    return(trainIn, trainOut, valIn, valOut, normInfoIn, normInfoOut)