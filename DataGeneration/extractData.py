# Make datasets from the data extracted via the C++ code.
import numpy as np
allData = []
for seed in range(1,1001):
    allData.append(np.loadtxt(str(seed)+"_seed_data.txt"))
data = np.concatenate(allData, axis=0)
print(data.shape)
np.save("testData.npy", data[:1000000,:])
np.save("validationData.npy", data[1000000:2000000,:])
np.save("trainData.npy", data[2000000:10000000,:])
