#Import all the required packages
import matplotlib
import tables
import click

import numpy as np
import matplotlib.pyplot as plt

@click.command()
@click.option('--trainDataName', default="trainData.npy", help='Name of train data file')
@click.option('--testDataName', default="testData.npy", help='Name of test data file')
@click.option('--networkName', default="genToReco", help='Name of network')
@click.option('--h5', default="genToReco.h5", help='Name of h5 file for data')
def main(traineataname, testdataname, networkname, h5):

    f = tables.open_file(h5, mode='r')
    
    
    medians = f.root.medianPrediction[:,:]
    weights = np.where(np.isnan(medians), 0, 1)
    medians = np.where(np.isnan(medians), 0, medians)
    
    batchSize=10000
    bins=50
    totalSamples = f.root.data.shape[0]
    ranges=[[170,310],[-5,5],[-3.2,3.2],[0,70]]
    ranges= [[20,100],[-5,5],[-3.2,3.2],[0,25]]
    allCounts=[]
    allEdges=[]
    quantCounts=[]
    quantEdges=[]
    
    totalCounts = 0
    for x in range(totalSamples//batchSize+1):
        predCounts=[]
        recoData = f.root.recoData[x*batchSize:min((x+1)*batchSize, totalSamples),:]
        predData = f.root.data[x*batchSize:min((x+1)*batchSize, totalSamples),:]
        genData = f.root.genData[x*batchSize:min((x+1)*batchSize, totalSamples),:]
        
        quants = f.root.genQuantile[x*batchSize:min((x+1)*batchSize, totalSamples),:]
        weights = np.ones(genData[:,0].shape)
        totalCounts+=np.sum(weights)
        for z in range(1):#numSamples):
            currentData = predData[:,4*z:4*(z+1)]
            temp = []
            for y in range(4):
                counts, _ = np.histogram(currentData[:,y], range=ranges[y], bins=bins, weights=weights)
                temp.append(counts)
            predCounts.append(temp)
        predCounts = np.array(predCounts)
        for y in range(4):
            recoCounts, edges = np.histogram(recoData[:,y], range=ranges[y], bins=bins, weights=weights)
            genCounts, _ = np.histogram(genData[:,y], range=ranges[y], bins=bins, weights=weights)
            edges = edges[1:]/2+edges[:-1]/2
            qCounts, qEdges = np.histogram(quants[:,y], range=(0,1), bins=bins, weights=weights)
            qEdges = qEdges[1:]
            if(x==0):
                allCounts.append([genCounts, predCounts[:,y,:], recoCounts])
                allEdges.append(edges)
                quantCounts.append([qCounts])
                quantEdges.append(qEdges)
            else:
                allCounts[y][0]+=genCounts
                allCounts[y][1]+=predCounts[:,y,:]
                allCounts[y][2]+=recoCounts
                quantCounts[y]+=qCounts
    
            
    
    print("Total counts", totalCounts)    
    print("Total samples", totalSamples)    
    
    
    font = {'family' : 'serif',
            'size'   : 10}
    matplotlib.rc('font', **font)
    matplotlib.rcParams.update({
        "text.usetex": True})
    labels=["$p_T$ (GeV)", "$\eta$", "$\phi$", "mass (GeV)"]
    titles=["pT", "eta", "phi", "mass"]
    for x in range(4):
        qSums = [quantCounts[x][0,0]]
        for y in range(1, len(qCounts)):
            qSums.append(qSums[-1]+quantCounts[x][0,y])
        fig, (ax1) = plt.subplots(1, 1, figsize=(3.5,2.5))#, gridspec_kw={'height_ratios': [0.5, 1]})
        ax1.scatter(qEdges, bins*quantCounts[x][0,:]/np.sum(quantCounts[x][0,:]), c="#d7301f", label="predicted", s=5)
        ax1.plot(qEdges, qEdges*0+1, c="k", linestyle="--", label="target")
        ax1.set_ylim((0.7,1.3))
        ax1.set_ylabel("p(quantile)")
        ax1.set_xlabel("quantile")
        ax1.set_title(labels[x])
        ax1.legend()
        plt.tight_layout()
        fig.subplots_adjust(wspace=0.0, hspace=0.1)
        fig.savefig("closure_"+titles[x].replace(" ","_")+"_g2r.pdf")
        fig.show()
        
    for x in range(4):
        
        recoSum = np.sum(allCounts[x][2])
        allCounts[x][2]=allCounts[x][2]/recoSum
        allCounts[x][1] = np.sum(allCounts[x][1], axis=0)
        predSum = np.sum(allCounts[x][1])
        allCounts[x][1]=allCounts[x][1]/predSum
        
    
    for x in range(4):      
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5*3/2.5,3.8), gridspec_kw={'height_ratios': [2,0.5]})
        
        ax1.step(allEdges[x], allCounts[x][2], where="mid", color="k", linewidth=0.5)#, linestyle="-.")
        ax1.step(allEdges[x], allCounts[x][1], where="mid", color="#d7301f", linewidth=0.5)
        ax1.scatter(allEdges[x], allCounts[x][1], label="predicted", color="#d7301f", marker="x", s=5, linewidth=0.5)
        ax1.scatter(allEdges[x], allCounts[x][2], label="reco",  color="k",facecolors='none', marker="o", s=5, linewidth=0.5)
        
        ax1.set_xlim(ranges[x])
        ax1.set_ylim(0, max(max(allCounts[x][2]),max(allCounts[x][1]))*1.1)
        ax1.set_ylabel("counts")
        ax1.set_xticklabels([])
        ax1.legend()
        
        ax2.scatter(allEdges[x], allCounts[x][2]/allCounts[x][2], color="k", marker="o",facecolors="none", s=5, linewidth=0.5)
        ax2.scatter(allEdges[x], allCounts[x][1]/allCounts[x][2], marker="x", s=5, linewidth=0.5, color="#d7301f")
        
        ax2.set_xlabel(labels[x])
        ax2.set_ylabel(r"$\frac{\textnormal{predicted}}{\textnormal{gen}}$")
        ax2.set_ylim((0.9,1.1))
        ax2.set_xlim(ranges[x])
        plt.tight_layout()
        fig.savefig("all_"+titles[x].replace(" ","_")+"_g2r.pdf")
        fig.subplots_adjust(wspace=0.0, hspace=0.1)
        fig.show()
        
    f.close()




if __name__ == '__main__':
    main()
