# from collections import OrderedDict
# import pandas as pd
# import numpy as np
# from BLP import Model
      
# def writeResults(filename, models):
#     writeDf = pd.concat([models[0].framer.prior] + [m.df for m in models])
#     writeDf.to_csv(filename, header=False)
    
# assetInfo = {'US Equity': 0.5, 'Foreign EQ': 0.4, 'Emerging EQ': 0.1}
# assetClasses = list(assetInfo.keys())
# assetWeights = list(assetInfo.values())  

# data = pd.read_csv('example_returndata.csv', usecols=assetClasses)
# covMatrix = data.cov()

# print('Computing models...')

# model_one = Model.fromPriorData(
#     assetClasses, 
#     assetWeights, 
#     riskAversion=3, 
#     covMatrix=covMatrix,
#     tau=0.1,
#     tauv=0.1, 
#     P=np.asarray([[1,0,0], [0,1,-1]]),
#     Q=np.asarray([[0.015],[0.03]]),
#     identifier=1
# )
# model_two = Model.fromPriorData(
#     assetClasses, 
#     assetWeights, 
#     riskAversion=3, 
#     covMatrix=covMatrix,
#     tau=0.1,
#     tauv=0.01, 
#     P=np.asarray([[1,0,0], [0,1,-1]]),
#     Q=np.asarray([[0.015],[0.03]]),
#     identifier=2
# )
# model_three = Model.fromPriorData(
#     assetClasses, 
#     assetWeights, 
#     riskAversion=3, 
#     covMatrix=covMatrix,
#     tau=0.1,
#     tauv=0.01, 
#     P=np.asarray([[1,-1,0], [0,0,1]]),
#     Q=np.asarray([[0.02],[0.015]]),
#     identifier=3
# )
# models = (model_one, model_two, model_three)
# outFile = 'new_example_output.csv'
# writeResults(outFile, models)

# print('Done.')
# print(f'Check the model results in { outFile }')

from collections import OrderedDict
import pandas as pd
import numpy as np
from BLP import Model
      
def writeResults(filename, models):
    writeDf = pd.concat([models[0].framer.prior] + [m.df for m in models])
    writeDf.to_csv(filename, header=False)

# Read the CSV file into a DataFrame
data = pd.read_csv('AA_indexes.csv', sep=';', decimal=',', thousands=" ")

# Set 'Data' as index of the dataframe
data.set_index('Data', inplace=True)
data.index = pd.to_datetime(data.index)
data = data.sort_index()

assetClasses = ['IMA-S', 'IMA-B', 'IRF-M', 'S&P U.S. Treasury Bond Current 10-Year', 'IHFA', 'Ibovespa', 'S&P 500']
assetWeights = [1/len(assetClasses) for _ in assetClasses] # Just to test

data = data[assetClasses]
data = data.pct_change().dropna()
covMatrix = data.cov()

print('Computing models...')

model_one = Model.fromPriorData(
    assetClasses, 
    assetWeights, 
    riskAversion=3,         
    covMatrix=covMatrix,
    tau=0.1,
    tauv=0.1, 
    P=np.asarray([[1,0,0,0,0,0,0],]),
    Q=np.asarray([[1],]),
    identifier=1
)
models = (model_one,)
outFile = 'new_example_output.csv'
writeResults(outFile, models)

print('Done.')
print(f'Check the model results in { outFile }')
