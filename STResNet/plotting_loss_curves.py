import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from params import Params as param

#print ("Enter the model_path_values name:")
#model_path = raw_input()

model_path = param.saved_model_path + "_values"

train_loss = np.load(model_path+'/training_loss.npy').tolist()
val_loss = np.load(model_path+'/validation_loss.npy').tolist()

print (train_loss)
print (val_loss)

sns.set_style("whitegrid")
sns.set_context("poster")
f, axArr = plt.subplots(1, figsize=(20, 20))

x_val = range(0, len(train_loss))

cl = sns.color_palette('bright', 4)
axArr.plot(x_val, train_loss, color=cl[0], label='Train Loss')
axArr.plot(x_val, val_loss, color=cl[1], label='Validation Loss')

axArr.set_xlabel("Epoch", fontsize=14)
axArr.set_ylabel("Loss", fontsize=14)

axArr.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=1, ncol=2, borderaxespad=0.1 )
f.savefig(model_path+'/loss_curve', dpi=f.dpi, bbox_inches='tight')

#TODO: Uncomment the below section if bias-variance curve has to be plotted
'''
print("Enter the model_paths_values list")
model_paths = []

#loading the train_loss and val_loss values for the given model_paths
train_loss = []
val_loss = []
for mp in model_paths:
   tloss = np.load(mp+'/training_loss.npy').tolist() 
   vloss = np.load(mp+'/validation_loss.npy').tolist()
   train_loss.append(sum(tloss)/float(len(tloss)))
   val_loss.append(sum(vloss)/float(len(vloss)))

print("Enter the training size values list")
#TODO change the below one appropriately
x_val = range(0, len(train_loss))

cl = sns.color_palette('bright', 4)
axArr.plot(x_val, train_loss, color=cl[0], label='Avg Train Loss')
axArr.plot(x_val, val_loss, color=cl[1], label='Avg Validation Loss')

axArr.set_xlabel("Training Size", fontsize=14)
axArr.set_ylabel("Average Loss", fontsize=14)

axArr.legend( bbox_to_anchor=(0., 1.02, 1., .102), loc=1, ncol=2, borderaxespad=0.1 )
f.savefig('/bias_variance', dpi=f.dpi, bbox_inches='tight')    
'''
