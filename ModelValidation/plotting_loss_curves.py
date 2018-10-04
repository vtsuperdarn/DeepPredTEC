import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../STResNet")
from params import Params as param

#param.saved_model_path = "./model_results/model_batch64_epoch100_resnet100_nresfltr6_nfltr12_of2_otec6_cf2_csl72_pf12_psl72_tf36_tsl8_gs32_ks55_exoT_nrmT_w0_yr_11_13_314.8617606163025"
param.saved_model_path = "./model_results/model_batch64_epoch100_resnet100_nresfltr24_nfltr12_of2_otec24_cf2_csl72_pf12_psl72_tf36_tsl8_gs32_ks55_exoT_nrmT_w2_yr_11_13_8.384427070617676"

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
