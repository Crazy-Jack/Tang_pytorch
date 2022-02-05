from train_torch import ImageDataset
from train_torch import net_one_neuron
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

nb_validation_samples = 1000
y_all_train = np.load('../Rsp.npy')
y_all_val = np.load('../valRsp.npy')
batch_size = 80
num_neurons = 10
x_train = np.load('../train_x.npy')
x_val = np.load('../val_x.npy')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = nn.ModuleList([net_one_neuron() for i in range(num_neurons)])

net.to(device)
Imageset = ImageDataset(x_train, y_all_train)
loader = DataLoader(Imageset, batch_size=batch_size, shuffle=True)
valset = ImageDataset(x_val, y_all_val)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

net.load_state_dict(torch.load('model_result_seperated_w'))

with torch.no_grad():
    net.eval()
    prediction = []
    actual = []
    for batch_num, (x, y) in enumerate(tqdm(val_loader)):

        x, y = x.to(device), y.to(device)
        outputs = np.stack([subnet(x).cpu().numpy() for subnet in net])
        prediction.extend(np.transpose(outputs,(1,0,2)))
        actual.extend(y.cpu().numpy())

    prediction = np.stack(prediction).reshape(nb_validation_samples,num_neurons)
    actual = np.stack(actual)
    #np.save('prediction',prediction)
    #np.save('actual',actual)
    R = np.zeros(299)
    VE = np.zeros(299)
    for neuron in range(num_neurons):
        pred1 = prediction[:, neuron]
        val_y = actual[:, neuron]
        y_arg = np.argsort(val_y)

        u2 = np.zeros((2, nb_validation_samples))
        u2[0, :] = np.reshape(pred1, (nb_validation_samples))
        u2[1, :] = np.reshape(val_y, (nb_validation_samples))

        c2 = np.corrcoef(u2)
        R[neuron] = c2[0, 1]

        VE[neuron] = 1 - np.var(pred1 - val_y) / np.var(val_y)

        plt.plot(pred1[y_arg],label='pred')
        plt.plot(np.sort(val_y),label='actual')
        plt.title(R[neuron])
        plt.legend()
        #plt.savefig('pics/'+str(neuron)+'tuningcurve.png')
        plt.show()



    print(R)
    print(VE)
    np.save('corr_val_w', R)

# corr_change = np.load('all_corr_change_w.npy')
# loss_change = np.load('loss_data_w.npy')
# plt.plot(corr_change)
# plt.savefig('pics1/corr_change')
# plt.show()
# plt.plot(np.average(corr_change,axis=1))
# plt.savefig('pics1/corr_avg_change')
# plt.show()
# plt.plot(loss_change)
# plt.savefig('pics1/loss_change')
# plt.show()
