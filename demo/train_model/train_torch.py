import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

batch_size = 2048
num_neurons = 10
class ImageDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = torch.tensor(self.data[index], dtype=torch.float)
        label = torch.tensor(self.labels[index,0:num_neurons], dtype=torch.float)
        return img, label

class net_one_neuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=30,kernel_size=(5,5),stride=(1,1)),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
            nn.Dropout2d(0.3),
            nn.Conv2d(in_channels=30,out_channels=30,kernel_size=(5,5),stride=(1,1)),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
            nn.Dropout2d(0.3),
            nn.Conv2d(in_channels=30,out_channels=30,kernel_size=(3,3),stride=(1,1)),
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
            nn.Dropout2d(0.3),
            nn.Conv2d(in_channels=30,out_channels=30,kernel_size=(3,3),stride=(1,1)),
            nn.BatchNorm2d(30),
            nn.Sigmoid(),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(5*5*30,1)
    def forward(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

class seperate_core_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.models = nn.ModuleList([net_one_neuron() for i in range(num_neurons)])
    def forward(self, x):
        outputs = [self.models[i].forward(x) for i in range(num_neurons)]
        outputs = torch.stack(outputs, dim = 1)
        return outputs.reshape((outputs.shape[0],outputs.shape[1]))

if __name__ == "__main__":

    y_all_train = np.load('../Rsp.npy')
    y_all_val = np.load('../valRsp.npy')

    x_train = np.load('../train_x.npy')
    x_val = np.load('../val_x.npy')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = nn.ModuleList([net_one_neuron() for i in range(num_neurons)])
    #net = nn.DataParallel(net)
    net.to(device)
    Imageset = ImageDataset(x_train, y_all_train)
    loader = DataLoader(Imageset, batch_size=batch_size, shuffle=True)
    valset = ImageDataset(x_val, y_all_val)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    #optimizer = torch.optim.Adam(net.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-08)
    optimizers = [torch.optim.Adam(sub_net.parameters(), lr=0.02, betas=(0.9, 0.999), eps=1e-08) for sub_net in net]
    criterion = torch.nn.MSELoss(reduction='none')
    #optimizer.load_state_dict(torch.load('optimizer_model'))
    #net.load_state_dict(torch.load('model_result'))
    num_epochs = 500
    all_loss = []
    all_corr = []
    for epoch in tqdm(range(num_epochs)):
        torch.save(net.state_dict(), "model_result_seperated_deep")
        # Train
        for subnet in net:
            subnet.train()
        avg_loss = 0.0
        for batch_num, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            for i, (subnet,optimizer) in enumerate(zip(net,optimizers)):
                optimizer.zero_grad()
                output = subnet(x)
                output = torch.reshape(output,(output.shape[0],))
                y_neuron = y[:,i]
                y_exp = torch.exp(y_neuron)
                loss = criterion(output,y_neuron)
                loss_w = torch.mean(loss*y_exp)
                loss_w.backward()
                optimizer.step()
                avg_loss += loss_w.item()
            avg_loss /= num_neurons

        # Validate
        torch.cuda.empty_cache()
        with torch.no_grad():
            net.eval()
            num_correct = 0
            test_loss = 0
            prediction = []
            actual = []
            for batch_num, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                outputs = torch.stack([subnet(x) for subnet in net] )
                outputs = torch.reshape(outputs, (outputs.shape[0], outputs.shape[1]))
                outputs = torch.transpose(outputs,0,1)
                y_exp = torch.exp(y)
                loss = criterion(outputs, y)
                loss_w = torch.mean(loss * y_exp)
                test_loss += loss_w.item()
                prediction.extend(outputs.cpu().numpy())
                actual.extend(y.cpu().numpy())
            test_loss /= (len(valset) / 128)
            # scheduler.step(test_loss)
            torch.cuda.empty_cache()
            all_loss.append(test_loss)

            prediction = np.stack(prediction)
            actual = np.stack(actual)

            R = np.zeros(num_neurons)
            VE = np.zeros(num_neurons)
            for neuron in range(num_neurons):
                pred1 = prediction[:, neuron]
                val_y = actual[:, neuron]
                y_arg = np.argsort(val_y)

                u2 = np.zeros((2, 1000))
                u2[0, :] = np.reshape(pred1, (1000))
                u2[1, :] = np.reshape(val_y, (1000))

                c2 = np.corrcoef(u2)
                R[neuron] = c2[0, 1]

                VE[neuron] = 1 - np.var(pred1 - val_y) / np.var(val_y)
            all_corr.append(R)
            print('Epoch: {}, test loss: {}, corr: {}'.format(epoch, test_loss, np.average(R)))
    np.save("all_corr_change_deep", np.stack(all_corr))
    np.save("loss_data_deep",np.stack(all_loss))

