import torch
import torch.nn as nn
from tqdm import tqdm
from train_torch import net_one_neuron
from train_torch import ImageDataset
import numpy as np
from PIL import Image

class image_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_opt = torch.rand(50,50)
        self.img = nn.Parameter(self.img_opt.reshape(1,1,50,50))

batch_size = 128
epoch = 1000
num_neurons = 10
x_val = np.load('../val_x.npy')
y_all_val = np.load('../valRsp.npy')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
corr = np.load('corr_val_w.npy')

net = nn.ModuleList([net_one_neuron() for i in range(num_neurons)])
net.to(device)
net.load_state_dict(torch.load('model_result_seperated_w'))
net.eval()

img_object = image_model()
img_object.to(device)
optimizers = [torch.optim.Adam(img_object.parameters(), lr=1e-1) for sub_net in net]
schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, threshold=0.0001, verbose=True) for optimizer in optimizers]
valset = ImageDataset(x_val, y_all_val)
criterion =  torch.nn.MSELoss()

target_img, target_response = valset.__getitem__(0)

target_response = target_response.to(device)

for epoch in tqdm(range(epoch)):
    for i, (subnet, optimizer, scheduler) in enumerate(zip(net, optimizers, schedulers)):
        outputs = subnet(img_object.img)
        loss = criterion(outputs,target_response[i])*corr[i]
        loss.backward()
        optimizer.step()
        print(loss.item())
        scheduler.step(loss)

img_result = img_object.img.data.cpu().numpy()
img_result = np.reshape(img_result,(50,50))
np.save('img_result',img_result)
img_result = np.load('img_result.npy')
img = Image.fromarray(np.uint8(img_result * 255) , 'L')
img.show()

target_img = np.reshape(target_img,(50,50))
img_original = Image.fromarray(np.uint8(target_img * 255) , 'L')
img_original.show()


print(img_result)