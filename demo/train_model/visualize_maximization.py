import torch
import torch.nn as nn
from tqdm import tqdm
from train_torch import net_one_neuron
from train_torch import ImageDataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class image_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_opt = torch.rand(50, 50)
        self.img = nn.Parameter(self.img_opt.reshape(1, 1, 50, 50))


batch_size = 128
epoch = 1000
num_neurons = 10
target_intensity = 6
x_val = np.load('../val_x.npy')
y_all_val = np.load('../valRsp.npy')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
corr = np.load('corr_val_w.npy')

net = nn.ModuleList([net_one_neuron() for i in range(num_neurons)])
net.to(device)
net.load_state_dict(torch.load('model_result_seperated_w'))
net.eval()

img_objects = torch.nn.ModuleList([image_model() for i in range(num_neurons)])
img_objects.to(device)
optimizers = [torch.optim.Adam(img_object.parameters(), lr=1e-4) for img_object in img_objects]
# schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, threshold=0.0001, verbose=True) for optimizer in optimizers]
valset = ImageDataset(x_val, y_all_val)
criterion = torch.nn.MSELoss()

losses = []

subnet = net[4]
optimizer = optimizers[4]
# scheduler = schedulers[4]
img_object = img_objects[4]

for epoch in tqdm(range(epoch)):
    # i, (subnet, optimizer, scheduler, img_object) in enumerate(zip(net, optimizers, schedulers, img_objects)):
    outputs = subnet(img_object.img)
    tmp1 = img_object.img[:, 1:] - img_object.img[:, :50 - 1]
    lap_diff1 = criterion(img_object.img[:, :, :, 1:], img_object.img[:, :, :, :50 - 1]) \
                + criterion(img_object.img[:, :, 1:, :], img_object.img[:, :, :50 - 1, :])
    loss = (torch.abs(outputs - target_intensity)) + lap_diff1*0.1
    loss.backward()
    # print(img_object.img.grad)

    optimizer.step()
    print(outputs)
    # print(torch.mean(img_object.img.grad))
    # scheduler.step(loss)
    losses.append(loss.item())

plt.plot(losses)
plt.show()
# for img_object in img_objects:
img_object = img_objects[4]
img_result = img_object.img.data.cpu().numpy()
img_result = np.reshape(img_result, (50, 50))
img = Image.fromarray(np.uint8(img_result * 255), 'L')
img.show()
