import spike_data_augmentation
import spike_data_augmentation.transforms as transforms

transform = transforms.Compose([transforms.TimeJitter(variance=3000),
                                transforms.SpatialJitter(variance_x=2, variance_y=2, sigma_x_y=0),
                                ])

testset = spike_data_augmentation.datasets.NMNIST(save_to='./data',
                                                  train=False,
                                                  download=True,
                                                  transform=transform)

# %%
testloader = spike_data_augmentation.datasets.Dataloader(testset, shuffle=True)

# %%
import numpy as np
surface_dimensions = [(5,5)]
number_of_features = [10]

test = iter(testloader)

for events, label in iter(testloader):
    print(label)

# %%

