import spike_data_augmentation
import spike_data_augmentation.transforms as transforms
from Network import Network

transform = transforms.Compose([#transforms.TimeJitter(variance=3000),
                                transforms.SpatialJitter(variance_x=2, variance_y=2, sigma_x_y=0),
                                ])

testset = spike_data_augmentation.datasets.IBMGesture(save_to='./data2',
                                                  train=False,
                                                  download=True,
                                                  transform=transform)
# %%
testloader = spike_data_augmentation.datasets.Dataloader(testset, shuffle=False)

# %%

surface_dimensions = [(11, 11)]
number_of_features = [16]
time_constants = [5e3]

iterator = iter(testloader)
events, target = next(iterator)
print(target)

# %%
#for events, label in iter(testloader):
#    print(label)
