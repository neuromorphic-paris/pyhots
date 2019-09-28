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

surface_dimensions = [(5, 5)]
number_of_features = [2]
time_constants = [1e5]
sensor_size = (34, 34)
minimum_events = [5]

iterator = iter(testloader)
events, target = next(iterator)

# %%
#for events, label in iter(testloader):
#    print(label)
