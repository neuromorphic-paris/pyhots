from POKERDVS import POKERDVS
testset = POKERDVS(save_to='./data', train=False, download=False)

# %%
from Dataloader import Dataloader
testloader = Dataloader(testset, shuffle=False)

# %%
from Network import Network

surface_dimensions = [(5, 5)]
number_of_features = [4]
time_constants = [1e5]
sensor_size = (34, 34)

#for events, label in iter(testloader):
#    print(label)

net = Network(surface_dimensions_per_layer=surface_dimensions,
              number_of_features_per_layer=number_of_features,
              time_constants_per_layer=time_constants,
              sensor_size=sensor_size)

testiterator = iter(testloader)
events, label = next(testiterator)
net(events)
