from POKERDVS import POKERDVS
from spike_data_augmentation.datasets.dataloader import Dataloader
from Network import Network

testset = POKERDVS(save_to='./data')

# %%
testloader = Dataloader(testset, shuffle=True)

# %%
surface_dimensions = [(11, 11)]
number_of_features = [16]
time_constants = [1e3]
sensor_size = (35, 35)
minimum_events = [5]
total_number_of_events = 89852 # di # 93082 # sp # testset.total_number_of_events()

net = Network(surface_dimensions_per_layer=surface_dimensions,
              number_of_features_per_layer=number_of_features,
              time_constants_per_layer=time_constants,
              sensor_size=sensor_size,
              plot_evolution=True,
              total_number_of_events=None)

testiterator = iter(testloader)

stop = 0
for events, label in testiterator:
    if label == 'di':
        print('Feeding ' + str(label) + '...')
        net(events)
    #stop += 1
    #if stop == 10:
    #    break
