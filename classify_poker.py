from pyhots.POKERDVS import POKERDVS
from spike_data_augmentation.datasets.dataloader import Dataloader
from pyhots.Network import Network
import ipdb

testset = POKERDVS(save_to='./data',
                   file_dir='/home/gregorlenz/Development/Github/HOTS-DOJO/Datasets/Cards/usable/pips')
# %%
surface_dimensions = [(5, 5)]
number_of_features = [16]
time_constants = [5e3]
sensor_size = (35, 35)
# total_number_of_events = testset.total_number_of_events()

net = Network(surface_dimensions_per_layer=surface_dimensions,
              number_of_features_per_layer=number_of_features,
              time_constants_per_layer=time_constants,
              sensor_size=sensor_size,
              plot_evolution=True,
              total_number_of_events=None)

counts = dict(zip(POKERDVS.classes, [0, 0, 0, 0]))

# pick 16 random files to choose bases from
testloader = Dataloader(testset, shuffle=True)
for index, events_and_label in enumerate(iter(testloader)):
    net(events_and_label[0])
    if index >= number_of_features[0]:
        break

# start the learning
testloader = Dataloader(testset, shuffle=True)
testiterator = iter(testloader)
for events, label in testiterator:
    if counts[label] < 17:
        net(events)
        counts[label] += 1
        print('Processed', end='')
        for key, value in counts.items():
            print(' ' + str(value) + ' ' + key + ',', end='')
        print('.')

first = net.layers[0]
