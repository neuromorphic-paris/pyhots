from pyhots.POKERDVS import POKERDVS
from spike_data_augmentation.datasets.dataloader import Dataloader
import spike_data_augmentation.transforms as transforms
from pyhots.Network import Network

transform = transforms.Compose([transforms.DropEvent(drop_probability=0.0),
                                ])

testset = POKERDVS(file_dir='/home/gregorlenz/Development/Github/HOTS-DOJO/Datasets/Cards/usable/pips',
                   transform=transform)
# %%
surface_dimensions = [(11, 11)]
number_of_features = [16]
time_constants = [5e3]
total_number_of_events = testset.total_number_of_events()

net = Network(surface_dimensions_per_layer=surface_dimensions,
              number_of_features_per_layer=number_of_features,
              time_constants_per_layer=time_constants,
              sensor_size=POKERDVS.sensor_size,
              plot_evolution=True,
              total_number_of_events=total_number_of_events,
              reboot_bases=False,
              merge_polarities=True,)
              #drop_off_events=False,)

# pick 16 random files and one surface each to initialise bases
testloader = Dataloader(testset, shuffle=True)
for index, events_and_label in enumerate(iter(testloader)):
    net(events_and_label[0])
    if index >= number_of_features[0]:
        break

counts = dict(zip(POKERDVS.classes, [0, 0, 0, 0]))

# start the learning
testloader = Dataloader(testset, shuffle=True)
testiterator = iter(testloader)
for events, label in testiterator:
    counts[label] += 1
    net(events)
    print('Processed', end='')
    for key, value in counts.items():
        print(' ' + str(value) + ' ' + key + ',', end='')
    print('.')

first = net.layers[0]
