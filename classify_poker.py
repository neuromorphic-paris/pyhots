from pyhots.POKERDVS import POKERDVS
from spike_data_augmentation.datasets.dataloader import Dataloader
from pyhots.Network import Network

testset = POKERDVS(save_to='./data',
                   file_dir='/home/jmatthieu/WORK/CODE/collaborations/HOTS-Dojo/Datasets/Cards/usable/pips')
# %%
# testloader = Dataloader(testset, shuffle=True)

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

# testiterator = iter(testloader)

n_boucle = 5
for n in range(n_boucle):
    print('\n------*\nLoop ' + str(n) + ' |\n------*')
    testloader = Dataloader(testset, shuffle=True)
    testiterator = iter(testloader)
    for events, label in testiterator:
        # if label == 'di':
        print('Feeding ' + str(label) + '...')
        net(events)

first = net.layers[0]
