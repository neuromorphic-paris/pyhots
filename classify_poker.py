from POKERDVS import POKERDVS
testset = POKERDVS(save_to='./data', train=False, download=False)

# %%
from Dataloader import Dataloader
testloader = Dataloader(testset, shuffle=False)

# %%
surface_dimensions = [(5,5)]
number_of_features = [10]

for events, label in iter(testloader):
    print(label)
