# pyhots

is supposed to provide public baselines for data sets such as [POKER-DVS](http://www2.imse-cnm.csic.es/caviar/POKERDVS.html), [NMNIST](https://www.garrickorchard.com/datasets/n-mnist) and [IBM gestures](http://www.research.ibm.com/dvsgesture/).

You will find different on- and offline algorithms using those datasets. 

Disclaimer: Not all libraries used in this repo are public yet.

## prerequisites
- `pip install -r requirements.txt` 
- install [jupyter-matplotlib](https://github.com/matplotlib/jupyter-matplotlib) to plot dynamically within notebooks
- install [jupyterlab-celltags](https://github.com/jupyterlab/jupyterlab-celltags) to enable papermill

## parallelisation
[papermill](https://github.com/nteract/papermill) is used to parametrize notebooks