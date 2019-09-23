# -*- coding: utf-8 -*-

from show_td import show_td_surface
from numpy.lib import recfunctions as rfn
import loris

file = loris.read_file('/home/gregorlenz/Téléchargements/test.es')
events = file['events']

if events.dtype.names is not None:
    new_events = rfn.structured_to_unstructured(events)

show_td_surface(new_events, frame_length=5000, decay_constant=5000, wait_delay=50)