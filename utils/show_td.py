import cv2
import numpy as np


def show_td(events, start_time=0, frame_length=24e3, wait_delay=1):
    timestamps = events[:,2]
    t_max = timestamps[-1]
    frame_start = timestamps[0]
    frame_end = frame_start + frame_length
    width = max(events[:,0])
    heigth = max(events[:,1])
    td_img = np.ones((heigth+1, width+1), dtype=np.uint8) * 128
    scale = 10
    resize_dims = (heigth*scale, width*scale)
    while frame_start < t_max:
        frame_data = events[(timestamps >= frame_start) & (timestamps < frame_end)]

        if frame_data.size > 0:
            for datum in frame_data:
                td_img[datum[1], datum[0]] = datum[3]

            td_img = np.piecewise(td_img, [td_img == 0, td_img == 1, td_img == 128], [0, 255, 128])
            resize = cv2.resize(td_img, resize_dims, interpolation = cv2.INTER_AREA)
            cv2.imshow('img', resize)
            cv2.waitKey(wait_delay)

        frame_start = frame_end + 1
        frame_end = frame_end + frame_length + 1

    cv2.destroyAllWindows()
    return

def show_td_surface(events, start_time=0, frame_length=24e3, decay_constant=1e5, wait_delay=1, scale=1):
    cv2.destroyAllWindows()
    t_index = 0
    x_index = 1
    y_index = 2
    timestamps = events[:,t_index]
    frame_start = timestamps[0]
    frame_end = frame_start + frame_length
    width = int(max(events[:,x_index]))
    heigth = int(max(events[:,y_index]))
    print('width: ' + str(width) + ', heigth: ' + str(heigth))
    polarities = 1
    ts_img = np.zeros((polarities, heigth+1, width+1))
    resize_dims = (width*scale, heigth*scale)
    for event in events:
        timestamp = event[t_index]
        ts_img[0, int(heigth - event[y_index]), int(event[x_index])] = timestamp

        if timestamp > frame_end:
            mask = np.where(ts_img != 0)
            surface = ts_img.copy()
            surface[mask] = (surface[mask] - timestamp)
            surface[mask] = np.exp(surface[mask] / decay_constant)
            surface = surface * 255
            #import ipdb; ipdb.set_trace()
            resize = cv2.resize(surface[0,:,:], resize_dims, interpolation = cv2.INTER_AREA)
            cv2.imshow('surface', resize)
            cv2.waitKey(wait_delay)
            frame_end = frame_end + frame_length

    cv2.destroyAllWindows()
    return