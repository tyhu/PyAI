import cv2
import imageio
import numpy as np

def yield_frames(vfn):
    reader = imageio.get_reader(vfn)
    try:
        for i, im in enumerate(reader):
            yield im
    except RuntimeError:
        print 'runtime error'
        pass

def get_flow_max_magnitude(frame,lastframe):
    flow = cv2.calcOpticalFlowFarneback(lastframe, frame,None, 0.5, 3, 15,3, 5, 1.2, 0)
    flow_mag = np.sqrt(flow[:,:,0]**2+flow[:,:,1]**2)
    return np.max(flow_mag)

def save_img(img, fn):
    imageio.imwrite(fn, img)
