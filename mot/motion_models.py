import sys
import numpy as np
from filterpy.kalman import KalmanFilter

def bb2z(bb):
    w = bb[2]-bb[0]
    h = bb[3]-bb[1]
    x = bb[0]+w/2.
    y = bb[1]+h/2.
    s = w*h
    r = w/float(h)
    return [x,y,s,r]
    
def z2bb(z):
    w = np.sqrt(z[2]*z[3])
    h = z[2]/w
    return [z[0]-w/2.,z[1]-h/2.,z[0]+w/2.,z[1]+h/2.]


class KFMotionModel(object):
    def __init__(self,bb):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4,0] = np.array(bb2z(bb))
        self.history = []
        self.predicted = False

    def update(self,bb):
        self.history = []
        bb = np.array(bb2z(bb))
        bb = np.expand_dims(bb, axis=1)
        self.kf.update(bb)
        self.predicted = False
        

    def predict(self):
        if not self.predicted:
            if((self.kf.x[6]+self.kf.x[2])<=0):
                self.kf.x[6] *= 0.0
            self.kf.predict()
            self.history.append(z2bb(self.kf.x))
            self.predicted=True
        return self.history[-1]

    def get_state(self):
        return z2bb(self.kf.x)

class LinearMotionModel(object):
    def __init__(self,bb):
        self.bb = bb
        self.z_his = [bb2z(bb)]
        #self.len_thres = 5
        self.len_thres = 3

    def update(self,bb):
        self.bb = bb
        self.z_his.append(bb2z(bb))
        if len(self.z_his)>self.len_thres: self.z_his.pop(0)

    def predict(self):
        self.bb = None
        his = np.array(self.z_his)
        if his.shape[0]==1:
            self.z_his*=2
        else:
            x,y,s,r = self.linear_prediction(his)

            self.z_his.append([x,y,s,r])
            if len(self.z_his)>self.len_thres: self.z_his.pop(0)

    def linear_prediction(self,his):
        t = np.array(range(his.shape[0]))
        t_mean, t_var = np.mean(t), np.var(t)
        x_mean, y_mean = np.mean(his[:,0]), np.mean(his[:,1])
        cov_xt = np.cov(his[:,0],y=t)[0,1]
        cov_yt = np.cov(his[:,1],y=t)[0,1]
        ax, ay = cov_xt/t_var, cov_yt/t_var
        bx, by = x_mean-ax*t_mean, y_mean-ay*t_mean

        x,y = ax*his.shape[0]+bx, ay*his.shape[0]+by
        s,r = np.mean(his[:,2]), np.mean(his[:,3])
        return x,y,s,r
        

    def get_state(self):
        if self.bb is None: return z2bb(self.z_his[-1])
        return self.bb

    def get_predicted_state(self):
        his = np.array(self.z_his)
        if his.shape[0]==1:
            return z2bb(self.z_his[-1])
        else:
            return z2bb(self.linear_prediction(his))
