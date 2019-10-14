from PIL import Image
import numpy as np
import visdom

# img = Image.open('./2007_000032.png')
# print(img.mode)

class LinePlotter(object):
    def __init__(self, env_name="main"):
        self.vis = visdom.Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.line(X=np.array([x, x]),
                                    Y=np.array([y, y]), env=self.env, opts=dict(
                                    legend=[split_name],
                                    title=var_name,
                                    xlabel="Iters",
                                    ylabel=var_name
                                    ))
        else:
            self.vis.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env,
                                win=self.plots[var_name], name=split_name, update='append')

# vis = visdom.Visdom(env='main')
# X = np.array([[1, 23]])
# Y = np.array([[1, 23]])
# assert X.shape[0] == Y.shape[0]
# vis.scatter(X=np.array([[1, 23],[2,34]]), Y=np.array([[1, 23]]), env='main',
# win=vis.scatter(X=np.array([[1, 23],[2,34]]), Y=np.array([[1, 23]])), update='append',name='loss')

plot = LinePlotter()
plot.plot('loss', 'train', 1, 0.35)
plot.plot('loss', 'train', 2, 0.78)
