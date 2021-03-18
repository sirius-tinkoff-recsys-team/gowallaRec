from config import config
from math import cos, asin, sqrt


class GeoDist:

    def __init__(self):
        self.max_dist = 100
        self.num_cats = config['CATS_IN_DIST']
        self.step = self.max_dist / self.num_cats

    @staticmethod
    def distance(x, y):
        lat1, lon1 = x[0], x[1]
        lat2, lon2 = y[0], y[1]
        p = 0.017453292519943295
        a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (
                    1 - cos((lon2 - lon1) * p)) / 2
        return 12742 * asin(sqrt(a))

    def geoseq2distseq(self, geoseq):
        distances = [0.] + [self.distance(x, y) for x, y in zip(geoseq[:-1], geoseq[1:])]
        assert len(distances) == len(geoseq)

        cats_of_dist = [min(x // self.step, self.num_cats - 1) for x in distances]
        return cats_of_dist
