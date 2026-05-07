import numpy as np


class CmpParams:
    def get_all_CMP(self, round=4):
        hitpoint = self.source_x
        interval = self.interval
        sensorNum = self.Num_sensor
        distance = self.distance
        cmp = np.zeros(sensorNum)
        for i in range(sensorNum):
            cmp[i] = np.round((distance[i] + hitpoint) / 2, round)
        return cmp

    def get_all_CMPdist(self):
        hitpoint = self.source_x
        interval = self.interval
        sensorNum = self.Num_sensor
        distance = self.distance
        dist = np.zeros(sensorNum)
        for i in range(sensorNum):
            dist[i] = (distance[i] - hitpoint) / 2
        return dist

    def get_all_CMPheight(self, dist):
        height = np.zeros(len(dist))
        if not hasattr(self, "z_distance"):
            return height
        distance_max, distance_min = np.max(dist), np.min(dist)
        ind_dmax, ind_dmin = np.argmax(dist), np.argmin(dist)
        for i, d in enumerate(dist):
            if d <= distance_min:
                height[i] = self.z_distance[ind_dmin]
            elif d >= distance_max:
                height[i] = self.z_distance[ind_dmax]
            else:
                plus_mask = dist > d
                minus_mask = dist < d
                plus_index = np.where(plus_mask)[0][np.argmin(dist[plus_mask])]
                minus_index = np.where(minus_mask)[0][np.argmax(dist[minus_mask])]
                height[i] = (
                    self.z_distance[plus_index] + self.z_distance[minus_index]
                ) / 2

        return height
