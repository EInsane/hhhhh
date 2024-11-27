import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import time
from sklearn.datasets import make_blobs
import random
from collections import defaultdict
import math
# from line_profiler import LineProfiler
from sklearn.neighbors import BallTree
import pandas as pd
import re
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_openml



class TreeNodeHST:
    def __init__(self, Lower, Upper, D, id_range):
        self.lower = Lower
        self.upper = Upper
        self.Dia = D
        self.id = id_range
        self.childrenlist = None


class kcenter_with_outliers:
    def __init__(self, data, k, z):
        self.data_ = data
        self.clusters_ = k
        self.outliers_ = z

    def cluster_radius(self, centers, outliers):
        pd = pairwise_distances(self.data_, centers)
        pd = np.min(pd, axis=1)
        pd_sort = np.sort(-pd)
        pd_arg_sort = np.argsort(-pd)
        return -pd_sort[outliers], pd_arg_sort[0:outliers]

    def Malkomes(self):
        ft = random.sample(range(0, self.data_.shape[0]), 1)[0]
        ft = self.data_[ft].reshape(1, -1)
        pd = (pairwise_distances(self.data_, ft))[:, 0]
        pd_id = np.zeros(pd.shape[0])

        for i in range(0, self.clusters_ + self.outliers_ - 1):
            furthest_id = np.argmax(pd)
            pd1 = pairwise_distances(self.data_, self.data_[furthest_id].reshape(1, -1))
            pd1 = np.min(pd1, axis=1)
            pd_diff = pd1 - pd
            small = np.argwhere(pd_diff < 0)[:, 0]
            pd[small] = pd1[small]
            pd_id[small] = i
            ft = np.vstack([ft, self.data_[furthest_id]])

        represents = ft
        centers_f = None

        weights1 = np.zeros(represents.shape[0])

        "Estimate the radius Upper Bound"
        rd = random.sample(range(0, ft.shape[0]), 1)[0]
        rd = ft[rd].reshape(1, -1)
        pd_rd = pairwise_distances(ft, rd)[:, 0]
        upper = np.max(pd_rd)

        "Estimate the radius Lower Bound"
        _, lower = self.Outlier_ESA_Box(self.data_, 0.2, 0.2, 10, self.outliers_)
        lower = lower / 2
        lower = max(lower, 1E-8)

        radius_epsilon = 0.2

        for i in range(0, weights1.shape[0]):
            weights1[i] = len(np.argwhere(pd_id == i)[:, 0])

        while (lower * (1 + radius_epsilon) < upper):
            radius = lower
            represents_temp = represents.copy()
            centers_now = None
            weights = weights1.copy()

            for i in range(0, self.clusters_):
                if (represents_temp.shape[0] < 1):
                    break
                max_id = None
                max_weights = -1
                for j in range(0, represents_temp.shape[0]):
                    center_now = represents_temp[j].reshape(1, -1)
                    pd = pairwise_distances(represents_temp, center_now)[:, 0]
                    yes_id = np.argwhere(pd < 2 * radius)[:, 0]
                    weights_sum = (weights[yes_id]).sum()
                    if (weights_sum > max_weights):
                        max_weights = weights_sum
                        max_id = j

                if (i == 0):
                    centers_now = represents_temp[max_id]
                else:
                    centers_now = np.vstack([centers_now, represents_temp[max_id]])
                center_now = represents_temp[max_id].reshape(1, -1)
                pd = pairwise_distances(represents_temp, center_now)[:, 0]
                large_id = np.argwhere(pd > 3 * radius)[:, 0]
                represents_temp = represents_temp[large_id]
                weights = weights[large_id]

            uncovered_points = weights.sum()

            lower = lower * (1 + radius_epsilon)

            if (uncovered_points <= z):
                "the first smallest radius"
                centers_f = centers_now.copy()
                break

        re_final = represents.copy()
        pd = pairwise_distances(re_final, centers_f)
        pd = np.min(pd, axis=1)
        if (centers_f.shape[0] < self.clusters_):
            while (centers_f.shape[0] < self.clusters_):
                furthest_id = np.argmax(pd)
                centers_f = np.vstack([centers_f, re_final[furthest_id]])
                # print("Add", centers_f.shape[0], self.clusters_)
                pd1 = pairwise_distances(re_final, re_final[furthest_id].reshape(1, -1))[:, 0]
                pd_diff = pd1 - pd
                small = np.argwhere(pd_diff < 0)[:, 0]
                pd[small] = pd1[small]

        return centers_f

    def Malkomes_Box(self, data1, z1):
        ft = random.sample(range(0, data1.shape[0]), 1)[0]
        ft = data1[ft].reshape(1, -1)
        pd = (pairwise_distances(data1, ft))[:, 0]
        pd_id = np.zeros(pd.shape[0])
        # print(pd_id.shape[0], data1.shape[0])

        if (data1.shape[0] < self.clusters_ + z1 - 1):
            ft = data1.shape[0]
            pd = np.zeros(data1.shape[0])
            pd_id = np.array([i for i in range(0, data1.shape[0])], dtype=int)
        else:
            for i in range(0, self.clusters_ + z1 - 1):
                furthest_id = np.argmax(pd)
                pd1 = pairwise_distances(data1, data1[furthest_id].reshape(1, -1))
                pd1 = np.min(pd1, axis=1)
                pd_diff = pd1 - pd
                small = np.argwhere(pd_diff < 0)[:, 0]
                pd[small] = pd1[small]
                pd_id[small] = i
                ft = np.vstack([ft, data1[furthest_id]])

        represents = ft
        centers_f = None

        weights1 = np.zeros(represents.shape[0])

        "Estimate the radius Upper Bound"
        rd = random.sample(range(0, ft.shape[0]), 1)[0]
        rd = ft[rd].reshape(1, -1)
        # print(ft.shape, rd.shape)
        pd_rd = pairwise_distances(ft, rd)[:, 0]
        upper = 2 * np.max(pd_rd)

        # print(ft, z1)
        "Estimate the radius Lower Bound"
        _, lower = self.Outlier_ESA_Box(data1, 0.2, 0.2, 20, z1)
        lower = lower / 2
        lower = max(lower, 1E-8)

        radius_epsilon = 0.2

        for i in range(0, weights1.shape[0]):
            weights1[i] = len(np.argwhere(pd_id == i)[:, 0])

        # print(weights1)
        # print(weights1.sum())

        while (lower * (1 + radius_epsilon) < upper):
            radius = lower
            represents_temp = represents.copy()
            centers_now = None
            weights = weights1.copy()

            for i in range(0, self.clusters_):
                if (represents_temp.shape[0] < 1):
                    break
                max_id = None
                max_weights = -1
                for j in range(0, represents_temp.shape[0]):
                    center_now = represents_temp[j].reshape(1, -1)
                    pd = pairwise_distances(represents_temp, center_now)[:, 0]
                    yes_id = np.argwhere(pd < 2 * radius)[:, 0]
                    weights_sum = (weights[yes_id]).sum()
                    if (weights_sum > max_weights):
                        max_weights = weights_sum
                        max_id = j

                if (i == 0):
                    centers_now = represents_temp[max_id].reshape(1, -1)
                else:
                    centers_now = np.vstack([centers_now, represents_temp[max_id]])
                center_now = represents_temp[max_id].reshape(1, -1)
                pd = pairwise_distances(represents_temp, center_now)[:, 0]
                large_id = np.argwhere(pd > 3 * radius)[:, 0]
                represents_temp = represents_temp[large_id]
                weights = weights[large_id]

            uncovered_points = weights.sum()

            lower = lower * (1 + radius_epsilon)

            if (uncovered_points <= z1):
                "the first smallest radius"
                centers_f = centers_now
                break

        re_final = represents.copy()
        pd = pairwise_distances(re_final, centers_f)
        pd = np.min(pd, axis=1)
        if (centers_f.shape[0] < self.clusters_):
            while (centers_f.shape[0] < self.clusters_):
                furthest_id = np.argmax(pd)
                centers_f = np.vstack([centers_f, re_final[furthest_id]])
                # print("Add", centers_f.shape[0], self.clusters_)
                pd1 = pairwise_distances(re_final, re_final[furthest_id].reshape(1, -1))[:, 0]
                pd_diff = pd1 - pd
                small = np.argwhere(pd_diff < 0)[:, 0]
                pd[small] = pd1[small]

        return centers_f

    def Outlier_NeurIPS(self, radius_epsilon):
        "Estimate the radius Upper Bound"
        rd = random.sample(range(0, self.data_.shape[0]), 1)[0]
        rd = self.data_[rd].reshape(1, -1)
        pd_rd = pairwise_distances(self.data_, rd)[:, 0]
        upper = np.max(pd_rd) * 2

        "Estimate the radius Lower Bound"
        _, lower = self.Outlier_ESA(0.2, 0.2, 10)
        lower = lower / 2
        lower = max(lower, 1E-8)

        centers_f = None
        radius_f = float("inf")
        # print("greedy radius lower", lower)


        while (lower * (1 + radius_epsilon) < upper):
            # print("Greedy Radius", lower)
            ft = random.sample(range(0, self.data_.shape[0]), 1)[0]
            center_list = [ft]
            centers_now = self.data_[ft].reshape(1, -1)
            ft = self.data_[ft].reshape(1, -1)
            pd = pairwise_distances(self.data_, ft)[:, 0]

            for i in range(0, self.clusters_ - 1):
                id_large = np.argwhere(pd > 2 * lower)[:, 0]
                if (id_large.shape[0] == 0):
                    break
                # print(id_large)
                nt = random.sample(range(0, id_large.shape[0]), 1)[0]

                if (nt):
                    center_list.append(id_large[nt])
                    next_center = self.data_[id_large[nt]]
                    centers_now = np.vstack([centers_now, next_center])
                    next_center = next_center.reshape(1, -1)
                    stop = 1
                    pd_new = pairwise_distances(self.data_, next_center)[:, 0]
                    pd_diff = pd_new - pd
                    id_small = np.argwhere(pd_diff < 0)[:, 0]
                    pd[id_small] = pd_new[id_small]
            radius_now, _ = self.cluster_radius(centers_now, self.outliers_)
            if (radius_now < radius_f):
                #print(center_list)
                radius_f = radius_now
                centers_f = centers_now.copy()
            lower = lower * (1 + radius_epsilon)
        return radius_f, centers_f

    def Outlier_NeurIPS_Box(self, data1, radius_epsilon, z):
        "Estimate the radius Upper Bound"
        rd = random.sample(range(0, data1.shape[0]), 1)[0]
        rd = data1[rd].reshape(1, -1)
        pd_rd = pairwise_distances(data1, rd)[:, 0]
        upper = np.max(pd_rd)

        # print("Estimation")

        "Estimate the radius Lower Bound"
        _, lower = self.Outlier_ESA_Box(data1, 0.2, 0.2, 20, z)
        lower = lower / 2
        lower = max(lower, 1E-8)

        centers_f = None
        radius_f = float("inf")
        # print("greedy radius lower", lower)

        while (lower * (1 + radius_epsilon) < upper):
            # print("Greedy Radius", lower)

            # print("lower", lower)
            ft = random.sample(range(0, data1.shape[0]), 1)[0]
            centers_now = data1[ft]
            ft = data1[ft].reshape(1, -1)
            pd = pairwise_distances(data1, ft)[:, 0]
            for i in range(0, self.clusters_ - 1):
                id_large = np.argwhere(pd > 2 * lower)[:, 0]
                if (id_large.shape[0] == 0):
                    break
                # print(id_large)
                nt = random.sample(range(0, id_large.shape[0]), 1)[0]
                if (nt):
                    next_center = data1[id_large[nt]]
                    centers_now = np.vstack([centers_now, next_center])
                    next_center = next_center.reshape(1, -1)
                    stop = 1
                    pd_new = pairwise_distances(data1, next_center)[:, 0]
                    pd_diff = pd_new - pd
                    id_small = np.argwhere(pd_diff < 0)[:, 0]
                    pd[id_small] = pd_new[id_small]
            radius_now = np.max(pd)
            if (radius_now < radius_f):
                radius_f = radius_now
                centers_f = centers_now
            lower = lower * (1 + radius_epsilon)
        return radius_f, centers_f

    def Outlier_ESA_Box(self, data1, epsilon, eta, rounds_amp, z):
        furthest_num = math.ceil((1 + epsilon) * z)
        sample_num = math.ceil(math.log2(1 / eta) * (1 + epsilon) / epsilon)

        "sample the first data point"
        ft = random.sample(range(0, data1.shape[0]), 1)[0]
        ft = data1[ft].reshape(1, -1)
        pd = (pairwise_distances(data1, ft))[:, 0]
        if (furthest_num > data1.shape[0]):
            return ft, 0

        for i in range(0, math.ceil(rounds_amp * self.clusters_)):
            if (ft.shape[0] > data1.shape[0]):
                break
            furthest_id = (np.argpartition(-pd, furthest_num))[0:furthest_num]
            sample_num = min(sample_num, furthest_id.shape[0])
            s_nt = random.sample(range(0, furthest_id.shape[0]), sample_num)
            if (len(s_nt) >= furthest_id.shape[0]):
                ft = np.vstack([ft, data1[furthest_id]])
                pd1 = pairwise_distances(data1, data1[furthest_id])
                pd1 = np.min(pd1, axis=1)
                pd_diff = pd1 - pd
                small = np.argwhere(pd_diff < 0)[:, 0]
                pd[small] = pd1[small]
            else:
                s_nt = np.array(s_nt, dtype=int)
                ft = np.vstack([ft, data1[furthest_id[s_nt]]])
                pd1 = pairwise_distances(data1, data1[furthest_id[s_nt]])
                pd1 = np.min(pd1, axis=1)
                pd_diff = pd1 - pd
                small = np.argwhere(pd_diff < 0)[:, 0]
                pd[small] = pd1[small]

        pd_sort = np.sort(-pd)

        return ft, -pd_sort[furthest_num]

    def Oulier_Uniform(self):
        "note that S = (log2* k / eta) * 3 * k / delta^2 epsilon1"
        eta = 0.5
        delta = 0.5
        S = math.ceil(5 * 1E-3 * self.data_.shape[0])
        # print(S)
        # epsilon1 = 3 * (math.log2(self.clusters_) / eta) * self.clusters_ / ((delta * delta) * S)
        epsilon2 = 0.2
        # print("epsilon2", epsilon2)
        z1 = math.ceil(2 * epsilon2 * S / self.clusters_)
        #print("Sample_Size", S, "Outliers", z1)

        radius_f = float("inf")
        centers_f = None

        for i in range(0, 10):
            data_sample_id = np.array(random.sample(range(0, self.data_.shape[0]), S), dtype=int)
            if (data_sample_id.shape[0] > self.data_.shape[0]):
                data_sample = self.data_.copy()
            else:
                data_sample = self.data_[data_sample_id]
            centers_now = self.Malkomes_Box(data_sample, z1)
            # print("check", centers_now)
            radius_now, _ = self.cluster_radius(centers_now, self.outliers_)
            if (radius_now < radius_f):
                radius_f = radius_now
                centers_f = centers_now.copy()

        return radius_f, centers_f

    def Outlier_ESA(self, epsilon, eta, rounds_amp):
        furthest_num = math.ceil((1 + epsilon) * self.outliers_)
        sample_num = math.ceil(math.log2(1 / eta) * (1 + epsilon) / epsilon)

        "sample the first data point"
        ft = random.sample(range(0, self.data_.shape[0]), 1)[0]
        ft = self.data_[ft].reshape(1, -1)
        pd = (pairwise_distances(self.data_, ft))[:, 0]
        if (furthest_num > self.data_.shape[0]):
            return ft

        for i in range(0, math.ceil(rounds_amp * self.clusters_)):
            furthest_id = (np.argpartition(-pd, furthest_num))[0:furthest_num]
            s_nt = random.sample(range(0, furthest_id.shape[0]), sample_num)
            if (len(s_nt) >= furthest_id.shape[0]):
                ft = np.vstack([ft, self.data_[furthest_id]])
                pd1 = pairwise_distances(self.data_, self.data_[furthest_id])
                pd1 = np.min(pd1, axis=1)
                pd_diff = pd1 - pd
                small = np.argwhere(pd_diff < 0)[:, 0]
                pd[small] = pd1[small]
            else:
                s_nt = np.array(s_nt, dtype=int)
                ft = np.vstack([ft, self.data_[furthest_id[s_nt]]])
                pd1 = pairwise_distances(self.data_, self.data_[furthest_id[s_nt]])
                pd1 = np.min(pd1, axis=1)
                pd_diff = pd1 - pd
                small = np.argwhere(pd_diff < 0)[:, 0]
                pd[small] = pd1[small]

        pd_sort = np.sort(-pd)

        return ft, -pd_sort[furthest_num]

    def Charikar(self, weights, data, z, radius_epsilon):
        if weights == None:
            weights = np.ones(data.shape[0])
        "Estimate the radius Upper Bound"
        rd = random.sample(range(0, data.shape[0]), 1)[0]
        rd = data[rd].reshape(1, -1)
        pd_rd = pairwise_distances(data, rd)[:, 0]
        upper = np.max(pd_rd)

        "Estimate the radius Lower Bound"
        _, lower = self.Outlier_ESA(0.2, 0.2, 10)

        centers_f = None
        max_radius = -1

        while (lower * (1 + radius_epsilon) < upper):
            # print("guessed_radius", lower)
            radius = lower
            data_to_cover = data.copy()
            centers_now = None
            for i in range(0, self.clusters_):
                if (data_to_cover.shape[0] < 1):
                    break
                "Finding the data point with maximum coverage"
                max_id = None
                max_weights = -1
                for j in range(0, data_to_cover.shape[0]):
                    center_now = data_to_cover[j].reshape(1, -1)
                    pd = pairwise_distances(data_to_cover, center_now)[:, 0]
                    yes_id = np.argwhere(pd < 2 * radius)[:, 0]
                    weights_sum = (weights[yes_id]).sum()
                    if (weights_sum > max_weights):
                        max_weights = weights_sum
                        max_id = j

                "To delete the covered data points"
                center_now = data_to_cover[max_id].reshape(1, -1)
                # print(centers_now)
                if i > 0:
                    centers_now = np.vstack([centers_now, data_to_cover[max_id]])
                else:
                    centers_now = data_to_cover[max_id].reshape(1, -1)
                pd = pairwise_distances(data_to_cover, center_now)[:, 0]
                large_id = np.argwhere(pd > 3 * radius)[:, 0]
                data_to_cover = data_to_cover[large_id]
                weights = weights[large_id]

            lower = lower * (1 + radius_epsilon)

            if (data_to_cover.shape[0] < z):
                centers_f = centers_now
                break

        return centers_f


class MultiView:
    def __init__(self, data, k, lda, delta, epsilon, problem, args):
        self.data_ = data
        self.clusters_ = k
        self.lda_ = lda
        self.delta_ = delta
        self.problem_ = problem
        self.args_ = args
        self.epsilon_ = epsilon
        self.root_ = self.shifted_quadtree(self.epsilon_)
        self.radius_list_ = self.range_cover(self.root_)
        # print(np.sort(self.radius_list_))

    # def CreateNode1(self, Node, shift, epsilon):
    #     k = self.clusters_
    #     if Node.id.shape[0] <= self.data_.shape[0] * epsilon / k:
    #         return None
    #     mid = 0.5 * (Node.lower + Node.upper)
    #     node_data = self.data_[Node.id]
    #     id_array = np.zeros(node_data.shape[0], dtype=int)
    #     mul = 0
    #     for i in range(self.data_.shape[1] - 1, 0, -1):
    #         data_diff = node_data[:, i] + shift[i] - mid[i]
    #         id_large = np.argwhere(data_diff > 0)[:, 0]
    #         id_array[id_large] += int(math.pow(2, mul))
    #         mul += 1

    #     id_array_sort = np.argsort(id_array)
    #     new_node_list = []
    #     start = 0
    #     for i in range(0, id_array_sort.shape[0] - 1):
    #         if (id_array[id_array_sort[i]] != id_array[id_array_sort[i + 1]]):
    #             binary = id_array[id_array_sort[i]]
    #             binary = bin(binary)[2:]
    #             binary = binary.zfill(self.data_.shape[1])
    #             binary = np.array(list(binary), dtype=int)
    #             lower_temp = Node.lower.copy()
    #             upper_temp = Node.upper.copy()
    #             small = np.argwhere(binary == 0)[:, 0]
    #             large = np.argwhere(binary == 1)[:, 0]
    #             lower_temp[large] = mid[large]
    #             upper_temp[small] = mid[small]
    #             D = math.sqrt(((upper_temp - lower_temp) ** 2).sum())
    #             new_node = TreeNodeHST(lower_temp, upper_temp, D, id_array_sort[np.arange(start, i + 1)])
    #             start = i + 1
    #             new_node_list.append(new_node)

    #     binary = id_array[id_array_sort[start]]
    #     binary = bin(binary)[2:]
    #     binary = binary.zfill(self.data_.shape[1])
    #     binary = np.array(list(binary), dtype=int)
    #     lower_temp = Node.lower.copy()
    #     upper_temp = Node.upper.copy()
    #     small = np.argwhere(binary == 0)[:, 0]
    #     large = np.argwhere(binary == 1)[:, 0]
    #     lower_temp[large] = mid[large]
    #     upper_temp[small] = mid[small]
    #     D = math.sqrt(((upper_temp - lower_temp) ** 2).sum())
    #     new_node = TreeNodeHST(lower_temp, upper_temp, D, id_array_sort[np.arange(start, node_data.shape[0])])
    #     new_node_list.append(new_node)

    #     return new_node_list

    # def CreateNode1(self, Node, shift, k, epsilon):
    #     if Node.id.shape[0] <= max(1, max(100, self.data_.shape[0] * epsilon / k)):
    #         # print("Pass")
    #         return None
    #
    #     if Node.id.shape[0] <= 1:
    #         return None
    #     d_diff = Node.upper - Node.lower
    #     if(np.max(d_diff)<1E-5):
    #         return None-0
    #
    #     #print(Node.id.shape[0])
    #
    #     mid = 0.5 * (Node.lower + Node.upper)
    #     #print("mid", mid)
    #     #print("Lower", Node.lower)
    #     #print("Upper", Node.upper)
    #
    #     node_data = self.data_[Node.id]
    #     str_count = np.full(node_data.shape[0], '', dtype=object)
    #
    #     for i in range(0, node_data.shape[1]):
    #         data_diff = node_data[:, i] + shift[i] - mid[i] # 取第i列的所有元素 变成一行
    #         id_large = np.argwhere(data_diff > 0)[:, 0]
    #         id_small = np.argwhere(data_diff <= 0)[:, 0] # 取data_diff<=0的所有的索引
    #         str_count[id_large] += '1'
    #         str_count[id_small] += '0'
    #
    #     str_count_id = np.argsort(str_count)
    #
    #     new_node_list = []
    #     start = 0
    #
    #     # print(str_count)
    #
    #     for i in range(0, str_count_id.shape[0] - 1):
    #         if (str_count[str_count_id[i]] != str_count[str_count_id[i + 1]]):
    #             str_now = str_count[str_count_id[i]]
    #             # print(str_now)
    #             large = [match.start() for match in re.finditer('1', str_now)]
    #             small = [match.start() for match in re.finditer('0', str_now)]
    #             large = np.array(large, dtype=int)
    #             small = np.array(small, dtype=int)
    #             lower_temp = Node.lower.copy()
    #             upper_temp = Node.upper.copy()
    #             lower_temp[large] = mid[large]
    #             upper_temp[small] = mid[small]
    #             D = math.sqrt(((upper_temp - lower_temp) ** 2).sum())
    #             if (start == i):
    #                 D = 0
    #             new_node = TreeNodeHST(lower_temp, upper_temp, D, str_count_id[np.arange(start, i + 1)])
    #             # print(start)
    #             start = i + 1
    #             new_node_list.append(new_node)
    #     temp = np.unique(str_count)
    #     # print("finished")
    #
    #     str_now = str_count[str_count_id[start]]
    #
    #     large = [match.start() for match in re.finditer('1', str_now)]
    #     small = [match.start() for match in re.finditer('0', str_now)]
    #     large = np.array(large, dtype=int)
    #     small = np.array(small, dtype=int)
    #     lower_temp = Node.lower.copy()
    #     upper_temp = Node.upper.copy()
    #     lower_temp[large] = mid[large]
    #     upper_temp[small] = mid[small]
    #     D = math.sqrt(((upper_temp - lower_temp) ** 2).sum())
    #     if (start == node_data.shape[0] - 1):
    #         D = 0
    #     new_node = TreeNodeHST(lower_temp, upper_temp, D, str_count_id[np.arange(start, node_data.shape[0])])
    #     new_node_list.append(new_node)
    #
    #     temp = 0
    #     for i in range(0, len(new_node_list)):
    #         temp += new_node_list[i].id.shape[0]
    #
    #     #print("Check Size", temp, "Real Size", Node.id.shape[0])
    #
    #     return new_node_list

        # for i in range(self.data_.shape[1] - 1, 0, -1):
        #     data_diff = node_data[:, i] + shift[i] - mid[i]
        #     id_large = np.argwhere(data_diff > 0)[:, 0]
        #     large_lsit.append(id_large)
        #     id_array[id_large] += int(math.pow(2, mul))
        #     mul += 1

        # for i in range(0, id_array_sort.shape[0] - 1):
        #     if (id_array[id_array_sort[i]] != id_array[id_array_sort[i + 1]]):
        #         binary = id_array[id_array_sort[i]]
        #         binary = bin(binary)[2:]
        #         binary = binary.zfill(data.shape[1])
        #         binary = np.array(list(binary), dtype=int)
        #         lower_temp = Node.lower.copy()
        #         upper_temp = Node.upper.copy()
        #         small = np.argwhere(binary == 0)[:, 0]
        #         large = np.argwhere(binary == 1)[:, 0]
        #         lower_temp[large] = mid[large]
        #         upper_temp[small] = mid[small]
        #         D = math.sqrt(((upper_temp - lower_temp) ** 2).sum())
        #         new_node = TreeNodeHST(lower_temp, upper_temp, D, id_array_sort[np.arange(start, i + 1)])

        #         start = i + 1
        #         new_node_list.append(new_node)

    def CreateNode1(self, Node, shift, k, epsilon):
        if Node.id.shape[0] <= max(1, max(100, self.data_.shape[0] * epsilon / k)):
            # print("Pass")
            return None
        if Node.id.shape[0] <= 1:
            return None
        d_diff = Node.upper - Node.lower
        if (np.max(d_diff) < 1E-5):
            return None

        # print(Node.id.shape[0])

        mid = 0.5 * (Node.lower + Node.upper)

        node_data = self.data_[Node.id]
        str_count = np.full(node_data.shape[0], '', dtype=object)

        for i in range(0, node_data.shape[1]):
            data_diff = node_data[:, i] + shift[i] - mid[i]  # 取第i列的所有元素 变成一行
            id_large = np.argwhere(data_diff > 0)[:, 0]
            id_small = np.argwhere(data_diff <= 0)[:, 0]  # 取data_diff<=0的所有的索引
            str_count[id_large] += '1'
            str_count[id_small] += '0'
        new_node_list = []
        u, indices = np.unique(str_count, return_inverse=True)
        for i in range(0, len(u)):
            str_now = u[i]
            large = [match.start() for match in re.finditer('1', str_now)]
            small = [match.start() for match in re.finditer('0', str_now)]
            large = np.array(large, dtype=int)
            small = np.array(small, dtype=int)
            lower_temp = Node.lower.copy()
            upper_temp = Node.upper.copy()
            lower_temp[large] = mid[large]
            upper_temp[small] = mid[small]
            D = math.sqrt(((upper_temp - lower_temp) ** 2).sum())
            id_temp = np.argwhere(indices == i)[:, 0]
            if len(id_temp) == 1:
                D = 0
            new_node = TreeNodeHST(lower_temp, upper_temp, D, Node.id[id_temp])
            new_node_list.append(new_node)
        return new_node_list

    def shifted_quadtree(self, epsilon):
        k = self.clusters_
        "Create the root node"
        D = math.sqrt(self.data_.shape[1])
        id_range = np.array([i for i in range(0, self.data_.shape[0])], dtype=int)
        lower = np.zeros(self.data_.shape[1])
        upper = np.ones(self.data_.shape[1])
        root = TreeNodeHST(lower, upper, D, id_range)

        "Start Partitioning"
        queue = [root]
        nodes = 0
        shift = np.zeros(self.data_.shape[1])
        for i in range(0, shift.shape[0]):
            shift[i] += np.random.uniform(0, 0.5)
        while queue:
            node = queue.pop(0)
            nodes += 1
            new_node_list = self.CreateNode1(node, shift, self.clusters_, epsilon)
            node.childrenlist = new_node_list

            if (node.childrenlist):
                for i in range(0, len(node.childrenlist)):
                    if (node.childrenlist[i]):
                        queue.append(node.childrenlist[i])
        return root

    def range_cover(self, root):
        lda = self.lda_
        delta = self.delta_
        radius_list = set()
        radius_list.add(root.Dia)
        "Start Range Cover"
        queue = [root]
        while queue:
            node = queue.pop(0)
            childrenlist = node.childrenlist
            for i in range(0, len(childrenlist)):
                if (childrenlist[i].childrenlist):
                    queue.append(childrenlist[i])
                rH = node.Dia / lda
                rL = max(childrenlist[i].Dia / lda, node.Dia / delta)
                #print(rL)
                tL = math.ceil(math.log2(rL) / math.log2(1 + lda))
                tR = math.floor(math.log2(rH) / math.log2(1 + lda))
                # print("Check", rL, rH)
                for j in range(tL, tR + 2):
                    # print(j)
                    radius_list.add(math.pow((1 + lda), j + 1) * lda)

        radius_list = np.array(list(radius_list))
        return np.sort(radius_list)

    def individual_kcenter(self, alpha):
        # alpha = self.args_[0]
        # rounds_amp = self.args_[3]
        "calculate the fairness radius"
        fair = np.zeros(self.data_.shape[0])
        l = math.ceil(self.data_.shape[0] / self.clusters_)
        for i in range(0, self.data_.shape[0]):
            if (i % 10000 == 0):
                print(i)
            data_query = self.data_[i]
            data_query = data_query.reshape(1, -1)
            pd = pairwise_distances(self.data_, data_query)[:, 0]
            fair[i] = np.partition(pd, l - 1)[l - 1]

        "Construct the core-points"
        # ft = random.sample(range(0, self.data_), 1)[0]
        # ft = self.data_[ft]
        # pd = pairwise_distances(self.data_, ft)[:,0]
        # r = np.zeros(self.data_.shape[0])
        # r_diff = pd - fair * alpha
        # small = np.argwhere(r_diff <= 0)[:,0]
        # r[small] = 1
        #
        # for i in range(0, math.ceil(rounds_amp * self.clusters_)):
        #     nt_id = np.argmax(pd)
        #     nt = self.data_[nt_id]
        #     nt = nt.reshape(1, -1)
        #     pd_nt = pairwise_distances(self.data_, nt)[:,0]
        #     pd_diff = pd_nt - pd
        #     small_id = np.argwhere(pd_diff < 0)
        #     pd[small_id] = pd_nt[small_id]
        #     r_diff = pd - fair * alpha
        #     small = np.argwhere(r_diff <=0)[:,0]
        #     r[small] = 1
        #     ft = np.vstack(ft, self.data_[nt_id])
        # for i in range(0, math.ceil(rounds_amp * self.clusters_)):
        #     large = np.argwhere(r == 0)[:,0]
        #     if(len(large) > 0):
        #         nt_id = np.argmin(fair[large])
        #         nt_id = large[nt_id]
        #     else:
        #         nt_id = np.argmin(fair)
        #
        #     nt = self.data_[nt_id]
        #     nt = nt.reshape(1, -1)
        #     pd_nt = pairwise_distances(self.data_, nt)[:,0]
        #     pd_diff = pd_nt - pd
        #     small_id = np.arghwere(pd_diff < 0)
        #     pd[small_id] = pd_nt[small_id]
        #     r_diff = pd - fair * alpha
        #     small = np.argwhere(r_diff <= 0)[:, 0]
        #     r[small] = 1
        #     ft = np.vstack(ft, self.data_[nt_id])

        # represents = np.unique(ft, axis=0, return_counts=False)

        "Construct the radius_list"
        root = self.shifted_quadtree(self.epsilon_)
        radius_list = self.range_cover(root)

        "Use Binary Search to Solve the Problem"
        id_lower = 0
        id_upper = radius_list.shape[0]
        best_rds = float('inf')

        centers_f = None
        best_rds = float("inf")

        while (1):
            if (id_lower > id_upper):
                break

            data_to_cover = self.data_.copy()
            fair_with_data = fair.copy()

            id_now = math.floor((id_lower + id_upper) / 2)
            radius = radius_list[id_now]
            # print("Upper and Lower", id_lower, id_upper)

            flag = 0

            centers_now = None

            for i in range(0, self.clusters_):
                centers_now_id = random.sample(range(0, data_to_cover.shape[0]), 1)[0]
                if (i == 0):
                    centers_now = data_to_cover[centers_now_id].reshape(1, -1)
                else:
                    centers_now = np.vstack([centers_now, data_to_cover[centers_now_id]])
                ft = data_to_cover[centers_now_id]
                ft = ft.reshape(1, -1)
                pd = pairwise_distances(data_to_cover, ft)[:, 0]
                radius_yes_id = np.argwhere(pd <= 2 * radius)[:, 0]
                radius_no_id = np.argwhere(pd > 2 * radius)[:, 0]
                fair_diff = pd[radius_yes_id] - fair_with_data[radius_yes_id] * alpha
                fair_no_id = np.argwhere(fair_diff > 0)[:, 0]
                id_to_cover = np.hstack([radius_no_id, fair_no_id])
                data_to_cover = data_to_cover[id_to_cover]
                fair_with_data = fair_with_data[id_to_cover]
                if (len(data_to_cover) <= 0):
                    flag = 1
                    break

            if (flag):
                "radius can be smaller"
                best_rds = min(best_rds, np.max(pd))
                centers_f = centers_now
                id_upper = id_now - 1
            else:
                "radius is too small"
                id_lower = id_now + 1

        return best_rds, centers_now

    def kcenter_with_outliers(self):
        z = self.args_[0]
        eta = self.args_[1]
        epsilon = self.args_[2]
        rounds_amp = self.args_[3]

        furthest_num = math.ceil((1 + epsilon) * z)
        sample_num = math.ceil(math.log2(1 / eta) * (1 + epsilon) / epsilon)
        #sample_num = 60



        "Construct the core-points"
        list_id = []
        ft = random.sample(range(0, self.data_.shape[0]), sample_num)
        list_id.append(ft)
        ft = self.data_[ft]
        pd = (pairwise_distances(self.data_, ft))
        pd_id = np.argmin(pd, axis=1)
        #pd_id = np.zeros(pd.shape[0])
        pd = np.min(pd, axis=1)
        # print(furthest_num)
        id_count_now = np.max(pd_id) + 1


        if (furthest_num > self.data_.shape[0]):
            return self.data_, np.ones(self.data_.shape[0])

        #id_count_now = 1

        for i in range(0, math.ceil(rounds_amp * k)):
            furthest_id = (np.argpartition(-pd, furthest_num))[0:furthest_num]
            # id_check = np.sort(np.argsort(-pd)[0:furthest_num])
            # id_check1 = np.sort(furthest_id)
            s_nt = random.sample(range(0, furthest_id.shape[0]), sample_num)
            list_id.append(furthest_id[s_nt])
            stop = 1
            if (len(s_nt) >= furthest_id.shape[0]):
                ft = np.vstack([ft, self.data_[furthest_id]])
                pd1 = pairwise_distances(self.data_, self.data_[furthest_id])
                pd_arg_temp = np.argmin(pd1, axis=1)
                pd_arg_temp = pd_arg_temp + id_count_now
                pd1 = np.min(pd1, axis=1)
                pd_diff = pd1 - pd
                small = np.argwhere(pd_diff < 0)[:, 0]
                pd[small] = pd1[small]
                pd_id[small] = pd_arg_temp[small]
                id_count_now = np.max(pd_arg_temp) + 1
            else:
                s_nt = np.array(s_nt, dtype=int)
                ft = np.vstack([ft, self.data_[furthest_id[s_nt]]])
                pd1 = pairwise_distances(self.data_, self.data_[furthest_id[s_nt]])
                pd_arg_temp = np.argmin(pd1, axis=1)
                pd_arg_temp = pd_arg_temp + id_count_now
                pd1 = np.min(pd1, axis=1)
                pd_diff = pd1 - pd
                small = np.argwhere(pd_diff < 0)[:, 0]
                pd[small] = pd1[small]
                pd_id[small] = pd_arg_temp[small]
                id_count_now = np.max(pd_arg_temp) + 1

        furthest_id = (np.argpartition(-pd, furthest_num))[0:furthest_num]
        sample_phase2 = math.ceil(math.log2(self.clusters_ / eta) * (1 + epsilon) * self.clusters_ / epsilon)
        sample_phase2 = min(sample_phase2, furthest_id.shape[0])
        s_nt = random.sample(range(0, furthest_id.shape[0]), sample_phase2)
        list_id.append(s_nt)
        s_nt = np.array(s_nt)
        batch_size = sample_num
        remaining = s_nt.shape[0]
        batch_start = 0
        while (batch_start < s_nt.shape[0]):
            id_range = s_nt[batch_start: min(batch_start + batch_size, s_nt.shape[0])]
            nt_temp = self.data_[furthest_id[id_range]]
            ft = np.vstack([ft, nt_temp])
            pd1 = pairwise_distances(self.data_, nt_temp)
            pd_arg_temp = np.argmin(pd1, axis=1)
            pd_arg_temp = pd_arg_temp + id_count_now
            pd1 = np.min(pd1, axis=1)
            pd_diff = pd1 - pd
            small = np.argwhere(pd_diff < 0)[:, 0]
            pd[small] = pd1[small]
            pd_id[small] = pd_arg_temp[small]
            id_count_now = np.max(pd_arg_temp) + 1
            batch_start = batch_start + batch_size

        # print("Cehck", ft.shape[0], np.unique(ft, axis=1).shape[0])

        list_tot = []
        for i in range(0,len(list_id)):
            for j in range(0,len(list_id[i])):
                list_tot.append(list_id[i][j])

        list_tot = np.array(list_tot, dtype=int)

        represents = ft
        # represents = np.unique(ft, axis=0, return_counts=False)
        # print("Size of Represents", represents.shape[0])

        "Construct the radius_list"
        radius_list = self.radius_list_.copy()

        "Use Binary Search to Solve the Problem"
        id_lower = 0
        id_upper = radius_list.shape[0]
        best_rds = float('inf')

        pd_argid = pd_id.copy()
        pd_id = pd.copy()

        centers_f = None

        represents_id = list_tot.copy()
        
        
        #weights_check = pairwise_distances(self.data_, represents)
        #weights_check = np.argmin(weights_check, axis=1)
        
        
        #print("represents size", represents.shape[0])
        
        
        # weights_temp = []
        # for i in range(0, represents.shape[0]):
        #     weights_temp.append([list_tot[i], len(np.argwhere(weights_check == i)[:, 0])])
            
        #print(weights_temp)
        #weights_temp_id_sort = np.argsort(-weights_temp)
        #print("largest_ids", list_tot[weights_temp_id_sort])
        
        
        
        # root_check = self.root_

        # print(np.sort(pd_id))
        #print(self.radius_list_)


        #print("Check_Duplicate", np.sort(list_tot))
        while (1):
            if (id_lower > id_upper):
                break
            id_now = math.floor((id_lower + id_upper) / 2)
            #print("Check", id_lower, id_upper)
            radius = radius_list[id_now]
            "Construct the weighted instance"
            # pd = pairwise_distances(self.data_, represents)
            # pd_id = np.min(pd, axis=1)
            # pd_argid = np.argmin(pd, axis=1)
            small_id = np.argwhere(pd_id <= 2 * radius)[:, 0]
            large_id = np.argwhere(pd_id > 2 * radius)[:, 0]
            "If there are many data points discarded, then continue to the next round"
            if (large_id.shape[0] >  (1+epsilon) * z):
                # print("Wrong")
                id_lower = id_now + 1
                continue

            pd_id_now = pd_argid[small_id]
            weights = np.zeros(represents.shape[0])

            #print(np.sort(pd_id_now))

            for i in range(0, weights.shape[0]):
                weights[i] = len((np.argwhere(pd_id_now == i))[:, 0])

            #print("weights", weights.sum())



            "Solve the weighted instance using greedy cover strategy"
            represents_temp = represents.copy()
            uncovered_points = large_id.shape[0]
            represents_id = list_tot.copy()
            centers_now = None
            list_now = []

            for i in range(0, self.clusters_):
                if (represents_temp.shape[0] < 1):
                    break
                "Finding the data point with maximum coverage"
                max_id = None
                max_weights = -1
                check = []
                for j in range(0, represents_temp.shape[0]):
                    center_now = represents_temp[j].reshape(1, -1)
                    pd = pairwise_distances(represents_temp, center_now)[:, 0]
                    yes_id = np.argwhere(pd < 2 * radius)[:, 0]
                    weights_sum = (weights[yes_id]).sum()
                    check.append([represents_id[j], weights_sum])
                    if (weights_sum > max_weights):
                        max_weights = weights_sum
                        max_id = j
                #print("cover", check)
                #print("max_id", represents_id[max_id], "cover_number", max_weights)

                stop  = 1

                list_now.append(represents_id[max_id])
                "To delete the representations covered"
                if (i == 0):
                    centers_now = represents_temp[max_id].reshape(1, -1)
                else:
                    centers_now = np.vstack([centers_now, represents_temp[max_id]])
                center_now = represents_temp[max_id].reshape(1, -1)
                pd = pairwise_distances(represents_temp, center_now)[:, 0]
                large_id = np.argwhere(pd > 3 * radius)[:, 0]
                represents_temp = represents_temp[large_id]
                represents_id = represents_id[large_id]
                weights = weights[large_id]
                
                #print("left id", represents_id, weights)

            #print(list_now)
            uncovered_points += weights.sum()

            if (uncovered_points <=  (1 + epsilon) * z):
                "radius can be smaller"
                #print("Pass", list_now)
                best_rds = min(best_rds, 3 * radius)
                centers_f = centers_now.copy()
                id_upper = id_now - 1
            else:
                "radius is too small"
                id_lower = id_now + 1

        #print(centers_f.shape[0])
        re_final = represents.copy()
        pd = pairwise_distances(re_final, centers_f)
        pd = np.min(pd, axis=1)
        if (centers_f.shape[0] < self.clusters_):
            while (centers_f.shape[0] < self.clusters_):
                furthest_id = np.argmax(pd)
                centers_f = np.vstack([centers_f, re_final[furthest_id]])
                # print("Add", centers_f.shape[0], self.clusters_)
                pd1 = pairwise_distances(re_final, re_final[furthest_id].reshape(1, -1))[:, 0]
                pd_diff = pd1 - pd
                small = np.argwhere(pd_diff < 0)[:, 0]
                pd[small] = pd1[small]

        return centers_f, best_rds / 3


def Check(data, centers, z):
    pd = pairwise_distances(data, centers)
    pd = np.min(pd, axis=1)
    pd_sort = np.sort(-pd)
    return -pd_sort[z]


def add_noise(data, delta, z):
    # print("number of outliers",z)
    d = data.shape[1]
    noise = ((np.random.rand(z, d) - 0.5) / 4) * delta + 3/8
    # print(np.max(noise))
    data = np.vstack((data, noise))
    noise_label = [i for i in range(len(data) - z, len(data))]
    return data, noise_label


if __name__ == "__main__":

    # data_path = "D:/codes/data/" + str(name_p[len(name_p) - 1])
    # data = np.loadtxt(data_path, delimiter=',', encoding='utf-8-sig')
    # data, _ = make_blobs(n_samples=50000000, n_features=10, centers=10)

    k_list = [30]
    name_p = ['covertype']
    #name_p = ['1']
    lda = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    epsilon = 0.01
    z_list = [0.1]
    itera = 1

    amp_list = [1]

    result = pd.DataFrame(
        columns=['dataset', 'k', 'z_rate', 'noise_range', 'method', 'best_cost', 'cost_mean', 'cost_std', 'time_mean',
                 'time_std', 'recall_mean', 'recall_std'])

    for d_i in range(len(name_p)):
        print(name_p[d_i])
        if (name_p[d_i] == 'covertype'):
            data = np.loadtxt("covertype.txt", delimiter=",")
        elif (name_p[d_i] == 'skin'):
            data = np.loadtxt("skin.txt", delimiter="	")
            data = data[:, 0:data.shape[1] - 1]
        elif(name_p[d_i] == 'NIPS'):
            data = np.loadtxt('nips.txt', delimiter="	")
        elif(name_p[d_i] == 'mnist'):
            mnist = fetch_openml('mnist_784', version=1, as_frame=False)
            data = mnist.data
            svd_model = TruncatedSVD(n_components=40)
            data = svd_model.fit_transform(data)
        elif (name_p[d_i] == 'SUSY'):
            data = np.loadtxt('SUSY1.csv', usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18),
                              delimiter=",")
            print(data[:,data.shape[1] - 1])
        elif (name_p[d_i] == 'higgs'):
            data = np.loadtxt('HIGGS.csv', delimiter=",")
            data = data[:, 1:data.shape[1]]
        else:
            data, _ = make_blobs(n_samples=50000, n_features=2, cluster_std=1.0)
        for noise_range in [1]:
            for d_j in range(len(k_list)):
                for d_z in range(len(z_list)):
                    print("k: ", k_list[d_j], "z: ", z_list[d_z])
                    axis_d_min = np.min(data)
                    axis_d_max = np.max(data)
                    
                    for i in range(0, data.shape[1]):
                        data[:, i] = (data[:, i] - axis_d_min) / (4 * (axis_d_max - axis_d_min)) + 1/4

                    amplify = axis_d_max - axis_d_min

                    k = k_list[d_j]
                    dataname = name_p[d_i]
                    z_rate = z_list[d_z]
                    z = math.floor(data.shape[0] * z_list[d_z])

                    #data1, true_labels = add_noise(data.copy(), 10, z)
                    data1 = data
                    true_labels = [i for i in range(0, z)]

                    t_scaling = time.time()
                    delta = 2 * data1.shape[0] * data1.shape[1]
                    #Solver = MultiView(data1, k, lda, delta, epsilon, "kcenter_with_outliers", [z, 0.001, 0.5, 1.2])
                    t_scaling = time.time() - t_scaling
                    Outlier_Solver = kcenter_with_outliers(data1, k, z)

                    # cost1f = float("inf")
                    # recall1f = -1
                    # time1_list = []
                    # cost1_list = []
                    # recall1_list = []

                    # for d_k in range(itera):
                    #     t1 = time.time()
                    #     centers_ours, rd_ours = Solver.kcenter_with_outliers()
                    #     t2 = time.time()
                    #     radius_ours, outlier_labels = Outlier_Solver.cluster_radius(centers_ours, z)
                    #     cost1_list.append(radius_ours)
                    #     time1_list.append(t2 - t1)
                    #     recall_ours = len(set(outlier_labels).intersection(set(true_labels))) / len(true_labels)
                    #     recall1_list.append(recall_ours)
                    #     if (radius_ours < cost1f):
                    #         cost1f = radius_ours
                    #     if (recall_ours > recall1f):
                    #         recall1f = recall_ours

                    # cost1_list = np.array(cost1_list)
                    # time1_list = np.array(time1_list)
                    # recall1_list = np.array(recall1_list)
                    # cost1_mean = np.mean(cost1_list)
                    # cost1_std = np.std(cost1_list)
                    # time1_mean = (np.sum(time1_list)) / itera +  + t_scaling/10
                    # time1_std = np.std(time1_list)
                    # recall1_mean = np.mean(recall1_list)
                    # recall1_std = np.mean(recall1_list)
                    # new = pd.DataFrame([[dataname, k, z_rate, "Ours", cost1f, cost1_mean, cost1_std, time1_mean,
                    #                      time1_std, recall1_mean, recall1_std]],
                    #                    columns=['dataset', 'k', 'z_rate', 'method', 'best_cost', 'cost_mean',
                    #                             'cost_std', 'time_mean', 'time_std', 'recall_mean', 'recall_std'])
                    # result = pd.concat([result, new])

                    # print("Ours", "best", cost1f, "time", time1_mean, 'recall', recall1f, "scaling_time", t_scaling)

                    # cost2f = float("inf")
                    # time2_list = []
                    # cost2_list = []
                    # recall2_list = []
                    # recall2f = -1

                    # for d_k in range(itera):
                    #     t1 = time.time()
                    #     radius_greedy, centers_greedy = Outlier_Solver.Outlier_NeurIPS(0.2)
                    #     t2 = time.time()
                    #     radius_greedy, labels_greedy = Outlier_Solver.cluster_radius(centers_greedy, z)
                    #     cost2_list.append(radius_greedy)
                    #     time2_list.append(t2 - t1)
                    #     recall_greedy = len(set(labels_greedy).intersection(set(true_labels))) / len(true_labels)
                    #     recall2_list.append(recall_greedy)
                    #     if (radius_greedy < cost2f):
                    #         cost2f = radius_greedy
                    #     if (recall_greedy > recall2f):
                    #         recall2f = recall_greedy

                    # cost2_list = np.array(cost2_list)
                    # time2_list = np.array(time2_list)
                    # recall2_list = np.array(recall2_list)
                    # cost2_mean = np.mean(cost2_list)
                    # cost2_std = np.std(cost2_list)
                    # time2_mean = np.mean(time2_list)
                    # time2_std = np.std(time2_list)
                    # recall2_mean = np.mean(recall2_list)
                    # recall2_std = np.std(recall2_list)
                    # new = pd.DataFrame([[dataname, k, z_rate, "NeurIPS_Greedy", cost2f, cost2_mean, cost2_std,
                    #                      time2_mean, time2_std, recall2_mean, recall2_std]],
                    #                    columns=['dataset', 'k', 'z_rate', 'method', 'best_cost', 'cost_mean',
                    #                             'cost_std', 'time_mean', 'time_std', 'recall_mean', 'recall_std'])
                    # result = pd.concat([result, new])

                    # print("GreedyNeurIPS", "best", cost2f, "time", time2_mean, 'recall', recall2f)

                    # cost3f = float("inf")
                    # time3_list = []
                    # cost3_list = []
                    # recall3_list = []
                    # recall3f = -1

                    # for d_k in range(itera):
                    #     t1 = time.time()
                    #     radius_uniform, centers_uniform = Outlier_Solver.Oulier_Uniform()
                    #     t2 = time.time()
                    #     radius_uniform, labels_uniform = Outlier_Solver.cluster_radius(centers_uniform, z)
                    #     cost3_list.append(radius_greedy)
                    #     time3_list.append(t2 - t1)
                    #     recall_uniform = len(set(labels_uniform).intersection(set(true_labels))) / len(true_labels)
                    #     recall3_list.append(recall_uniform)
                    #     if (radius_uniform < cost3f):
                    #         cost3f = radius_uniform
                    #     if (radius_uniform > recall3f):
                    #         recall3f = recall_uniform

                    # cost3_list = np.array(cost3_list)
                    # time3_list = np.array(time3_list)
                    # recall3_list = np.array(recall3_list)
                    # cost3_mean = np.mean(cost3_list)
                    # cost3_std = np.std(cost3_list)
                    # time3_mean = np.mean(time3_list)
                    # time3_std = np.std(time3_list)
                    # recall3_mean = np.mean(recall3_list)
                    # recall3_std = np.std(recall3_list)
                    # new = pd.DataFrame([[dataname, k, z_rate, "Ding", cost3f, cost3_mean, cost3_std, time3_mean,
                    #                      time3_std, recall3_mean, recall3_std]],
                    #                    columns=['dataset', 'k', 'z_rate', 'method', 'best_cost', 'cost_mean',
                    #                             'cost_std', 'time_mean', 'time_std', 'recall_mean', 'recall_std'])
                    # result = pd.concat([result, new])

                    # print("Ding", "best", cost3f, "time", time3_mean, "recall", recall3f)

                    cost4f = float("inf")
                    time4_list = []
                    cost4_list = []

                    for d_k in range(itera):
                        t1 = time.time()
                        centers_Mal = Outlier_Solver.Malkomes()
                        t2 = time.time()
                        radius_Mal, _ = Outlier_Solver.cluster_radius(centers_Mal, z)
                        cost4_list.append(radius_Mal)
                        time4_list.append(t2 - t1)
                        if(radius_Mal<cost4f):
                            cost4f = radius_Mal

                    cost4_list = np.array(cost4_list)
                    time4_list = np.array(time4_list)
                    cost4_mean = np.mean(cost4_list)
                    cost4_std = np.std(cost4_list)
                    time4_mean = np.mean(time4_list)
                    time4_std = np.std(time4_list)
                    new = pd.DataFrame([[dataname, k, z_rate, "Malkomes", cost4f, cost4_mean, cost4_std, time4_mean, time4_std]],
                    columns = ['dataset', 'k', 'z_rate', 'method', 'best_cost', 'cost_mean', 'cost_std', 'time_mean', 'time_std'])
                    result = pd.concat([result, new])

                    print("NeurIPSTwoStage", "best", cost4f, "time", time4_mean)

            print('-' * 80)

    result.to_csv("result_covertype_Malcomesz.csv", index=False)

    # print("Charikar Start")
    # # centers_charikar = Outlier_Solver.Charikar(None, data, z, 0.5)
    # # print(centers_charikar.shape)

    # t3 = time.time()

    # radius_ours = Outlier_Solver.cluster_radius(rd, z)
    # # radius_charikar = Outlier_Solver.cluster_radius(centers_charikar, z)

    # t4 = time.time()
    # print("Uniform Sampling")
    # #radius_uniform = Outlier_Solver.Oulier_Uniform()
    # t5 = time.time()

    # print("Greedy Sampling")
    # radius_greedy, centers_greedy = Outlier_Solver.Outlier_NeurIPS(0.2)

    # t6 = time.time()

    # #radius_ours_indi, centers_ours_indi = Solver.individual_kcenter(2)

    # t7 = time.time()

    # print("Radius Ours", radius_ours)
    # # print("Radius Charikar", radius_charikar)
    # #print("Radius Uniform", radius_uniform)
    # print("Radius Greedy", radius_greedy)

    # print("Ours Time", t2 - t1)
    # print("Charikar Time", t3 - t2)
    # #print("Uniform Time", t5 - t4)
    # print("Greedy Time", t6 - t5)

    # #print("Individual Ours Time", t7 - t6, "radius", radius_ours_indi)

