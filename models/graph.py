import numpy as np


class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - motionbert
        - vitpose
        - mediapipe

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='motionbert',
                 strategy='spatial',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node,
                                        self.edge,
                                        max_hop=max_hop)
        self.get_adjacency(strategy)


    def __str__(self):
        return self.A


    def get_edge(self, layout):
        # edge is a list of [child, parent] paris

        if layout == 'vitpose':
            self.num_node = 35
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (34, 33), (33, 0), (0, 1), (1, 3), (0, 2), (2, 4), (33, 5),
                (5, 7), (7, 9), (9, 26), (26, 27), (9, 24), (24, 25), (9, 23),
                (33, 6), (6, 8), (8, 10), (10, 31), (31, 32), (10, 29), (29, 30),
                (10, 28), (34, 11), (11, 13), (13, 15), (15, 18), (15, 19), (15, 17),
                (34, 12), (12, 14), (14, 16), (16, 20), (16, 22), (16, 21)
            ]
            self.edge = self_link + neighbor_link
            self.center = 34
        
        elif layout == 'mediapipe':
            self.num_node = 29
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (28, 27), (27, 0), (0, 1), (1, 3), (0, 2), (2, 4), (27, 5),
                (5, 7), (7, 9), (9, 11), (9, 13), (9, 15), (27, 6), (6, 8), 
                (8, 10), (10, 16), (10, 14), (10, 12), (28, 17), (17, 19), 
                (19, 21), (21, 23), (21, 25), (28, 18), (18, 20), (20, 22),
                (22, 24), (22, 26)
            ]
            self.edge = self_link + neighbor_link
            self.center = 28
        
        elif layout == 'motionbert':
            self.num_node = 17
            self_link = [(i, i) for i in range(self.num_node)]
            # I set the direction of the arrow to be away from the center
            neighbor_link = [
                (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
                (12, 13), (8, 14), (14, 15), (15, 16), (0, 4), (4, 5),
                (5, 6), (0, 1), (1, 2), (2, 3)
            ]
            self.edge = self_link + neighbor_link
            self.center = 0
        
        else:
            raise ValueError(f'Do Not Exist This Layout: {layout}')


    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        # Adjacency matrix
        adjacency = np.zeros((self.num_node, self.num_node))
        
        # for hop in valid_hop:
        #     adjacency[self.hop_dis == hop] = 1
        
        # nodes that can be reached within the maximum number of hops are considered connected
        adjacency[self.hop_dis <= self.max_hop] = 1

        # elements in the same column sum to 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            # only one type of A, which is normalize_adjacency
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            # There are max_hop + 1 types of A
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                # the adjacency matrix for the given hop distance
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            # number of hops from j to the center is equal to the number of hops from i to the center
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            # Number of hops from j to the center is greater than from i to the center
                            elif self.hop_dis[j, self.center] > self.hop_dis[
                                    i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            # Number of hops from j to the center is less than from i to the center
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))

    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis
