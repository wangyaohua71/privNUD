import math


def best_fan_out(left, right, j_left, j_right, MIN, MAX):
    var_subtree_per_b = []
    query_times_subtree_minB_PerPoint = []
    max_query_times_subtree_minB_PerPoint = []
    query_times_root = (left - MIN + 1) * (MAX - right + 1) - (j_left - MIN + 1) * (MAX - j_right + 1)
    for b in range(2, 21):
        tree = [[[left, right]]]
        h_subtree = 1   # 对于子树，根节点需要划分用户
        Height = math.ceil(math.log((right - left + 1), b)) + 1
        query_times_tree = 0
        var_tree = query_times_root * Height
        query_times_subtree_PerB = []
        max_query_times_subtree_PerB = []
        if math.log((right - left + 1), b) < 1:
            break
        while True:
            tij = []
            query_times_level = 0
            var_level = 0
            flag = 1
            max_level = []
            h_subtree += 1
            for j in range(0, len(tree[-1])):  # 上一层节点数
                if tree[-1][j][0] == tree[-1][j][1]:
                    continue
                parent_left = tree[-1][j][0]
                parent_right = tree[-1][j][1]
                len_bin = math.ceil((parent_right - parent_left + 1) / b)
                for k in range(0, b):      # 每个节点最多划分b叉
                    new_left = len_bin * k + parent_left
                    new_right = len_bin * (k + 1) - 1 + parent_left
                    if new_left > parent_right:
                        break
                    if new_right > parent_right:
                        new_right = parent_right

                    query_times_point = (new_left - MIN + 1) * (MAX - new_right + 1) - (parent_left - MIN + 1) * (MAX - parent_right + 1)
                    if new_left == new_right:
                        flag *= 1
                        var_per_point = query_times_point * Height / (Height - h_subtree + 1)
                    else:
                        flag *= 0
                        var_per_point = query_times_point * Height
                    ti = [new_left, new_right]
                    tij.append(ti)

                    query_times_level += query_times_point
                    var_level += var_per_point
                    max_level.append(query_times_point)
            query_times_tree += query_times_level
            var_tree += var_level
            query_times_subtree_PerB.append(query_times_level)  # 每层总次数的list
            max_query_times_subtree_PerB.append(max(max_level))  # 每层节点查询次数最大值的list

            tree.append(tij)

            if flag == 1:
                break
        # var_subtree_per_b.append((query_times_tree + query_times_root) * h_subtree)
        var_subtree_per_b.append(var_tree)
        query_times_subtree_minB_PerPoint.append(query_times_subtree_PerB)
        max_query_times_subtree_minB_PerPoint.append(max_query_times_subtree_PerB)
        # print('b', b, 'H', Height, 'h', h_subtree, var_tree)
    min_b = var_subtree_per_b.index(min(var_subtree_per_b)) + 2
    # print(min_b)

    return [min_b, query_times_subtree_minB_PerPoint, max_query_times_subtree_minB_PerPoint]


# best_fan_out(0, 200, 0, 200, 0, 200)


#   计算根节点 [MIN, MAX] 的最佳fan-out
def root_fan_out(MIN, MAX):
    var_subtree_per_b = []
    query_times_subtree_minB_PerPoint = []
    max_query_times_subtree_minB_PerPoint = []
    for b in range(2, 21):
        tree = [[[MIN, MAX]]]
        h_subtree = 0   # 根节点不需要划分用户
        Height = math.ceil(math.log((MAX - MIN + 1), b))
        query_times_tree = 0
        var_tree = 0
        query_times_subtree_PerB = []
        max_query_times_subtree_PerB = []
        if math.log((MAX - MIN + 1), b) < 1:
            break
        while True:
            tij = []
            query_times_level = 0
            var_level = 0
            flag = 1
            max_level = []
            h_subtree += 1
            for j in range(0, len(tree[-1])):  # 上一层节点数
                if tree[-1][j][0] == tree[-1][j][1]:
                    continue
                parent_left = tree[-1][j][0]
                parent_right = tree[-1][j][1]
                len_bin = math.ceil((parent_right - parent_left + 1) / b)
                for k in range(0, b):      # 每个节点最多划分b叉
                    new_left = len_bin * k + parent_left
                    new_right = len_bin * (k + 1) - 1 + parent_left
                    if new_left > parent_right:
                        break
                    if new_right > parent_right:
                        new_right = parent_right

                    query_times_point = (new_left - MIN + 1) * (MAX - new_right + 1) - (parent_left - MIN + 1) * (MAX - parent_right + 1)
                    if new_left == new_right:
                        flag *= 1
                        var_per_point = query_times_point * Height / (Height - h_subtree + 1)
                    else:
                        flag *= 0
                        var_per_point = query_times_point * Height
                    ti = [new_left, new_right]
                    tij.append(ti)

                    query_times_level += query_times_point
                    var_level += var_per_point
                    max_level.append(query_times_point)
            query_times_tree += query_times_level
            var_tree += var_level
            query_times_subtree_PerB.append(query_times_level)  # 每层总次数的list
            max_query_times_subtree_PerB.append(max(max_level))  # 每层节点查询次数最大值的list

            tree.append(tij)

            if flag == 1:
                break
        # var_subtree_per_b.append((query_times_tree + query_times_root) * h_subtree)
        var_subtree_per_b.append(var_tree)
        query_times_subtree_minB_PerPoint.append(query_times_subtree_PerB)
        max_query_times_subtree_minB_PerPoint.append(max_query_times_subtree_PerB)
        # print('b', b, 'H', Height, 'h', h_subtree, var_tree)
    min_b = var_subtree_per_b.index(min(var_subtree_per_b)) + 2
    #print('min_b', min_b)

    return min_b

# root_fan_out(0, 511)


def best_level_fan_out(query_times_list, MIN, MAX, node):  # 一层用一个fan-out
                                                           # query_times_list:已建好的level的查询次数
    var_subtree_per_b = []
    query_times_subtree_minB_PerPoint = []
    max_query_times_subtree_minB_PerPoint = []
    query_times_root = sum(query_times_list)
    l_r_list = [(i[0][1] - i[0][0] + 1) for i in node[-1]]
    for b in range(2, 21):
        tree = [node[-1]]
        h_subtree = 1 + len(query_times_list)   # 对于子树，根节点需要划分用户
        Height = math.ceil(math.log((max(l_r_list) + 1), b)) + 1 + len(query_times_list)
        query_times_tree = 0
        var_tree = query_times_root * Height
        query_times_subtree_PerB = []
        max_query_times_subtree_PerB = []
        if math.log((max(l_r_list) + 1), b) < 1:
            break
        while True:
            tij = []
            query_times_level = 0
            var_level = 0
            flag = 1
            max_level = []
            h_subtree += 1
            for j in range(0, len(tree[-1])):  # 上一层节点数
                if tree[-1][j][0][0] == tree[-1][j][0][1]:
                    continue
                parent_left = tree[-1][j][0][0]
                parent_right = tree[-1][j][0][1]
                len_bin = math.ceil((parent_right - parent_left + 1) / b)
                for k in range(0, b):      # 每个节点最多划分b叉
                    new_left = len_bin * k + parent_left
                    new_right = len_bin * (k + 1) - 1 + parent_left
                    if new_left > parent_right:
                        break
                    if new_right > parent_right:
                        new_right = parent_right

                    query_times_point = (new_left - MIN + 1) * (MAX - new_right + 1) - (parent_left - MIN + 1) * (MAX - parent_right + 1)
                    if new_left == new_right:
                        flag *= 1
                        var_per_point = query_times_point * Height / (Height - h_subtree + 1)
                    else:
                        flag *= 0
                        var_per_point = query_times_point * Height
                    ti = [new_left, new_right]
                    tij.append([ti])

                    query_times_level += query_times_point
                    var_level += var_per_point
                    max_level.append(query_times_point)
            query_times_tree += query_times_level
            var_tree += var_level
            query_times_subtree_PerB.append(query_times_level)  # 每层总次数的list
            max_query_times_subtree_PerB.append(max(max_level))  # 每层节点查询次数最大值的list

            tree.append(tij)

            if flag == 1:
                break
        # var_subtree_per_b.append((query_times_tree + query_times_root) * h_subtree)
        var_subtree_per_b.append(var_tree)
        query_times_subtree_minB_PerPoint.append(query_times_subtree_PerB)
        max_query_times_subtree_minB_PerPoint.append(max_query_times_subtree_PerB)
        # print('b', b, 'H', Height, 'h', h_subtree, var_tree)
    min_b = var_subtree_per_b.index(min(var_subtree_per_b)) + 2
    # print(min_b)

    return [min_b, query_times_subtree_minB_PerPoint, max_query_times_subtree_minB_PerPoint]


# best_fan_out(0, 200, 0, 200, 0, 200)
