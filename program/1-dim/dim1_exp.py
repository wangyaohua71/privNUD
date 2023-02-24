# 一维 最终版

import copy
import math
import count
import re
import numpy as np
import Binomial_Distribution
import split_list
import min_b_dim1



def read_data(data_path, Domain):
    print('reading data and partitioning users...')
    dataset = np.loadtxt(data_path, np.int32)
    users = dataset.shape[0]
    np.random.shuffle(dataset)
    space_true_value = np.zeros(Domain, dtype=int)
    for data_per_user in dataset:
        space_true_value[data_per_user] += 1
    return [dataset, space_true_value, users]


def construct_tree(MIN, MAX, users):
    print('constructing tree ...')
    # 用户分配
    user_perPoint = []  # 每个节点分配的用户数
    user_perPath_rest = []  # 剩余用户
    user_domain_perPoint = []  # 每个节点对应的用户切片
    query_times_subtree_minB = []   # 每个节点对应子树的每一层查询次数(加阈值时用)
    max_query_times_subtree_minB = []   # 每个节点对应的子树的每一层查询次数最大值(加阈值时用)
    user_perPoint.append([0])  # 根节点不分配用户
    users_rest = users  # 剩余可分配的用户数
    user_perPath_rest.append([users_rest])
    user_domain_perPoint.append([[0, 0]])
    # 建树
    node = []
    t1 = [[MIN, MAX], -1, 0, 1]  # node[i][j][0]存储第i层第j个节点的区间
    tj1 = [t1]
    node.append(tj1)
    query_times_per_tree = 0
    query_times_per_tree_list = []
    all_query_times = (MAX - MIN + 1) * (MAX - MIN + 2) / 2
    b_root = min_b_dim1.root_fan_out(MIN=MIN, MAX=MAX)
    fan_out = [[b_root]]
    while True:
        flag_h = 1
        query_times_per_level = 0  # 每层查询次数（小层）
        query_times_per_level_list = []
        prob_query_times_per_level_list = []
        fan_out_level = []
        user_perPoint_level = []
        user_perPath_rest_level = []
        user_domain_level = []  # 每一层的用户切片
        query_times_subtree_minB_PerLevel = []
        max_query_times_subtree_minB_PerLevel = []
        tij = []  # 每层的node【i】【j】
        index_k = 0
        for j in range(0, len(fan_out[-1])):
            while node[-1][index_k][0][0] == node[-1][index_k][0][1]:
                index_k += 1
            A = node[-1][index_k][0]  # 当前节点的父节点的区间
            this_dim_left = A[0]
            this_dim__right = A[1]
            len_bin = math.ceil((this_dim__right - this_dim_left + 1) / fan_out[-1][j])
            for k in range(0, fan_out[-1][j]):
                users_rest = user_perPath_rest[-1][index_k]
                new_left = this_dim_left + len_bin * k
                new_right = this_dim_left + len_bin * (k + 1) - 1
                if new_left > this_dim__right:
                    break
                if new_right >= this_dim__right:
                    new_right = this_dim__right
                    fan_out[-1][j] = k + 1

                A_new = [new_left, new_right]
                ti = [A_new]
                ti.append(index_k)
                tij.append(ti)
                query_times_per_node = (new_left - MIN + 1) * (MAX - new_right + 1) - (
                        this_dim_left - MIN + 1) * (MAX - this_dim__right + 1)

                if A_new[0] != A_new[1]:
                    inform_this_node = min_b_dim1.best_fan_out(new_left, new_right, this_dim_left, this_dim__right, MIN, MAX)
                    b_this_node = inform_this_node[0]
                    query_times_subtree_minB_PerPoint = inform_this_node[1]
                    max_query_times_subtree_minB_PerPoint = inform_this_node[2]
                    fan_out_level.append(b_this_node)
                    flag_h *= 0
                    leaf_nodes_subtree = A_new[1] - A_new[0] + 1
                    n_perPoint = math.floor(users_rest / (math.ceil(math.log(leaf_nodes_subtree, b_this_node)) + 1))
                    user_perPoint_level.append(n_perPoint)
                    user_perPath_rest_level.append(user_perPath_rest[-1][index_k] - n_perPoint)
                    user_domain_level.append(
                        [user_domain_perPoint[-1][index_k][1], user_domain_perPoint[-1][index_k][1] + n_perPoint])
                    aaaa = query_times_subtree_minB_PerPoint[b_this_node - 2]
                    aaaa.append(query_times_per_node)
                    query_times_subtree_minB_PerLevel.append(aaaa)
                    max_query_times_subtree_minB_PerLevel.append(max_query_times_subtree_minB_PerPoint[b_this_node - 2])
                else:
                    flag_h *= 1
                    user_perPoint_level.append(users_rest)
                    user_perPath_rest_level.append(0)
                    user_domain_level.append([user_domain_perPoint[-1][index_k][1], users])


                query_times_per_level += query_times_per_node
                query_times_per_level_list.append(query_times_per_node)
                prob_query_times_per_level_list.append(query_times_per_node / all_query_times)
            index_k += 1
        query_times_per_tree += query_times_per_level
        query_times_per_tree_list.append(query_times_per_level_list)
        user_perPoint.append(user_perPoint_level)
        user_perPath_rest.append(user_perPath_rest_level)
        user_domain_perPoint.append(user_domain_level)
        node.append(tij)
        if flag_h == 1:
            break
        fan_out.append(fan_out_level)
        query_times_subtree_minB.append(query_times_subtree_minB_PerLevel)
        max_query_times_subtree_minB.append(max_query_times_subtree_minB_PerLevel)

    return [node, fan_out, user_domain_perPoint, user_perPoint, query_times_subtree_minB,
            max_query_times_subtree_minB]



def user_partition(data_path, tree_height, Domain, dim):
    print('reading data and partitioning users...')
    dataset = np.loadtxt(data_path, np.int32)
    users = dataset.shape[0]
    np.random.shuffle(dataset)     # 随机打乱用户（数据）
    space_true_value = np.zeros([Domain] * dim, dtype=int)
    for data_per_user in dataset:
        space_true_value[tuple(data_per_user)] += 1
    # 划分用户
    children_list_len = math.ceil(len(dataset) / (tree_height - 1))   # 根节点不需要用户
    code_list = split_list.list_of_groups(dataset, children_list_len)  # 用户分组，随机，每层均分
    return [code_list, space_true_value, users]


def add_threshold(node, eps, user_domain_perPoint, data_list, user_perPoint,
                  query_times_subtree_minB, max_query_times_subtree_minB, fan_out, users):
    print('adding threshold ...')

    #  标记node第二维index 与 fanout第二维index 的对应关系
    for i in range(1, len(node)):
        index_j_fanout = 0
        for j in range(0, len(node[i])):  # 遍历整棵树每一个节点,查找不需要划分的节点
            if node[i][j][0][0] != node[i][j][0][1]:
                node[i][j].append(index_j_fanout)  # node[i][j][2]   # 记录非叶节点对应的fanout[i][j]中的j
                index_j_fanout += 1
            else:
                node[i][j].append(-1)

    for i in range(1, len(node)):
        for j in range(0, len(node[i])):
            if node[i][j][2] != -100:  # node[i][j][2]=-100用来标记该节点不再划分
                # ---------------加噪------------------
                begin = user_domain_perPoint[i][j][0]
                end = user_domain_perPoint[i][j][1]
                number1 = count.count(node[i][j][0][0], node[i][j][0][1], data_list[begin:end])
                number = user_perPoint[i][j]
                f_est = Binomial_Distribution.binomial(eps, number, number1) / number
                node[i][j].append(f_est)
                if not (node[i][j][0][0] == node[i][j][0][1]):
                    var_oue = 4 * math.exp(eps) / pow((math.exp(eps) - 1), 2) / number
                    var_oue_no_partition = 4 * math.exp(eps) / pow((math.exp(eps) - 1), 2) / (users - user_domain_perPoint[i][j][0])
                    index_j_fanout = node[i][j][2]
                    B = fan_out[i][index_j_fanout]
                    var_patition = (query_times_subtree_minB[i - 1][index_j_fanout][-1] +
                                    query_times_subtree_minB[i - 1][index_j_fanout][0]) * var_oue
                    var1 = pow((f_est - f_est / B), 2)
                    var2 = pow((f_est / B), 2)
                    var_no_patition = query_times_subtree_minB[i - 1][index_j_fanout][-1] * var_oue_no_partition + (
                            query_times_subtree_minB[i - 1][index_j_fanout][0] -
                            max_query_times_subtree_minB[i - 1][index_j_fanout][0]) * var2 + max_query_times_subtree_minB[i - 1][index_j_fanout][0] * var1

                    if var_no_patition < var_patition:
                        # ------------------更新加噪值---------------
                        begin_rest_user = user_domain_perPoint[i][j][1]
                        end_rest_user = users
                        number_rest_user = users - begin_rest_user
                        number1_rest_user = count.count(node[i][j][0][0], node[i][j][0][1], data_list[begin_rest_user:end_rest_user])
                        f_est_rest = Binomial_Distribution.binomial(eps, number_rest_user,
                                                                    number1_rest_user) / number_rest_user
                        # 更新频率，用剩余所有用户回答
                        node[i][j][3] = (f_est * number + f_est_rest * number_rest_user) / (number + number_rest_user)
                        node[i][j][2] = -10
                        # 将所覆盖的孩子节点标记-100
                        now_level_parent = [j]
                        for no_patition_i in range(i + 1, len(node)):
                            next_level_parent = []
                            for no_patition_j in range(0, len(node[no_patition_i])):
                                if node[no_patition_i][no_patition_j][1] in now_level_parent:
                                    node[no_patition_i][no_patition_j][2] = -100

                                    next_level_parent.append(no_patition_j)
                                    if (no_patition_j + 1) < len(node[no_patition_i]) and node[no_patition_i][no_patition_j + 1][1] not in now_level_parent:
                                        break
                            now_level_parent = next_level_parent


def reconstruct_tree(node, fan_out):
    real_node = []
    real_fan_out = []
    for i in range(0, len(node)):
        real_node_perLevel = []
        real_fan_out_perLevel = []
        index_fan_out = 0
        for j in range(0, len(node[i])):
            if node[i][j][0][0] == node[i][j][0][1]:
                if node[i][j][2] != -100:
                    node[i][j][2] = -10  # 标记叶节点
                    index_parent = node[i][j][1]
                    node[i][j][1] = node[i - 1][index_parent][0]
                    real_node_perLevel.append(node[i][j])

            else:
                if node[i][j][2] != -100:
                    index_parent = node[i][j][1]
                    node[i][j][1] = node[i - 1][index_parent][0]
                    real_node_perLevel.append(node[i][j])
                    if node[i][j][2] != -10:
                        real_fan_out_perLevel.append(fan_out[i][index_fan_out])
                index_fan_out += 1
        if real_node_perLevel:
            real_node.append(real_node_perLevel)
        if real_fan_out_perLevel:
            real_fan_out.append(real_fan_out_perLevel)

    for ii in range(0, len(real_node)):
        for jj in range(0, len(real_node[ii])):
            for p in range(0, len(real_node[ii - 1])):
                if real_node[ii - 1][p][0] == real_node[ii][jj][1]:
                    real_node[ii][jj][1] = p
                    break
    real_node[0][0][1] = -1

    return [real_node, real_fan_out]


def non_negative(node, Domain, dim):
    print('non_negative producing...')

    nodes_perLevel = np.zeros(len(node))  # 这一层所覆盖的domain
    for i in range(0, len(node)):
        for j in range(0, len(node[i])):
            nodes_perLevel[i] += node[i][j][0][1] - node[i][j][0][0] + 1

    while True:
        flag = 0
        count_positive = np.zeros(len(node))  # 每层正数节点的个数
        sum_positive = np.zeros(len(node))  # 每层正数节点的f之和

        for i in range(0, len(node)):
            if nodes_perLevel[i] == Domain ** dim:  # 能覆盖整个domain
                for j in range(0, len(node[i])):
                    if node[i][j][3] < 0:
                        node[i][j][3] = 0
                        flag = 1

                    if node[i][j][3] > 0:
                        count_positive[i] += 1
                        sum_positive[i] += node[i][j][3]
        if flag == 0:
            break

        for i in range(0, len(node)):

            if count_positive[i] != 0:
                if nodes_perLevel[i] == Domain ** dim:  # 能覆盖整个domain
                    different = (1 - sum_positive[i]) / count_positive[i]
                    for j in range(0, len(node[i])):
                        if node[i][j][3] > 0:
                            node[i][j][3] += different

    while True:
        flag = 0
        count_positive = np.zeros(len(node))  # 每层正数节点的个数
        sum_positive = np.zeros(len(node))  # 每层正数节点的f之和

        for i in range(0, len(node)):
            if nodes_perLevel[i] != Domain ** dim:  # 不能覆盖整个domain
                for j in range(0, len(node[i])):
                    if node[i][j][3] < 0:
                        node[i][j][3] = 0
                        flag = 1

                    if node[i][j][3] > 0:
                        count_positive[i] += 1
                        sum_positive[i] += node[i][j][3]
        if flag == 0:
            break

        for i in range(0, len(node)):
            parents = []
            if count_positive[i] != 0:
                if nodes_perLevel[i] != Domain ** dim:  # 不能覆盖整个domain

                    sum_parents_non_nagetive_value = 0
                    for ii in range(0, len(node[i])):
                        parent_index = node[i][ii][1]
                        parent_node = node[i - 1][parent_index]
                        parents.append(parent_node)
                    parents_non_same = [parents[0]]
                    for iii in range(1, len(parents)):
                        if parents[iii] != parents[iii - 1]:
                            parents_non_same.append(parents[iii])
                    for i_parent in range(0, len(parents_non_same)):
                        sum_parents_non_nagetive_value += parents_non_same[i_parent][3]
                    different = (sum_parents_non_nagetive_value - sum_positive[i]) / count_positive[i]
                    for j in range(0, len(node[i])):
                        if node[i][j][3] > 0:
                            node[i][j][3] += different


def consistency(fan_out, node):
    # -----------------------------   一、存每个节点到根节点经过的路径，自上而下
    print('consistency  Part 1 ......')
    t = []
    for i in range(0, fan_out[0][0]):  # 第二层，每个节点的路径都是【V1】
        t.append([node[0][0][3]])
    path_non_consistency = [t]  # 加噪后的值(右侧)

    for LEVEL in range(2, len(node)):
        index_c = 0
        path_non_level = []
        for j in range(0, len(fan_out[LEVEL - 1])):
            while node[LEVEL - 1][index_c][0][2] == node[LEVEL - 1][index_c][0][3] and \
                    node[LEVEL - 1][index_c][0][0] == node[LEVEL - 1][index_c][0][1]:
                index_c += 1
            # path_non_j = []
            for k in range(0, fan_out[LEVEL - 1][j]):
                y = copy.deepcopy(path_non_consistency[-1][index_c])  # !!!!!!!!!!!!原来是path_non_consistency[-1][j]!!!!!
                y.append(node[LEVEL - 1][index_c][3])
                # path_non_j.append(y)  # 加噪值（右侧）
                path_non_level.append(y)
            index_c += 1
        path_non_consistency.append(path_non_level)

    # -------------------------------- 二、存系数 ，存在node[i][j][4]和[5]中
    print('consistency  Part 2 ......')
    weight = []
    for i in range(0, len(node)):
        for j in range(0, len(node[i])):
            if node[i][j][0][2] == node[i][j][0][3] and node[i][j][0][0] == node[i][j][0][1]:
                node[i][j].append(1)  # 叶节点所存 左侧系数为1
                node[i][j].append(1)  # 叶节点所存 右侧系数为1

    for level in range(len(node) - 2, -1, -1):  # 自下而上,从倒数第二层到根节点
        index_w = 0
        for j in range(0, len(fan_out[level])):
            sum_weight_point = 0
            for k in range(0, fan_out[level][j]):
                while node[level][index_w][0][2] == node[level][index_w][0][3] and \
                        node[level][index_w][0][0] == node[level][index_w][0][1]:
                    index_w += 1
                child_order = sum(fan_out[level][0:j]) + k
                sum_weight_point += node[level + 1][child_order][4]
            weight_point_l = float(sum_weight_point / (sum_weight_point + 1))  # 先存 系数的分母！
            weight_point_r = float(1 / (sum_weight_point + 1))
            node[level][index_w].append(weight_point_l)
            node[level][index_w].append(weight_point_r)
            index_w += 1

    # ---------------------------------------- 三、存右侧,存在node[i][j][6]中
    print('consistency  Part 3 ......')
    # 自下而上
    for i in range(1, len(node)):
        for j in range(0, len(node[i])):
            if node[i][j][0][2] == node[i][j][0][3] and node[i][j][0][0] == node[i][j][0][1]:
                sum_RHS_of_leaf_nodes = sum(path_non_consistency[i - 1][j]) + node[i][j][3]
                node[i][j].append(sum_RHS_of_leaf_nodes)

    for level in range(len(node) - 2, -1, -1):  # 自下而上,从 倒数第二层 到 第一层
        index_R = 0
        for j in range(0, len(fan_out[level])):
            sum_RHS_perNode = 0
            for k in range(0, fan_out[level][j]):
                while node[level][index_R][0][2] == node[level][index_R][0][3] and \
                        node[level][index_R][0][0] == node[level][index_R][0][1]:
                    index_R += 1
                child_order = sum(fan_out[level][0:j]) + k
                sum_RHS_perNode += node[level + 1][child_order][6]
            weighted_sum_RHS_perNode = sum_RHS_perNode * node[level][index_R][5]
            node[level][index_R].append(weighted_sum_RHS_perNode)
            index_R += 1

    # --------------------------------------- 四、计算一致性值，自上而下       一致性值存在node[i][j][7]中
    print('consistency  Part 4 ......')
    # node[0][0].append(node[0][0][6])      # 根节点： 一致性值=右侧
    node[0][0].append(1)  # 根节点一致性=1

    # path_consistency = []
    ttt = []
    for i in range(0, fan_out[0][0]):  # 第二层，每个节点的路径都是根节点
        ttt.append(node[0][0][6])
        data_consistency_2 = node[1][i][6] - node[1][i][4] * node[0][0][6]
        node[1][i].append(data_consistency_2)
    path_consistency = [ttt]

    for LEVEL in range(2, len(node)):
        index_con = 0
        path_level = []
        y_j = copy.deepcopy(path_consistency[LEVEL - 2])
        for j in range(0, len(fan_out[LEVEL - 1])):
            while node[LEVEL - 1][index_con][0][2] == node[LEVEL - 1][index_con][0][3] and \
                    node[LEVEL - 1][index_con][0][0] == node[LEVEL - 1][index_con][0][1]:
                index_con += 1

            for k in range(0, fan_out[LEVEL - 1][j]):
                nn = y_j[index_con] + node[LEVEL - 1][index_con][7]
                path_level.append(nn)
                child_order = sum(fan_out[LEVEL - 1][0:j]) + k
                data_consistency = node[LEVEL][child_order][6] - node[LEVEL][child_order][4] * nn
                node[LEVEL][child_order].append(data_consistency)

            index_con += 1
        path_consistency.append(path_level)


def consistency_non_partition(real_fan_out, real_node):
    # -----------------------------   一、存每个节点到根节点经过的路径，自上而下
    print('consistency  Part 1 ......')
    t = []
    for i in range(0, real_fan_out[0][0]):  # 第二层，每个节点的路径都是【V1】
        t.append([real_node[0][0][3]])
    path_non_consistency = [t]  # 加噪后的值(右侧)

    for LEVEL in range(2, len(real_node)):
        index_c = 0
        path_non_level = []
        for j in range(0, len(real_fan_out[LEVEL - 1])):
            while real_node[LEVEL - 1][index_c][2] == -10:
                index_c += 1
            # path_non_j = []
            for k in range(0, real_fan_out[LEVEL - 1][j]):
                y = copy.deepcopy(path_non_consistency[-1][index_c])
                y.append(real_node[LEVEL - 1][index_c][3])
                # path_non_j.append(y)  # 加噪值（右侧）
                path_non_level.append(y)
            index_c += 1
        path_non_consistency.append(path_non_level)

    # -------------------------------- 二、存系数 ，存在node[i][j][4]和[5]中
    print('consistency  Part 2 ......')
    weight = []
    for i in range(0, len(real_node)):
        for j in range(0, len(real_node[i])):
            if real_node[i][j][2] == -10:
                real_node[i][j].append(1)  # 叶节点所存 左侧系数为1
                real_node[i][j].append(1)  # 叶节点所存 右侧系数为1

    for level in range(len(real_node) - 2, -1, -1):  # 自下而上,从倒数第二层到根节点
        index_w = 0
        for j in range(0, len(real_fan_out[level])):
            sum_weight_point = 0
            for k in range(0, real_fan_out[level][j]):
                while real_node[level][index_w][2] == -10:
                    index_w += 1
                child_order = sum(real_fan_out[level][0:j]) + k
                sum_weight_point += real_node[level + 1][child_order][4]
            weight_point_l = float(sum_weight_point / (sum_weight_point + 1))
            weight_point_r = float(1 / (sum_weight_point + 1))
            real_node[level][index_w].append(weight_point_l)
            real_node[level][index_w].append(weight_point_r)
            index_w += 1

    # ---------------------------------------- 三、存右侧,存在node[i][j][6]中
    print('consistency  Part 3 ......')
    # 自下而上
    for i in range(1, len(real_node)):
        for j in range(0, len(real_node[i])):
            if real_node[i][j][2] == -10:
                sum_RHS_of_leaf_nodes = sum(path_non_consistency[i - 1][j]) + real_node[i][j][3]
                real_node[i][j].append(sum_RHS_of_leaf_nodes)

    for level in range(len(real_node) - 2, -1, -1):  # 自下而上,从 倒数第二层 到 第一层
        index_R = 0
        for j in range(0, len(real_fan_out[level])):
            sum_RHS_perNode = 0
            for k in range(0, real_fan_out[level][j]):
                while real_node[level][index_R][2] == -10:
                    index_R += 1
                child_order = sum(real_fan_out[level][0:j]) + k
                sum_RHS_perNode += real_node[level + 1][child_order][6]
            weighted_sum_RHS_perNode = sum_RHS_perNode * real_node[level][index_R][5]
            real_node[level][index_R].append(weighted_sum_RHS_perNode)
            index_R += 1

    # --------------------------------------- 四、计算一致性值，自上而下       node[i][j][7]
    print('consistency  Part 4 ......')
    # node[0][0].append(node[0][0][6])      # 根节点： 一致性值=右侧
    real_node[0][0].append(1)  #  根节点一致性=1

    # path_consistency = []
    ttt = []
    for i in range(0, real_fan_out[0][0]):  # 第二层，每个节点的路径都是根节点
        ttt.append(real_node[0][0][6])
        data_consistency_2 = real_node[1][i][6] - real_node[1][i][4] * real_node[0][0][6]
        real_node[1][i].append(data_consistency_2)
    path_consistency = [ttt]

    for LEVEL in range(2, len(real_node)):
        index_con = 0
        path_level = []
        y_j = copy.deepcopy(path_consistency[LEVEL - 2])
        for j in range(0, len(real_fan_out[LEVEL - 1])):
            while real_node[LEVEL - 1][index_con][2] == -10:
                index_con += 1

            for k in range(0, real_fan_out[LEVEL - 1][j]):
                nn = y_j[index_con] + real_node[LEVEL - 1][index_con][7]
                path_level.append(nn)
                child_order = sum(real_fan_out[LEVEL - 1][0:j]) + k
                data_consistency = real_node[LEVEL][child_order][6] - real_node[LEVEL][child_order][4] * nn
                real_node[LEVEL][child_order].append(data_consistency)

            index_con += 1
        path_consistency.append(path_level)


def query(query_list, Domain, dim, node):
    num = 0
    space_query = np.zeros([Domain] * dim, dtype=int)
    space_query[query_list[0]:query_list[1]+1] = 1
    while space_query[:].sum() != 0:
        for i in range(0, len(node)):
            for j in range(0, len(node[i])):
                dim1_l = node[i][j][0][0]
                dim1_r = node[i][j][0][1]
                area_node = (dim1_r - dim1_l + 1)
                sum_space_node = space_query[dim1_l:dim1_r+1].sum()
                if area_node == sum_space_node:
                    num += node[i][j][7]    # 非一致性，自上而下查询
                    space_query[dim1_l:dim1_r + 1] = 0
                elif node[i][j][2] == -10:
                    num += (node[i][j][7] * float(sum_space_node) / float(area_node))
                    space_query[dim1_l:dim1_r + 1] = 0
    return num


def leaves_information(Domain, dim, node):
    space_leaf_node = np.zeros([Domain] * dim, dtype=float)
    for i in range(0, len(node)):
        for j in range(0, len(node[i])):
            if node[i][j][0][0] == node[i][j][0][1]:
                point = [node[i][j][0][0], node[i][j][0][2]]
                eeeee = node[i][j][7]
                space_leaf_node[tuple(point)] += eeeee
    return space_leaf_node


def leaf_query(query_space, Domain, dim, node):
    begin_dim1 = query_space[0]
    end_dim1 = query_space[1]
    space_leaf_node = leaves_information(Domain, dim, node)
    sum_leaf_query = space_leaf_node[begin_dim1:end_dim1+1].sum()    # 一致性值
    return sum_leaf_query


def leaf_query_non_partition(query_list, Domain, dim, real_node):
    space_query = np.zeros([Domain] * dim, dtype=int)
    space_query[query_list[0]:query_list[1] + 1, query_list[2]:query_list[3] + 1] = 1

    num = 0

    real_leaves = []
    for i in range(0, len(real_node)):
        for j in range(0, len(real_node[i])):
            if real_node[i][j][2] == -10:
                real_leaves.append(real_node[i][j])
    while space_query[:, :].sum() != 0:
        for leave in real_leaves:
            dim1_l = leave[0][0]
            dim1_r = leave[0][1]
            dim2_l = leave[0][2]
            dim2_r = leave[0][3]
            area_node = (dim1_r - dim1_l + 1) * (dim2_r - dim2_l + 1)
            sum_space_node = space_query[dim1_l:dim1_r + 1, dim2_l:dim2_r + 1].sum()

            factor = float(sum_space_node) / float(area_node)
            num += leave[7] * factor
            space_query[dim1_l:dim1_r + 1, dim2_l:dim2_r + 1] = 0
    return num


def answer_queries(QUERY_TIME, query_path, space_true_value, user_scale, domain, real_node, consis_MSE_list):
    inf_query = open(query_path, 'r+')
    query_list = []
    for i in range(0, QUERY_TIME):
        strrr = inf_query.readline()
        lisss = re.findall(r'\d+', strrr)
        arrrr = list(map(int, lisss))
        query_list.append([arrrr[0], arrrr[1]])
    inf_query.close()

    var_list_consistency = []

    for times in range(0, QUERY_TIME):  # 查询次数
        query_MIN_dim1 = query_list[times][0]
        query_MAX_dim1 = query_list[times][1]

        num = query(query_list[times], Domain=domain, dim=1, node=real_node)   # 用树查询

        true_count = space_true_value[query_MIN_dim1:query_MAX_dim1 + 1].sum() / user_scale
        var_consistency = math.pow(true_count - num, 2)

        var_list_consistency.append(var_consistency)

        print(query_list[times], 'real_frequency:', true_count, 'noise_frequency', num)
    f = np.array(var_list_consistency)
    print('mean_var_consistency:', np.mean(f))
    consis_MSE_list.append(np.mean(f))
    for i in range(0, len(consis_MSE_list)):
        print(i, consis_MSE_list[i])

if __name__ == '__main__':

    domain = pow(2, 9)
    epsilon = 1

    data_set_path = '.\compare\dataset\BFive_produced.txt'
    query_path = '.\compare\query_table\dim1_query_2^9.txt'

    consistency_MSE_list = []
    Loop = 10

    for loop in range(0, Loop):
        # 建树
        MIN = 0
        MAX = domain - 1

        # 读数据
        DATA = read_data(data_path=data_set_path, Domain=domain)
        data_list = DATA[0]
        space_true_value = DATA[1]
        users = DATA[2]

        # 建树（交替划分），动态分配用户
        TREE = construct_tree(MIN=MIN, MAX=MAX, users=users)
        node = TREE[0]
        fan_out = TREE[1]
        user_domain_perPoint = TREE[2]
        user_perPoint = TREE[3]
        query_times_subtree_minB = TREE[4]
        max_query_times_subtree_minB = TREE[5]

        # 加噪 + 加阈值
        add_threshold(node=node, eps=epsilon, user_domain_perPoint=user_domain_perPoint,
                      data_list=data_list, user_perPoint=user_perPoint, query_times_subtree_minB=query_times_subtree_minB,
                      max_query_times_subtree_minB=max_query_times_subtree_minB, fan_out=fan_out, users=users)

        # 重构，去掉虚点
        real_node, real_fan_out = reconstruct_tree(node=node, fan_out=fan_out)

        # 非负处理
        non_negative(node=real_node, Domain=domain, dim=1)

        # 一致性
        consistency_non_partition(real_fan_out=real_fan_out, real_node=real_node)

        # 回答查询
        answer_queries(QUERY_TIME=1000, query_path=query_path, space_true_value=space_true_value, user_scale=users,
                       domain=domain, real_node=real_node, consis_MSE_list=consistency_MSE_list)

    consistency_MSE_list = np.array(consistency_MSE_list)
    print('mean_MSE', np.mean(consistency_MSE_list))


