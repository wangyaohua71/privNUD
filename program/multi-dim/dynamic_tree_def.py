import copy
import math
import numpy as np


def list_of_groups(init_list, children_list_len):
    list_of_groups = zip(*(iter(init_list),) *children_list_len)
    end_list = [list(i) for i in list_of_groups]
    count = len(init_list) % children_list_len
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def user_partition(user_record, group_num):
    print('partitioning users...')

    # 划分用户
    children_list_len = math.ceil(len(user_record) / group_num)
    user_record_group_list = list_of_groups(user_record, children_list_len)  # 用户分组，随机，每层均分
    return user_record_group_list


def best_fan_out(left, right, j_left, j_right, MIN, MAX):
    var_subtree_per_b = []
    query_times_subtree_minB_PerPoint = []
    max_query_times_subtree_minB_PerPoint = []
    query_times_root = (left - MIN + 1) * (MAX - right + 1) - (j_left - MIN + 1) * (MAX - j_right + 1)
    for b in range(2, 21):   # b的范围？？
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


#   计算根节点 [MIN, MAX] 的最佳fan-out
def root_fan_out(MIN, MAX):
    var_subtree_per_b = []
    query_times_subtree_minB_PerPoint = []
    max_query_times_subtree_minB_PerPoint = []
    for b in range(2, 21):   # b的范围？？
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


def construct_tree(MIN, MAX, users):
    print('constructing tree ...')
    # 用户分配
    user_perPoint = []  # 每个节点分配的用户数
    user_perPath_rest = []  # 剩余用户
    user_domain_perPoint = []  # 每个节点对应的用户切片
    query_times_subtree_minB = []   # 每个节点对应子树的每一层查询次数(加阈值时用)
    max_query_times_subtree_minB = []   # 每个节点对应的子树的每一层查询次数最大值(加阈值时用)
    user_perPoint.append([0])  # 修改：根节点不分配用户
    users_rest = users  # 剩余可分配的用户数
    user_perPath_rest.append([users_rest])
    user_domain_perPoint.append([[0, 0]])
    # 建树
    node = []
    t1 = [[MIN, MAX], -1, 0, 1]  # node[i][j][0]存储第i层第j个节点的区间，例如【0,255,0,255,0,255,0,255】,四维，每维domain=256
    tj1 = [t1]
    node.append(tj1)
    query_times_per_tree = 0
    query_times_per_tree_list = []
    prob_query_times_per_tree_list = []  # 每个节点查询次数 占 全部查询次数 的比例
    mean_prob_query_times_per_tree_list = []  # 每层节点，查询次数比例的均值
    all_query_times = (MAX - MIN + 1) * (MAX - MIN + 2) / 2
    b_root = root_fan_out(MIN=MIN, MAX=MAX)
    # fan_out = [[5]]  # 第一层fan-out为5
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
        # level_data = code_list[(big_level - 1) * dim + d - 1]
        # space_this_level = np.zeros([Domian]*dim, dtype=int)
        # for item in level_data:
        #     space_this_level[tuple(item)] += 1    # 这层的用户在多维空间上的映射
        index_k = 0
        for j in range(0, len(fan_out[-1])):
            # while node[-1][index_k][0][2 * d - 2] == node[-1][index_k][0][2 * d - 1]:  # 父节点在该维度无法继续划分
            while node[-1][index_k][0][0] == node[-1][index_k][0][1]:
                index_k += 1
            A = node[-1][index_k][0]  # 当前节点的父节点的区间
            this_dim_left = A[0]  # 父节点区间左端点
            this_dim__right = A[1]   # 父节点区间右端点
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
                # A_new = copy.deepcopy(A)    # 当前节点的多维区间
                A_new = [new_left, new_right]
                ti = [A_new]
                ti.append(index_k)
                # ti.append(0)
                tij.append(ti)
                # if A_new[(2 * (d+1) - 2) % (2*dim)] != A_new[(2 * (d+1) - 1) % (2*dim)]:
                query_times_per_node = (new_left - MIN + 1) * (MAX - new_right + 1) - (
                        this_dim_left - MIN + 1) * (MAX - this_dim__right + 1)

                if A_new[0] != A_new[1]:
                    inform_this_node = best_fan_out(new_left, new_right, this_dim_left, this_dim__right, MIN, MAX)
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
        prob_query_times_per_tree_list.append(prob_query_times_per_level_list)
        mean_prob_query_times_per_tree_list.append(
            sum(prob_query_times_per_level_list) / len(prob_query_times_per_level_list))
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


def binomial(eps, number, number1):
    p = 0.5
    q = float(1 / (math.exp(eps) + 1))
    number0 = number - number1
    k1 = np.random.binomial(number1, p)
    k2 = np.random.binomial(number0, q)
    k = k1 + k2
    value = (k - number * q) / (p - q)
    return value


def count(left, right, data):
    sum = 0
    for i in data:
        if left <= i < right + 1:
            sum += 1
    return sum


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
                # now_parent = [j]
                # ---------------加噪------------------
                begin = user_domain_perPoint[i][j][0]
                end = user_domain_perPoint[i][j][1]
                number1 = count(node[i][j][0][0], node[i][j][0][1], data_list[begin:end])
                number = user_perPoint[i][j]
                f_est = binomial(eps, number, number1) / number
                node[i][j].append(f_est)
                if not (node[i][j][0][0] == node[i][j][0][1]):
                    var_oue = 4 * math.exp(eps) / pow((math.exp(eps) - 1), 2) / number
                    var_oue_no_partition = 4 * math.exp(eps) / pow((math.exp(eps) - 1), 2) / (users - user_domain_perPoint[i][j][0])
                    index_j_fanout = node[i][j][2]
                    B = fan_out[i][index_j_fanout]
                    var_patition = (query_times_subtree_minB[i - 1][index_j_fanout][-1] +
                                    query_times_subtree_minB[i - 1][index_j_fanout][0]) * var_oue
                    var1 = pow((f_est - f_est / B), 2)  # 节点频率为f对应的误差
                    var2 = pow((f_est / B), 2)  # 节点频率为0对应的误差
                    var_no_patition = query_times_subtree_minB[i - 1][index_j_fanout][-1] * var_oue_no_partition + (
                            query_times_subtree_minB[i - 1][index_j_fanout][0] -
                            max_query_times_subtree_minB[i - 1][index_j_fanout][0]) * var2 + max_query_times_subtree_minB[i - 1][index_j_fanout][0] * var1

                    # theta = math.sqrt((B + 1) * var_oue)
                    # theta = float('-inf')
                    # var_patition = float('-inf')
                    # if f_est < theta:
                    # noise_count_add = math.sqrt(number * 4 * math.exp(eps) / pow((math.exp(eps) - 1), 2))
                    # true_count_est = number * f_est
                    if var_no_patition < var_patition:  # 不划分
                        # ------------------更新加噪值---------------
                        begin_rest_user = user_domain_perPoint[i][j][1]
                        end_rest_user = users
                        number_rest_user = users - begin_rest_user
                        number1_rest_user = count(node[i][j][0][0], node[i][j][0][1], data_list[begin_rest_user:end_rest_user])
                        f_est_rest = binomial(eps, number_rest_user,
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
                                    '''uniform_est_num = ((node[no_patition_i][no_patition_j][0][1] -
                                                        node[no_patition_i][no_patition_j][0][0] + 1) / (node[i][j][0][1] - node[i][j][0][0] + 1)) * node[i][j][
                                                          3]
                                    node[no_patition_i][no_patition_j].append(uniform_est_num)'''

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

def tree_iteration(node):
    new_node = []
    for i in range(len(node)):
        new_node_perLevel = []
        for j in range(0, len(node[i])):
            point = node[i][j]
            new_point = [point[0], point[1], point[2], point[7]]
            new_node_perLevel.append(new_point)
        new_node.append(new_node_perLevel)

    return new_node


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
            parents = []
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
    node[0][0].append(1)  # 修改：  根节点一致性=1

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
                y = copy.deepcopy(path_non_consistency[-1][index_c])  # !!!!!!!!!!!!原来是path_non_consistency[-1][j]!!!!!
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
            weight_point_l = float(sum_weight_point / (sum_weight_point + 1))  # 先存 系数的分母！
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

    # --------------------------------------- 四、计算一致性值，自上而下       一致性值存在node[i][j][7]中
    print('consistency  Part 4 ......')
    # node[0][0].append(node[0][0][6])      # 根节点： 一致性值=右侧
    real_node[0][0].append(1)  # 修改：  根节点一致性=1

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
    return real_node


def leaf_nodes(node, domain):
    leaves_nodes = [0] * domain
    for i in range(0, len(node)):
        for j in range(0, len(node[i])):
            if node[i][j][0] == node[i][j][1]:
                index_leaves_nodes = node[i][j][1]
                leaves_nodes[index_leaves_nodes] = node[i][j][7]
            elif node[i][j][2] == -10:
                for index in range(node[i][j][0][0], node[i][j][0][1] + 1):
                    leaves_nodes[index] = node[i][j][7] / (node[i][j][0][1] - node[i][j][0][0] + 1)
    np.array(leaves_nodes)
    return leaves_nodes


def main(user_record, group_num, epsilon, domain):
    user_record_group_list = user_partition(user_record=user_record, group_num=group_num)

    MIN = 0
    MAX = domain - 1

    all_dim_leaf_nodes_list = []
    for i in range(group_num):

        # 数据
        data_list_all_dim = np.array(user_record_group_list[i])
        data_list = data_list_all_dim[:, i]

        # 建树
        node, fan_out, user_domain_perPoint, user_perPoint, query_times_subtree_minB, max_query_times_subtree_minB = construct_tree(MIN=MIN, MAX=MAX, users=len(data_list))

        # 加噪 + 加阈值
        add_threshold(node=node, eps=epsilon, user_domain_perPoint=user_domain_perPoint,
                      data_list=data_list, user_perPoint=user_perPoint,
                      query_times_subtree_minB=query_times_subtree_minB,
                      max_query_times_subtree_minB=max_query_times_subtree_minB, fan_out=fan_out, users=len(data_list))

        # 重构，去掉虚点
        real_node, real_fan_out = reconstruct_tree(node=node, fan_out=fan_out)

        # 非负处理
        non_negative(node=real_node, Domain=domain, dim=1)

        # 一致性
        consistency_non_partition(real_fan_out=real_fan_out, real_node=real_node)

        leaves = leaf_nodes(node=real_node, domain=domain)
        all_dim_leaf_nodes_list.append(leaves)
    return all_dim_leaf_nodes_list






