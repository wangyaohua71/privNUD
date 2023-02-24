import numpy as np
import utility_metric as UM
import generate_query as GenQuery
import random
import TDG
import HDG
import parameter_setting as para

# 生成查询集并输出到txt文件（指定查询长度0.1~0.9）


def setup_args(args = None):
    # args.algorithm_name = 'TDG'
    args.algorithm_name = 'HDG'

    args.user_num = 0
    args.attribute_num = 5
    args.domain_size = pow(2, 6)

    args.epsilon = 1
    args.dimension_query_volume = 0.1
    args.query_num = 200
    args.query_dimension = 3


def sys_test():
    #   txt_dataset_path = "test_dataset/laplace2_dataset_users_1000000_attributes_6_domain_64.txt"
    args = para.generate_args()  # define the parameters
    setup_args(args=args)  # setup the parameters

    # generate range query****************************************************************
    # random_seed = 1
    # random.seed(random_seed)
    # np.random.seed(seed=random_seed)

    range_query = GenQuery.RangeQueryList(query_dimension=args.query_dimension,
                                          attribute_num=args.attribute_num,
                                          query_num=args.query_num,
                                          dimension_query_volume=args.dimension_query_volume, args=args)
    # 生成随机查询（属性+左右端点）
    range_query.generate_range_query_list_yuanshi()

    # 输出查询集
    txt_query_set_output = "query_set_output/query_list_d{}_lamda{}_len_{}.txt".format(args.attribute_num,
                                                                                       args.query_dimension,
                                                                                       args.dimension_query_volume)
    with open(txt_query_set_output, "w") as txt_output:
        range_query.output_range_query_list(txt_output)


if __name__ == '__main__':
    sys_test()
