#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <utility>
#include <unordered_set>
#include <execution>
#include <omp.h>
#include <Eigen/Dense>


// 不同波段需要修改的参数
constexpr int IND_SIZE = 124001;
constexpr int POP_SIZE = 50000;  // 可调试时先用小一点
constexpr int GENE_MAX = 2;  // 基因可取值范围 [0, GENE_MAX)
const std::vector<int> values = { 4, 19, 20, 34, 50, 78, 102, 124, 125, 139 };

// 预先估计一个每组的成员数量，方便分配内存
constexpr int group_members_num_est = 100000;

// 全局定义超参数
constexpr int GENERATIONS = 100000;
constexpr int TOURN_SIZE = 2;
constexpr double CROSS_PROB = 0.8;
constexpr double MUT_PROB = 0.002;
constexpr int MUT_POINTS = 100;


// 随机数生成器
std::random_device rd;
std::mt19937 gen(rd());  // 或 std::mt19937 gen(1234); 固定种子
std::uniform_int_distribution<> pop_dist(0, POP_SIZE - 1);  // 随机选一个个体索引
std::uniform_int_distribution<> gene_dist(0, GENE_MAX - 1);  // 随机生成一个基因型的分组号
std::uniform_real_distribution<> prob_dist(0.0, 1.0); // 0~1概率
std::uniform_int_distribution<> bin_dist(0, 1);  // 随机取0或1
std::uniform_int_distribution<> ind_dist(0, IND_SIZE - 1);  // 基因点位
std::uniform_int_distribution<> state_dist(0, 45 - 1);  // 基因点位


// 读取算好的二进制文件
template <typename T>
Eigen::Array<T, -1, -1> read_mat(std::string file_dir, int rows, int cols) {

    // Eigen 默认是 column-major
    Eigen::Array<T, -1, -1> mat(rows, cols);

    std::ifstream file(file_dir, std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件" << std::endl;
        exit(-1);
    }

    // 按列读取数据（因为是列优先存的）
    file.read(reinterpret_cast<char*>(mat.data()), sizeof(T) * rows * cols);
    file.close();

    return mat;
}


// 初始化-并行版
Eigen::ArrayXX<uint8_t> init_pop_omp(int pop_size, int ind_size, int gene_max) {
    Eigen::ArrayXX<uint8_t> pop(ind_size, pop_size);

#pragma omp parallel
    {
        std::mt19937 local_gen(9012 + omp_get_thread_num());  // 每个线程自己的 RNG
        std::uniform_int_distribution<int> local_gene_dist(0, gene_max - 1);

#pragma omp for schedule(guided)
        for (int i = 0; i < ind_size; ++i) {
            for (int j = 0; j < pop_size; ++j) {
                pop(i, j) = static_cast<uint8_t>(local_gene_dist(local_gen));
            }
        }
    }
    return pop;
}



// 锦标赛选择-并行版本
int tournament_selection_omp(
    std::vector<int>& selected_idx,
    const Eigen::ArrayXX<uint8_t>& pop,
    const std::vector<double>& fitness,
    int tourn_size
) {

#pragma omp parallel
    {
        // 每个线程一个本地 RNG，使用线程号作为种子偏移
        std::mt19937 local_gen(1234 + omp_get_thread_num());
        std::uniform_int_distribution<int> local_pop_dist(0, POP_SIZE - 1);

#pragma omp for schedule(guided)
        for (int j = 0; j < POP_SIZE; ++j) {
            int best_idx = -1;
            double best_fit = std::numeric_limits<double>::max();

            for (int t = 0; t < tourn_size; ++t) {
                int idx = local_pop_dist(local_gen);
                if (fitness[idx] < best_fit) {
                    best_fit = fitness[idx];
                    best_idx = idx;
                }
            }
            selected_idx[j] = best_idx;
        }
    }

    return 0;
}


// 交叉-并行版
int crossover_omp(
    const std::vector<int>& selected_idx,
    const Eigen::ArrayXX<uint8_t>& pop,
    Eigen::ArrayXX<uint8_t>& offspring,
    double CROSS_PROB) {

#pragma omp parallel
        {
            std::mt19937 local_gen(1234 + omp_get_thread_num());  // 每个线程独立种子
            std::uniform_real_distribution<> local_prob_dist(0.0, 1.0);
            std::uniform_int_distribution<> local_bin_dist(0, 1);

#pragma omp for schedule(guided)
            for (int i = 0; i < POP_SIZE; i += 2) {
                int selected_idx1 = selected_idx[i];
                int selected_idx2 = selected_idx[i + 1];

                if (local_prob_dist(local_gen) < CROSS_PROB) {
                    for (int j = 0; j < IND_SIZE; ++j) {
                        if (local_bin_dist(local_gen)) {
                            offspring(j, i) = pop(j, selected_idx1);
                            offspring(j, i + 1) = pop(j, selected_idx2);
                        }
                        else {
                            offspring(j, i) = pop(j, selected_idx2);
                            offspring(j, i + 1) = pop(j, selected_idx1);
                        }
                    }
                }
                else {
                    offspring.col(i) = pop.col(selected_idx1);
                    offspring.col(i + 1) = pop.col(selected_idx2);
                }
            }
        }
        return 0;
}


// 变异-并行版
void mutate_omp(Eigen::ArrayXX<uint8_t>& pop) {

#pragma omp parallel
    {
        std::mt19937 local_gen(5678 + omp_get_thread_num());  // 每个线程独立 RNG
        std::uniform_real_distribution<double> local_prob_dist(0.0, 1.0);
        std::uniform_int_distribution<int> local_ind_dist(0, IND_SIZE - 1);
        std::uniform_int_distribution<int> local_gene_dist(0, GENE_MAX - 1);

#pragma omp for schedule(guided)
        for (int i = 0; i < POP_SIZE; ++i) {
            if (local_prob_dist(local_gen) < MUT_PROB) {
                std::unordered_set<int> unique_indices;

                while ((int)unique_indices.size() < MUT_POINTS) {
                    unique_indices.insert(local_ind_dist(local_gen));
                }

                for (int idx : unique_indices) {
                    pop(idx, i) = local_gene_dist(local_gen);
                }
            }
        }
    }
}


// 二分法线性插值
double interpolate_mse_binary(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& x_query,
    const std::vector<double>& y_true
) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("x and y must have the same size.");
    }
    if (x_query.size() != y_true.size()) {
        throw std::invalid_argument("x_query and y_true must have the same size.");
    }

    size_t n = x_query.size();
    double mse = 0.0;

    // 更新左边的迭代器
    auto it_left = x.begin();

    for (size_t i = 0; i < n; ++i) {
        double xq = x_query[i];
        double y_pred;

        // 边界处理
        if (xq <= x.front()) {
            y_pred = y.front();
        }
        else if (xq >= x.back()) {
            y_pred = y.back();
        }
        else {
            // 使用二分查找找第一个 >= xq 的位置
            auto it = std::lower_bound(it_left, x.end(), xq);
            size_t idx = std::distance(x.begin(), it);

            // 插值区间是 [idx - 1, idx]
            double x0 = x[idx - 1];
            double x1 = x[idx];
            double y0 = y[idx - 1];
            double y1 = y[idx];

            double t = (xq - x0) / (x1 - x0);
            y_pred = y0 + t * (y1 - y0);

            it_left = x.begin() + idx - 1;
        }

        double diff = y_pred - y_true[i];
        mse += diff * diff;
    }

    //return mse / n;
    // 之前是除n，现在放到最后除以N
    return mse;
}


// 单评价函数
double f_eval_single(const Eigen::ArrayXX<uint8_t>& pop,
    int n, // 当前个体索引
    int A,
    int B,
    const Eigen::ArrayXXd& log_k_mat,
    const Eigen::ArrayXXd& Eb_eta_mat,
    const Eigen::ArrayXXi& sort_mat,
    std::vector<std::vector<std::vector<double> > >& Eb_list_A_allthread,
    std::vector<std::vector<std::vector<double> > >& k_list_A_allthread,
    std::vector<std::vector<std::vector<double> > >& Eb_list_B_allthread,
    std::vector<std::vector<std::vector<double> > >& k_list_B_allthread,
    int N_groups  // 分组总数量 
)
{
    double Loss = 0.0;

    // 当前个体
    const Eigen::ArrayX<uint8_t>& ind = pop.col(n);

    int tid = omp_get_thread_num();
    std::vector<std::vector<double> >& Eb_list_A = Eb_list_A_allthread[tid];
    std::vector<std::vector<double> >& k_list_A = k_list_A_allthread[tid];
    std::vector<std::vector<double> >& Eb_list_B = Eb_list_B_allthread[tid];
    std::vector<std::vector<double> >& k_list_B = k_list_B_allthread[tid];

    for (int i = 0; i < N_groups; ++i) {
        Eb_list_A[i].clear();
        k_list_A[i].clear();
        Eb_list_B[i].clear();
        k_list_B[i].clear();
    }

    int group_now_A = -1; // 当前分组
    int sort_now_A = -1;  // 当前序号
    int group_now_B = -1;
    int sort_now_B = -1;
    for (int i = 0; i < IND_SIZE; ++i) {
        // 所有点遍历一遍
        sort_now_A = sort_mat(i, A);
        group_now_A = ind(sort_now_A);

        sort_now_B = sort_mat(i, B);
        group_now_B = ind(sort_now_B);

        Eb_list_A[group_now_A].push_back(Eb_eta_mat(sort_now_A, B));
        k_list_A[group_now_A].push_back(log_k_mat(sort_now_A, B));



        Eb_list_B[group_now_B].push_back(Eb_eta_mat(sort_now_B, B));
        k_list_B[group_now_B].push_back(log_k_mat(sort_now_B, B));
    }

    for (int i = 0; i < N_groups; ++i) {
        std::vector<double> Eb_cumsum_A(Eb_list_A[i].size());
        //std::partial_sum(Eb_list_A[i].begin(), Eb_list_A[i].end(), Eb_cumsum_A.begin());
        std::inclusive_scan(Eb_list_A[i].begin(), Eb_list_A[i].end(), Eb_cumsum_A.begin());

        

        std::vector<double> Eb_cumsum_B(Eb_list_B[i].size());
        //std::partial_sum(Eb_list_B[i].begin(), Eb_list_B[i].end(), Eb_cumsum_B.begin());
        std::inclusive_scan(Eb_list_B[i].begin(), Eb_list_B[i].end(), Eb_cumsum_B.begin());

        Loss += interpolate_mse_binary(Eb_cumsum_B, k_list_B[i], Eb_cumsum_A, k_list_A[i]);
    }

    return Loss / IND_SIZE;
}


// 生成两两组合
std::vector<std::pair<int, int>> generate_combinations(const std::vector<int>& values) {
    
    std::vector<std::pair<int, int>> combinations;

    int n = values.size();
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            combinations.emplace_back(values[i], values[j]);
        }
    }
    return combinations;
}

std::vector<int> generate_index_list() {
    std::vector<int> idx_list_45(45);
    for (int i = 0; i < 45; ++i) {
        idx_list_45[i] = i;
    }
    return idx_list_45;
}

// 函数用于找出 vector 中最小值的索引
int findMinIndex(const std::vector<double>& vec) {
    if (vec.empty()) {
        return -1;
    }
    auto minIt = std::min_element(vec.begin(), vec.end());
    return std::distance(vec.begin(), minIt);
}


// 带权重的抽样A和B
int sample_combination(
    const std::vector<double>& weights,
    const std::vector<std::pair<int, int>>& combinations)
{
    // 检查一致性
    if (weights.empty() || weights.size() != combinations.size()) {
        throw std::runtime_error("weights and combinations size mismatch");
    }

    // 1. 归一化权重
    double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (sum_weights <= 0.0) {
        throw std::runtime_error("Sum of weights must be positive");
    }

    std::vector<double> norm_weights(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        norm_weights[i] = weights[i] / sum_weights;
    }

    // 2. 构造离散分布
    std::discrete_distribution<> dist(norm_weights.begin(), norm_weights.end());

    // 3. 抽样一个索引
    int comb_idx = dist(gen);

    // 4. 返回对应组合
    return comb_idx;
}

