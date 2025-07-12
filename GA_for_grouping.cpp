#define _CRT_SECURE_NO_WARNINGS



#include "fn.hpp"



// ==== 主遗传算法 ====
int run_ga(int threadCount) {

    // 设置 OpenMP 线程数
    omp_set_num_threads(threadCount);

    // 读取数据
    // double
    Eigen::ArrayXXd log_k_mat = read_mat<double>("..\\data\\log_k_mat", IND_SIZE, 150);
    Eigen::ArrayXXd Eb_eta_mat = read_mat<double>("..\\data\\Eb_eta_mat", IND_SIZE, 150);
    // int
    Eigen::ArrayXXi sort_mat = read_mat<int>("..\\data\\sort_mat", IND_SIZE, 150);


    // 生成两两组合
    auto combinations = generate_combinations(values);
    printf("combinations_size = %d\n", combinations.size());

    // 45种组合下的精英索引
    constexpr int comb_num = 45;
    std::vector<int> idx_list_45(comb_num);
    for (int i = 0; i < comb_num; ++i) {
        idx_list_45[i] = i;
    }



    printf("初始化种群\n");
    Eigen::ArrayXX<uint8_t> pop = init_pop_omp(POP_SIZE, IND_SIZE, GENE_MAX);
    std::vector<double> fitness(POP_SIZE);

    Eigen::ArrayXX<uint8_t> offspring(IND_SIZE, POP_SIZE);


    // 预先开辟内存，计算会用到
    std::vector<std::vector<std::vector<double> > > Eb_list_A_allthreads(threadCount, std::vector<std::vector<double> >(GENE_MAX, std::vector<double>{}));  // (线程数，分组数，组成员数)
    std::vector<std::vector<std::vector<double> > >  k_list_A_allthreads(threadCount, std::vector<std::vector<double> >(GENE_MAX, std::vector<double>{}));
    std::vector<std::vector<std::vector<double> > >  Eb_list_B_allthreads(threadCount, std::vector<std::vector<double> >(GENE_MAX, std::vector<double>{}));
    std::vector<std::vector<std::vector<double> > >  k_list_B_allthreads(threadCount, std::vector<std::vector<double> >(GENE_MAX, std::vector<double>{}));
    // 预先分配防止频繁拷贝
    for (int i = 0; i < threadCount; ++i) {
        for (int j = 0; j < GENE_MAX; ++j) {
            Eb_list_A_allthreads[i][j].reserve(group_members_num_est);
            k_list_A_allthreads[i][j].reserve(group_members_num_est);
            Eb_list_B_allthreads[i][j].reserve(group_members_num_est);
            k_list_B_allthreads[i][j].reserve(group_members_num_est);
        }
    }


    printf("初始化精英\n");
    Eigen::ArrayXX<uint8_t> elites(IND_SIZE, comb_num + 1);
    for (int i = 0; i < comb_num + 1; ++i) {
        for (int j = 0; j < IND_SIZE; ++j) {
            elites(j, i) = static_cast<uint8_t>(gene_dist(gen));
        }
    }
    std::vector<double> elites_fitness(comb_num + 1, 10000.0);


    printf("第0代评估\n");
    auto state_pair = combinations[state_dist(gen)];
    //printf("--");

    // 单评估
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < POP_SIZE; ++i) {
        fitness[i] = f_eval_single(pop, i, state_pair.first, state_pair.second, log_k_mat, Eb_eta_mat, sort_mat, Eb_list_A_allthreads, k_list_A_allthreads, Eb_list_B_allthreads, k_list_B_allthreads, GENE_MAX);
        //printf("%g, ", fitness[i]);
    }

    int best_idx = findMinIndex(fitness);
    auto best_ind = pop.col(best_idx);
    double best_val = fitness[best_idx];
    double full_val = 0;

    std::pair<int, int> state_pair_full_0;
    // 全评估
    std::vector<double> weights(comb_num);
    for (int i = 0; i < comb_num; ++i) {
        state_pair_full_0 = combinations[i];
        weights[i] = f_eval_single(pop, best_idx, state_pair_full_0.first, state_pair_full_0.second, log_k_mat, Eb_eta_mat, sort_mat, Eb_list_A_allthreads, k_list_A_allthreads, Eb_list_B_allthreads, k_list_B_allthreads, GENE_MAX);
        full_val += weights[i];
    }
    printf("%d\t%d-%d\tf_single = %.6f\tf_full = %.6f\n", 0, state_pair.first, state_pair.second, best_val, full_val);


    char log_name[128];
    sprintf(log_name, "log_TS=%d_N=%d.txt", TOURN_SIZE, POP_SIZE);
    FILE* fp = fopen(log_name, "w");
    fprintf(fp, "%d\t%d-%d\t%.4f\t%.4f\t%.4f\n", 0, state_pair.first, state_pair.second, best_val, full_val, full_val);
    fclose(fp);

    printf("开始迭代\n");

    std::vector<int> selected_idx(POP_SIZE);

    // 全局最好值
    double best_full_alltime = 1e10;
    // 全局最好值有多久没变了
    int no_progress_times = 0;
    // 结束循环标志
    int end_sgn = 0;

    double time0, time1;
    for (int generation = 1; generation < GENERATIONS; ++generation) {

        time0 = omp_get_wtime();

        // 最佳值长期没有改善，结束循环
        if (end_sgn)
            break;

        // 锦标赛选择
        tournament_selection_omp(selected_idx, pop, fitness, 2);

        // 交叉和变异
        crossover_omp(selected_idx, pop, offspring, CROSS_PROB);
        mutate_omp(offspring);
        
        // 种群复制为新生成子代
        pop.swap(offspring);

        // 前 46 精英复制
        for (int i = 0; i < comb_num + 1; ++i) {
            pop.col(i) = elites.col(i);
            fitness[i] = elites_fitness[i];
        }

        // 按 loss 权重抽取 state_pair
        int idx_comb = sample_combination(weights, combinations);
        state_pair = combinations[idx_comb];

        // 计算 fitness
#pragma omp parallel for schedule(guided)
        for (int i = 0; i < POP_SIZE; ++i) {
            fitness[i] = f_eval_single(pop, i, state_pair.first, state_pair.second, log_k_mat, Eb_eta_mat, sort_mat, Eb_list_A_allthreads, k_list_A_allthreads, Eb_list_B_allthreads, k_list_B_allthreads, GENE_MAX);
        }


        // 检查结果
        best_idx = findMinIndex(fitness);
        best_ind = pop.col(best_idx);
        best_val = fitness[best_idx];
        full_val = 0;
        // 计算 Best 的全评价
#pragma omp parallel for schedule(guided) reduction(+:full_val)
        for (int i = 0; i < comb_num; ++i) {
            std::pair<int, int> state_pair_full = combinations[i];
            weights[i] = f_eval_single(pop, best_idx, state_pair_full.first, state_pair_full.second, log_k_mat, Eb_eta_mat, sort_mat, Eb_list_A_allthreads, k_list_A_allthreads, Eb_list_B_allthreads, k_list_B_allthreads, GENE_MAX);
            full_val += weights[i];
        }

        // 更新精英
        // single 比较
        if (best_val < elites_fitness[idx_comb]) {
            elites.col(idx_comb) = pop.col(best_idx);
            elites_fitness[idx_comb] = best_val;
        }
        // full 比较
        if (full_val < elites_fitness[comb_num]) {
            elites.col(comb_num) = pop.col(best_idx);
            elites_fitness[comb_num] = full_val;
        }

        // 如果全局最小值更新
        if (full_val < best_full_alltime) {
            best_full_alltime = full_val;
            no_progress_times = 0;
        }
        else {
            no_progress_times++;
        }
        // 100代都没有更新最好值了，结束
        if (no_progress_times > 100)
            end_sgn = 1;

        printf("%d\t%d-%d\tf_single = %.6f\tf_full = %.6f\t", generation, state_pair.first, state_pair.second, best_val, full_val);
        fp = fopen(log_name, "a");
        fprintf(fp, "%d\t%d-%d\t%.4f\t%.4f\t%.4f\n", generation, state_pair.first, state_pair.second, best_val, full_val, best_full_alltime);
        fclose(fp);


        // 保存
        if (generation % 100 == 0) {
            char file_name[128];
            sprintf(file_name, "res_TS=%d_N=%d_%dgen_%.4f.txt", TOURN_SIZE, POP_SIZE, generation, elites_fitness[comb_num]);
            FILE* fp_best = fopen(file_name, "w");
            int cnt = 0;
            for (int k = 0; k < IND_SIZE; ++k) {
                fprintf(fp_best, "%d ", elites(k, comb_num));
                cnt++;
                if (cnt % 32 == 0)
                    fprintf(fp, "\n");
            }
            fclose(fp_best);

        }

        // 输出每一步时间
        time1 = omp_get_wtime() - time0;
        printf("%.2fs\n", time1);

    }

    return 0;

}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "用法: " << argv[0] << " <线程数量>" << std::endl;
        return 1;
    }

    // 将命令行参数转换为整数
    int threadCount = std::atoi(argv[1]);

    if (threadCount <= 0) {
        std::cerr << "线程数量必须是正整数。" << std::endl;
        return 1;
    }

    run_ga(threadCount);
    return 0;
}

