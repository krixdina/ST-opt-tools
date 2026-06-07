/*
    MIT License

    Copyright (c) 2025 Senming Tan (senmingtan5@gmail.com)
    Copyright (c) 2025 Deping Zhang (beiyuena@foxmail.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef TRAJ_OPT_HPP
#define TRAJ_OPT_HPP

#include "SplineTrajectory.hpp"
#include "grid_map.hpp"
#include <lbfgs.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>

namespace TrajOpt {

struct TrajectoryParams {
    double total_len = 10;
    double total_time = 10;
    double piece_len = 4;
    double rho_v = 10000;           // Velocity penalty weight
    double rho_collision = 100000;  // Collision penalty weight
    double rho_T = 100;             // Time penalty weight
    double rho_energy = 100;        // Energy (smoothness) penalty weight
    double max_v = 1.0;             // Maximum velocity
    double safe_threshold = 0.5;    // Safety distance threshold
    
    int int_K = 32;                 // Integration sample points
    int mem_size = 256;             // L-BFGS memory size
    int past = 3;                   // L-BFGS parameter
    double g_epsilon = 1e-6;        // Gradient convergence threshold
    double min_step = 1e-32;        // Minimum step size
    double delta = 1e-5;            // Function change convergence threshold
    int max_iter = 10000;           // Maximum iterations
};

class TrajectoryOptimizer {
public:
    using Vector2d = Eigen::Vector2d;
    using Vector3d = Eigen::Vector3d;
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    using PPoly2D = SplineTrajectory::PPolyND<2>;
    using CubicSpline2D = SplineTrajectory::CubicSpline2D;

    TrajectoryOptimizer(std::shared_ptr<grid_map::GridMap> map,
                       const std::vector<Eigen::Vector2d>& astar_path,
                       const TrajectoryParams& params = TrajectoryParams())
        : map_(map), astar_path_(astar_path), params_(params), in_opt_(false) 
    {
        preprocessAstarPath();
    }

    /*
     * 作用：将 astar_path_（即前端 A* 规划后得到的二维离散路径）整理成后端样条优化所需的初始条件，
     * 具体包括起点与终点的边界状态、沿路径按弧长重采样得到的中间控制点，以及初始总时间，
     * 然后调用后续优化流程生成连续轨迹。

     * 输入：本函数没有显式形参；它依赖 astar_path_（即待优化的前端路径点序列），
     * map_（即用于后续碰撞代价计算的地图环境），以及 params_（即总长度、总时间、期望分段长度和优化权重等参数配置）。

     * 输出：返回值是一个 bool，用来向调用方报告 optimizeSE2Traj()（即真正执行样条轨迹优化的内部流程）的返回结果；
     * 同时该函数还会为后续优化写入初始边界条件、中间控制点，并在优化成功时更新 trajectory_（即类内保存的最终连续轨迹）。
     */
    bool plan() {
        if (astar_path_.size() < 2) return false;
        
        // 根据离散路径的首尾两段方向，构造样条起点和终点的边界条件：
        // 第一列是起点/终点位置，第二列是沿路径切向设置的初始边界速度。
        Eigen::Matrix2d init_cond, end_cond;
        init_cond.col(0) = astar_path_.front();
        end_cond.col(0) = astar_path_.back();
        init_cond.col(1) = (astar_path_[1] - astar_path_[0]).normalized() * 0.1;
        end_cond.col(1) = (astar_path_.back() - astar_path_[astar_path_.size()-2]).normalized() * 0.1;

        // 根据总路径长度和期望的单段长度，决定样条需要分成多少段；
        // 若计算出的段数过少，则至少保留两段，并反向修正每段的实际平均长度。
        double total_len = params_.total_len;
        int piece_num = (int)(total_len / params_.piece_len);
        if (piece_num < 2) piece_num = 2;
        params_.piece_len = total_len / piece_num;
        
        // 为后端样条准备中间控制点：
        // 这里不是直接使用全部 A* 路径点，而是按总弧长均匀重采样，
        // 从而得到数量更少、分布更均匀的中间约束点。
        Eigen::MatrixXd inner_pos(2, piece_num-1);
        std::vector<Eigen::Vector2d> inner_pos_node;
        
        double step_len = total_len / piece_num;
        double accumulated_len = 0.0;
        
        for (int i = 1; i < piece_num; ++i) {
            double target_len = i * step_len;
            // 沿当前离散折线路径累计长度，找到目标弧长落入的那一小段；
            // 一旦找到，就在该线段内部做线性插值，得到对应的中间控制点。
            while (accumulated_len < target_len && current_segment_ < astar_path_.size() - 1) {
                double seg_len = (astar_path_[current_segment_ + 1] - astar_path_[current_segment_]).norm();
                if (accumulated_len + seg_len >= target_len) {
                    double ratio = (target_len - accumulated_len) / seg_len;
                    Eigen::Vector2d point = astar_path_[current_segment_] + 
                                          ratio * (astar_path_[current_segment_ + 1] - astar_path_[current_segment_]);
                    inner_pos_node.push_back(point);
                    break;
                }
                accumulated_len += seg_len;
                current_segment_++;
            }
        }
        
        // 将按弧长重采样得到的中间控制点，整理成 2×N 的矩阵形式，
        // 以便后续作为样条内部路径点的优化变量初值传入。
        inner_pos.resize(2, inner_pos_node.size());
        for (size_t i = 0; i < inner_pos_node.size(); i++) {
            inner_pos.col(i) = inner_pos_node[i];
        }

        // 取前端给出的总时间初值，并连同边界条件和中间控制点一起送入真正的后端优化流程。
        double total_time = params_.total_time;
        int result = optimizeSE2Traj(init_cond, inner_pos, end_cond, total_time);
        return result;
    }

    PPoly2D getOptimizedTrajectory() const { return trajectory_; }

    struct TrajectoryMetrics {
        double max_velocity;
        double min_clearance;
        double total_time;
        double trajectory_energy;
        double path_deviation;
    };
    
    TrajectoryMetrics evaluateTrajectory() const {
        TrajectoryMetrics metrics;
        metrics.max_velocity = 0.0;
        metrics.min_clearance = std::numeric_limits<double>::max();
        metrics.total_time = trajectory_.getDuration();
        metrics.trajectory_energy = 0.0;
        metrics.path_deviation = 0.0;

        if (!trajectory_.isInitialized()) return metrics;

        const double dt = 0.01;
        int sample_count = 0;
        for (double t = 0.0; t < metrics.total_time; t += dt) {
            Vector2d pos = trajectory_.evaluate(t, 0);
            Vector2d vel = trajectory_.evaluate(t, 1);
            Vector2d acc = trajectory_.evaluate(t, 2);
            
            metrics.trajectory_energy += acc.squaredNorm() * dt;
            metrics.max_velocity = std::max(metrics.max_velocity, vel.norm());
            metrics.min_clearance = std::min(metrics.min_clearance, map_->getDistance(pos));
            
            double min_dist = std::numeric_limits<double>::max();
            for (const auto& astar_pt : astar_path_) {
                min_dist = std::min(min_dist, (pos - astar_pt).norm());
            }
            metrics.path_deviation += min_dist;
            sample_count++;
        }
        
        if (sample_count > 0) {
            metrics.path_deviation /= sample_count;
        }
        
        return metrics;
    }

    std::vector<Eigen::Vector2d> sampleTrajectory(double dt = 0.1) const {
        std::vector<Eigen::Vector2d> path;
        if (!trajectory_.isInitialized()) return path;
        
        const double total_time = trajectory_.getDuration();
        for (double t = 0.0; t <= total_time; t += dt) {
            path.push_back(trajectory_.evaluate(t, 0));
        }
        
        if (!path.empty() && path.back() != trajectory_.evaluate(total_time, 0)) {
            path.push_back(trajectory_.evaluate(total_time, 0));
        }
        
        return path;
    }

private:
    void preprocessAstarPath() {
        // Prepare A* path for fast nearest neighbor queries
    }

    /*
     * 作用：将上一层准备好的边界条件、中间控制点和总时间初值组织成一个连续轨迹优化问题，
     * 先生成一条初始三次样条轨迹并评估其质量，再调用 L-BFGS 对路径中间点和总时间参数进行联合优化。

     * 输入：initPos 表示轨迹起点的边界条件，其中第一列是起点位置、第二列是起点切向速度；
     * innerPtsPos 表示位于起点与终点之间的中间控制点矩阵，每一列都是一个二维路径点；
     * endPos 表示轨迹终点的边界条件，其中第一列是终点位置、第二列是终点切向速度；
     * totalTime 表示整条轨迹的总时间初值，后续会被映射成统一分配到各段的初始时间长度。

     * 输出：返回值是 L-BFGS 求解器的状态码，其中非负值通常表示收敛、满足停止条件或被取消，
     * 负值表示参数非法或搜索失败等错误；同时该函数会更新类内保存的起终边界条件、段数信息、
     * 当前样条对象以及最终优化后的连续轨迹。
     */
    int optimizeSE2Traj(const MatrixXd& initPos, const MatrixXd& innerPtsPos,
                        const MatrixXd& endPos, double totalTime) {
        in_opt_ = true;
        piece_pos_ = innerPtsPos.cols() + 1;

        // 保存本次优化使用的起点与终点边界条件，
        // 便于代价函数在每次迭代重建轨迹时复用同一组边界约束。
        init_pos_ = initPos;
        end_pos_ = endPos;

        // 构造优化变量的维度：
        // 前 1 个标量表示总时间的参数化变量，
        // 后面的 2*(段数-1) 个标量表示所有中间控制点的二维坐标。
        int variable_num = 2 * (piece_pos_ - 1) + 1;
        dim_T = 1;

        Eigen::VectorXd x;
        x.resize(variable_num);

        // 将优化向量拆成两部分视图：
        // tau 表示总时间的单变量参数化形式，
        // Ppos 表示中间控制点矩阵，它直接共享优化向量中的后续内存。
        double& tau = x(0);
        Eigen::Map<Eigen::MatrixXd> Ppos(x.data() + dim_T, 2, piece_pos_ - 1);
        
        // 用调用方给出的总时间初值来初始化时间变量，
        // 并直接把重采样得到的中间控制点作为空间变量初值。
        tau = logC2(totalTime);
        Ppos = innerPtsPos;

        // 根据当前总时间参数生成每一段的初始时间长度。
        // 这里采用的是“所有段统一平均分配时间”的方式，而不是为每一段单独设置时间变量。
        Eigen::VectorXd Tpos;
        Tpos.resize(piece_pos_);
        calTfromTauUni(tau, Tpos);

        // 用当前的边界条件、中间控制点和分段时间先生成一条初始样条轨迹，
        // 后续优化与评估都围绕这条连续轨迹展开。
        generateTrajectory(initPos, endPos, Ppos, Tpos);

        // 在真正进入迭代之前，先输出初始轨迹的速度、最小障碍距离和相对原始路径的偏差，
        // 便于观察初值质量。
        auto metrics = evaluateTrajectory();
        std::cout << "Initial Trajectory:" << std::endl;
        std::cout << "Max velocity: " << metrics.max_velocity << " m/s" << std::endl;
        std::cout << "Min clearance: " << metrics.min_clearance << " m" << std::endl;
        std::cout << "Path deviation: " << metrics.path_deviation << " m" << std::endl;

        // 将本类的参数配置拷贝到 L-BFGS 求解器参数中，
        // 这些参数决定了有限记忆长度、收敛阈值、最小步长和最大迭代次数等优化行为。
        lbfgs::lbfgs_parameter_t lbfgs_params;
        lbfgs_params.mem_size = params_.mem_size;
        lbfgs_params.past = params_.past;
        lbfgs_params.g_epsilon = params_.g_epsilon;
        lbfgs_params.min_step = params_.min_step;
        lbfgs_params.delta = params_.delta;
        lbfgs_params.max_iterations = params_.max_iter;

        // 调用 L-BFGS 开始真正的数值优化：
        // 优化变量就是向量 x，其中既包含总时间参数，也包含中间控制点；
        // 代价函数与梯度由当前对象的 costFunction() 负责计算。
        double final_cost;
        int result = lbfgs::lbfgs_optimize(
            x, final_cost,
            [](void* instance, const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
                return static_cast<TrajectoryOptimizer*>(instance)->costFunction(x, grad);
            },
            nullptr, nullptr, this, lbfgs_params);

        // 输出本次优化结束后的状态码和最终代价值，便于调试求解是否正常结束。
        std::cout << "Optimization finished with result: " << result << std::endl;
        std::cout << "Final cost: " << final_cost << std::endl;

        // 清理“正在优化”标志，并将求解器状态码返回给调用方。
        in_opt_ = false;
        return result;
    }

    /*
     * 作用：作为 L-BFGS 的目标函数回调，在当前优化变量给定时重建一条连续样条轨迹，
     * 计算总代价值，并同时给出总代价对所有优化变量的梯度。

     * 输入：x 表示当前迭代的优化变量向量，其中第一个分量是总时间的参数化变量，
     * 后续分量是所有中间控制点的二维坐标；grad 表示由本函数回填的梯度向量，
     * 其布局与 x 完全一致，用于告诉优化器每个变量朝哪个方向下降总代价最快。

     * 输出：返回值是当前优化变量对应的总代价值，包含约束代价、平滑能量代价和总时间代价；
     * 同时本函数会把 grad 写满，使外层 L-BFGS 可以继续更新总时间参数和中间控制点。
     */
    double costFunction(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
        double cost = 0.0;

        // 将优化向量拆成两部分：
        // tau 表示总时间的单变量参数化形式，
        // Ppos 表示所有中间控制点的二维坐标；
        // grad_tau 和 gradPpos 则分别对应这两部分变量的梯度存储位置。
        const double& tau = x(0);
        double& grad_tau = grad(0);
        Eigen::Map<const Eigen::MatrixXd> Ppos(x.data() + dim_T, 2, piece_pos_ - 1);
        Eigen::Map<Eigen::MatrixXd> gradPpos(grad.data() + dim_T, 2, piece_pos_ - 1);
     
        // 先由当前时间参数恢复出每一段的时间长度，
        // 再用固定的起终边界条件和当前中间控制点重建出一条连续样条轨迹。
        Eigen::VectorXd Tpos;
        Tpos.resize(piece_pos_);
        calTfromTauUni(tau, Tpos);
        generateTrajectory(init_pos_, end_pos_, Ppos, Tpos);

        // 计算与环境和运动约束相关的代价：
        // 包括速度超限代价、碰撞距离代价，以及它们对样条系数和分段时间的梯度。
        double constrain_cost = 0.0;
        Eigen::MatrixXd gdCpos_constrain;
        Eigen::VectorXd gdTpos_constrain;
        calculateConstraintCostGrad(trajectory_, constrain_cost, gdCpos_constrain, gdTpos_constrain);
        
        // 将“对样条多项式系数和分段时间的梯度”
        // 反向传播成“对中间控制点和分段时间的梯度”，
        // 这样外层优化器才能直接更新真正的空间变量。
        Eigen::MatrixXd gradPpos_constrain;
        Eigen::VectorXd gradTpos_constrain;
        calGradCTtoQT(gdCpos_constrain, gdTpos_constrain, gradPpos_constrain, gradTpos_constrain);
        
        // 从样条对象中读取平滑能量，
        // 这里的能量本质上衡量轨迹的弯曲和加速度变化程度。
        double energy = cubic_spline_.getEnergy();
        double energy_cost = params_.rho_energy * energy;
        
        // 读取平滑能量对中间控制点和分段时间的梯度，
        // 为后续与约束项梯度相加做准备。
        CubicSpline2D::MatrixType gradP_energy = cubic_spline_.getEnergyGradInnerP();
        Eigen::VectorXd gradT_energy = cubic_spline_.getEnergyGradTimes();
        
        // 将约束项和平滑项在空间变量上的梯度合并，
        // 得到总代价对所有中间控制点坐标的梯度。
        gradPpos = gradPpos_constrain + params_.rho_energy * gradP_energy.transpose();
        Eigen::VectorXd gradTpos_total = gradTpos_constrain + params_.rho_energy * gradT_energy;

        // 显式加入总时间惩罚项，使轨迹不会无约束地通过延长时间来降低其他代价。
        // 由于本实现只优化一个总时间参数，而不是逐段独立时间，
        // 所以先把所有分段时间梯度汇总，再通过链式法则映射到 tau 上。
        double tau_cost = params_.rho_T * expC2(tau);
        double grad_Tsum = params_.rho_T + gradTpos_total.sum() / piece_pos_;
        grad_tau = grad_Tsum * getTtoTauGrad(tau);

        // 总代价由约束项、平滑项和总时间项三部分组成，并返回给外层 L-BFGS。
        cost = constrain_cost + energy_cost + tau_cost;
        return cost;
    }

    /*
     * 作用：沿当前连续样条轨迹做数值积分，计算与运动约束和环境约束相关的总代价，
     * 具体包括速度超限代价和与障碍物距离不足的碰撞代价；
     * 同时给出这些约束代价对样条多项式系数和各段时间长度的梯度。

     * 输入：traj 表示当前已经生成好的二维分段多项式轨迹，它提供每一段的时间范围和多项式系数；
     * cost 表示由本函数回填的约束总代价；reload
     * gdCpos 表示由本函数回填的“代价对样条多项式系数的梯度矩阵”；
     * gdTpos 表示由本函数回填的“代价对每一段时间长度的梯度向量”。

     * 输出：本函数没有显式返回值；
     * 它会修改 cost、gdCpos 和 gdTpos，使调用方能够将约束项并入总代价，
     * 并继续把梯度从样条系数空间反传回中间控制点和总时间变量。
     */
    void calculateConstraintCostGrad(
        PPoly2D& traj,
        double& cost,
        Eigen::MatrixXd& gdCpos,
        Eigen::VectorXd& gdTpos)
    {
        cost = 0.0;
        double v_cost = 0.0;
        double occ_cost = 0.0;
        double path_cost = 0.0;
        
        // 根据当前轨迹的分段数量初始化梯度存储：
        // 每一段三次多项式在二维平面上有 4 组系数，因此系数梯度矩阵大小为 4*N 行、2 列；
        // 每一段时间长度对应一个时间梯度，因此时间梯度向量大小为 N。
        const int N = traj.getNumSegments();
        gdCpos.resize(4 * N, 2);
        gdCpos.setZero();
        gdTpos.resize(N);
        gdTpos.setZero();

        // 读取轨迹的分段时间节点和多项式系数。
        // breaks 表示每一段的起止时间，coeffs 表示每一段二维三次多项式的 4 组系数。
        const auto& breaks = traj.getBreakpoints();
        const MatrixXd& coeffs = traj.getCoefficients();

        Eigen::Vector2d pos, vel, acc;
        double grad_time = 0.0;
        Eigen::Vector2d grad_p = Eigen::Vector2d::Zero();
        Eigen::Vector2d grad_v = Eigen::Vector2d::Zero();
        Eigen::Vector2d grad_sdf = Eigen::Vector2d::Zero();
        Eigen::Matrix<double, 4, 1> beta0, beta1, beta2;
        double s1, s2, s3;
        double step, alpha, omg;

        // 对轨迹的每一段分别进行积分近似。
        // 外层循环遍历样条段，内层循环在该段内部按固定数量采样，
        // 把连续时间上的约束代价近似为离散采样点上的加权求和。
        for (int i = 0; i < N; ++i) {
            const Eigen::Matrix<double, 4, 2>& c = coeffs.block<4, 2>(i * 4, 0);
            step = (breaks[i+1] - breaks[i]) / params_.int_K;
            s1 = 0.0;

            for (int j = 0; j <= params_.int_K; ++j) {
                // alpha 表示当前采样点在本段时间内的归一化位置；
                // grad_p、grad_v 和 grad_time 分别累计当前位置、当前速度和当前段时间的局部梯度贡献。
                alpha = 1.0 / params_.int_K * j;
                grad_p.setZero();
                grad_v.setZero();
                grad_sdf.setZero();
                grad_time = 0.0;

                // 利用三次多项式基函数及其一阶、二阶导数，
                // 从当前段的系数中恢复出该采样点的当前位置、速度和加速度。
                s2 = s1 * s1;
                s3 = s2 * s1;
                beta0 << 1.0, s1, s2, s3;
                beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2;
                beta2 << 0.0, 0.0, 2.0, 6.0 * s1;
                pos = c.transpose() * beta0;
                vel = c.transpose() * beta1;
                acc = c.transpose() * beta2;

                // 采用梯形积分权重：
                // 段首和段尾采样点权重减半，中间采样点权重为 1。
                omg = (j == 0 || j == params_.int_K) ? 0.5 : 1.0;

                // 1. Velocity constraint
                // 若当前速度平方超过允许的最大速度平方，
                // 则对超限量施加三次罚函数，使速度超限越大时惩罚增长越快。
                double vxy_snorm = vel.squaredNorm();
                double vViola = vxy_snorm - params_.max_v * params_.max_v;
                if (vViola > 0) {
                    grad_v += params_.rho_v * 6 * vViola * vViola * vel;
                    double cost_v = params_.rho_v * vViola * vViola * vViola;
                    cost += cost_v * omg * step;
                    v_cost += cost_v * omg * step;
                    grad_time += omg * (cost_v / params_.int_K + step * alpha * grad_v.dot(acc));
                }

                // 2. Collision constraint
                // 查询当前采样点到障碍物的有符号距离和距离梯度。
                // 当距离小于安全阈值时，说明轨迹过于靠近障碍物，需要加入避障惩罚。
                double sdf_value;
                map_->getDistanceAndGradient(pos, sdf_value, grad_sdf);
                double cViola = params_.safe_threshold - sdf_value;
                if (cViola > 0 && sdf_value < 5) {
                    double penalty;
                    Eigen::Vector2d grad_pc;
                    
                    // 对轻微碰撞风险使用二次惩罚，使代价和梯度更平滑；
                    // 对更严重的碰撞风险使用线性惩罚，避免惩罚值增长过快导致数值不稳定。
                    if (cViola < 0.1) {
                        penalty = cViola * cViola;
                        grad_pc = -2.0 * cViola * grad_sdf;
                    } else {
                        penalty = cViola;
                        grad_pc = -grad_sdf;
                    }
                    
                    double cost_c = params_.rho_collision * penalty;
                    Eigen::Vector2d grad_pc_scaled = params_.rho_collision * grad_pc;
                    
                    cost += cost_c * omg * step;
                    occ_cost += cost_c * omg * step;
                    grad_time += omg * (cost_c / params_.int_K + step * alpha * grad_pc_scaled.dot(vel));
                    grad_p += grad_pc_scaled;
                }

                // 将当前采样点对位置和速度产生的局部梯度，
                // 投影回当前段的三次多项式系数；
                // 同时累计当前采样点对本段时间长度的梯度贡献。
                gdCpos.block<4, 2>(i * 4, 0) += 
                    (beta0 * grad_p.transpose() + beta1 * grad_v.transpose()) * omg * step;
                gdTpos(i) += grad_time;
                
                // 前进到当前段中的下一个积分采样位置。
                s1 += step;
            }
        }

        // 输出速度约束与碰撞约束的代价分解，便于观察哪一类约束主导了当前轨迹的总惩罚。
        std::cout << "Cost breakdown - Vel: " << v_cost << ", Coll: " << occ_cost << std::endl;
    }

    /*
     * 作用：把约束项或其他代价项对样条多项式系数、分段时间的梯度，
     * 转换成对真正优化变量的梯度；
     * 在本项目里，真正被 L-BFGS 更新的是中间控制点坐标和时间参数，
     * 因此这里承担了从样条表示空间回到优化变量空间的梯度传播工作。

     * 输入：gdCpos 表示代价对二维三次样条各段多项式系数的梯度；
     * gdTpos 表示代价对各段时间长度的梯度；
     * gradPpos 表示由本函数回填的“代价对中间控制点坐标的梯度矩阵”；
     * gradTpos_out 表示由本函数回填的“代价对各段时间长度的梯度向量”，
     * 它是经过样条内部关系传播后的结果。

     * 输出：本函数没有显式返回值；
     * 它会修改 gradPpos 和 gradTpos_out，使调用方能够继续将这些梯度并入总梯度，
     * 供外层优化器更新空间变量和时间变量。
     */
    void calGradCTtoQT(
        const Eigen::MatrixXd& gdCpos,
        const Eigen::VectorXd& gdTpos,
        Eigen::MatrixXd& gradPpos,
        Eigen::VectorXd& gradTpos_out)
    {
        // 将通用的 Eigen 矩阵转换成样条库内部使用的矩阵类型，
        // 以便调用样条库提供的梯度传播接口。
        CubicSpline2D::MatrixType gdC_typed = gdCpos;
        CubicSpline2D::MatrixType gradByPoints;
        Eigen::VectorXd gradByTimes;
        
        // 调用样条库内部的梯度传播过程：
        // 输入是“对多项式系数和分段时间的梯度”，
        // 输出是“对路径点和分段时间的梯度”。
        cubic_spline_.propagateGrad(gdC_typed, gdTpos, gradByPoints, gradByTimes);
        
        // 样条库返回的路径点梯度是“每个路径点一行”，
        // 而本优化器内部使用的是“每个路径点一列”的布局，因此这里转置后再输出。
        gradPpos = gradByPoints.transpose();
        gradTpos_out = gradByTimes;
        
        // 输出梯度范数，便于调试时观察当前梯度规模是否异常。
        std::cout << "Gradient norm - Ppos: " << gradPpos.norm() 
                << ", Tpos: " << gradTpos_out.norm() << std::endl;
    }

    /*
     * 作用：根据起点边界条件、终点边界条件、中间控制点和每一段的时间长度，
     * 生成当前迭代对应的一条二维三次样条轨迹，并写入类内的样条对象与轨迹对象。

     * 输入：initPos 表示起点边界条件矩阵，其中第一列是起点位置、第二列是起点速度；
     * endPos 表示终点边界条件矩阵，其中第一列是终点位置、第二列是终点速度；
     * innerPts 表示位于起点与终点之间的中间控制点矩阵，每一列都是一个二维路径点；
     * Tpos 表示每一段样条对应的时间长度向量。
     
     * 输出：本函数没有显式返回值；
     * 它会更新 cubic_spline_（即类内保存的三次样条对象）和 trajectory_（即类内保存的连续分段多项式轨迹），
     * 供后续代价评估、梯度计算与轨迹采样直接使用。
     */
    void generateTrajectory(const MatrixXd& initPos, const MatrixXd& endPos,
                          const MatrixXd& innerPts, Eigen::VectorXd Tpos) {
        // 先将“每一段持续多久”转换成“每个路径点对应的绝对时间节点”。
        // 样条库更新接口需要的是时间节点序列，而不是仅有的分段时长。
        std::vector<double> times;
        times.reserve(piece_pos_ + 1);
        
        double t = 0;
        for (int i = 0; i < Tpos.size(); ++i) {
            times.push_back(t);
            t += Tpos(i);
        }
        times.push_back(t);

        // 依次拼装样条经过的空间路径点：
        // 包括起点、中间控制点以及终点，
        // 这些点共同定义了当前轨迹的几何形状。
        SplineTrajectory::SplineVector2D waypoints;
        waypoints.push_back(initPos.col(0));
        for (int i = 0; i < innerPts.cols(); ++i) {
            waypoints.push_back(innerPts.col(i));
        }
        waypoints.push_back(endPos.col(0));

        // 设置样条的起点和终点边界速度，
        // 使生成的三次样条不仅经过给定路径点，还满足首末端的切向约束。
        SplineTrajectory::BoundaryConditions<2> bc;
        bc.start_velocity = initPos.col(1);
        bc.end_velocity = endPos.col(1);

        // 调用样条库根据时间节点、空间路径点和边界速度重建当前三次样条，
        // 并同步更新类内保存的连续轨迹对象。
        cubic_spline_.update(times, waypoints, bc);
        trajectory_ = cubic_spline_.getTrajectory();
    }

    double calculatePathLength(const std::vector<Eigen::Vector2d>& path) {
        double length = 0.0;
        for (size_t i = 1; i < path.size(); ++i) {
            length += (path[i] - path[i-1]).norm();
        }
        return length;
    }

    inline double logC2(const double& T) {
        return T > 1.0 ? (sqrt(2.0 * T - 1.0) - 1.0) : (1.0 - sqrt(2.0 / T - 1.0));
    }

    inline void calTfromTauUni(const double& tau, Eigen::VectorXd& T) {
        T.setConstant(expC2(tau) / T.size());
    }

    inline double expC2(const double& tau) {
        return tau > 0.0 ? ((0.5 * tau + 1.0) * tau + 1.0) : 1.0 / ((0.5 * tau - 1.0) * tau + 1.0);
    }

    inline double getTtoTauGrad(const double& tau) {
        if (tau > 0)
            return tau + 1.0;
        else {
            double denSqrt = (0.5 * tau - 1.0) * tau + 1.0;
            return (1.0 - tau) / (denSqrt * denSqrt);
        } 
    }

    std::shared_ptr<grid_map::GridMap> map_;
    std::vector<Eigen::Vector2d> astar_path_;
    TrajectoryParams params_;
    bool in_opt_;
    int piece_pos_;
    int dim_T;
    MatrixXd init_pos_;
    MatrixXd end_pos_;
    PPoly2D trajectory_;
    CubicSpline2D cubic_spline_; 
    int current_segment_ = 0;
};

} // namespace TrajOpt

#endif // TRAJ_OPT_HPP
