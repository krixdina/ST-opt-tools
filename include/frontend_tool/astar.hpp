/*
    MIT License

    Copyright (c) 2025 Senming Tan (senmingtan5@gmail.com)

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

#pragma once

#include "grid_map.hpp"
#include <queue>
#include <unordered_map>
#include <chrono>
#include <functional>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <algorithm>
#include <Eigen/Dense>

namespace path_planning {

class AStar {
public:
    // Node structure containing position and search state
    struct Node {
        Eigen::Vector2d position;
        double g_score = 0;  // Actual cost from start to current node
        double f_score = 0;  // g_score + heuristic estimate
        Node* parent = nullptr;
        
        bool operator>(const Node& other) const { 
            return f_score > other.f_score; 
        }
    };

    // 5D state (x,y,theta,dtheta,ds)
    struct PathState {
        Eigen::Vector2d position;
        double theta = 0;       // Orientation angle
        double delta_theta = 0; // Angle change
        double delta_s = 0;     // Path segment length
    };

    // Timed trajectory point
    struct TimedTrajectoryPoint {
        Eigen::Vector3d state;  // x, y, theta
        double time = 0;
    };

    // Trajectory data structure
    struct Trajectory {
        // Path data
        std::vector<Eigen::Vector2d> raw_path;          // Original path
        std::vector<Eigen::Vector2d> optimized_path;    // Optimized path
        std::vector<PathState> path_states;             // Path states
        std::vector<TimedTrajectoryPoint> timed_trajectory; // Timed trajectory
        
        // State information
        Eigen::Vector3d start_state = Eigen::Vector3d::Zero();  // Start state (x_vel, y_vel, omega)
        Eigen::Vector3d final_state = Eigen::Vector3d::Zero();  // Final state
        Eigen::Vector3d start_state_XYTheta = Eigen::Vector3d::Zero(); // Start pose (x,y,theta)
        Eigen::Vector3d final_state_XYTheta = Eigen::Vector3d::Zero(); // Final pose
        
        // Path characteristics
        double total_time = 0;      // Total time
        double total_length = 0;    // Total length
        double weighted_length = 0; // Weighted length (considering angle changes)
        
        // Additional information
        bool if_cut = false;  // Whether trajectory is truncated
        std::vector<Eigen::Vector3d> UnOccupied_positions; // Unoccupied positions sequence
        double UnOccupied_initT = 0.0; // Initial time
    };

    AStar(grid_map::GridMap& map, double safe_threshold = 0.3) 
        : map_(map), safe_threshold_(safe_threshold) {
        // Initialize default parameters (can be loaded from config file)
        max_vel_ = 1.0;
        max_acc_ = 1.0;
        time_resolution_ = 0.2;
        min_traj_num_ = 3;
        traj_cut_length_ = 8.0;
        distance_weight_ = 1.0;
        yaw_weight_ = 0.5;
    }

    Trajectory planWithPostProcessing(
        const Eigen::Vector2d& start,
        const Eigen::Vector2d& goal,
        int timeout_ms = 1000
    ) {
        Trajectory traj;
        auto start_time = std::chrono::steady_clock::now();
        
        // 1. Original A* search
        traj.raw_path = originalAStarSearch(start, goal, timeout_ms);
        if (traj.raw_path.empty()) return traj;

        // 2. Path optimization (remove redundant waypoints)
        traj.optimized_path = optimizePath(traj.raw_path);
        
        // 3. State sampling (generate 5D states)
        traj.path_states = samplePathStates(traj.optimized_path);
        
        // 4. Time allocation (trapezoidal velocity profile)
        assignTrajectoryTiming(traj);
        
        // 5. Fill additional information
        fillAdditionalTrajectoryInfo(traj, start, goal);
        
        return traj;
    }

    std::vector<Eigen::Vector2d> originalAStarSearch(
        const Eigen::Vector2d& start,
        const Eigen::Vector2d& goal,
        int timeout_ms = 1000
    ) {
        auto start_time = std::chrono::steady_clock::now();
        
        // Check start/goal validity
        if (checkCollision(start)) {
            std::cerr << "Start point in collision!" << std::endl;
            return {};
        }
        if (checkCollision(goal)) {
            std::cerr << "Goal point in collision!" << std::endl;
            return {};
        }

        // Open set (priority queue)
        auto cmp = [](Node* a, Node* b) { return *a > *b; };
        std::priority_queue<Node*, std::vector<Node*>, decltype(cmp)> open_set(cmp);
        
        // Node storage and hash map
        std::vector<std::unique_ptr<Node>> nodes;
        std::unordered_map<std::string, Node*> node_map;

        // Start node
        nodes.emplace_back(new Node{start, 0, heuristic(start, goal)});
        open_set.push(nodes.back().get());
        node_map[positionToKey(start)] = nodes.back().get();

        while (!open_set.empty()) {
            // Check timeout
            if (std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time).count() > timeout_ms) {
                std::cerr << "A* timeout!" << std::endl;
                return {};
            }

            Node* current = open_set.top();
            open_set.pop();

            // Reached goal (using resolution as tolerance)
            if ((current->position - goal).norm() < map_.getResolution()) {
                return reconstructPath(current);
            }

            // Expand neighbors (8 directions)
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    if (dx == 0 && dy == 0) continue;

                    Eigen::Vector2d neighbor_pos = current->position + 
                        Eigen::Vector2d(dx, dy) * map_.getResolution();

                    // Skip collision points
                    if (checkCollision(neighbor_pos)) {
                        continue;
                    }

                    // Calculate tentative_g_score
                    double step_cost = (dx * dy == 0) ? 1.0 : 1.414;  // Orthogonal/diagonal cost
                    double tentative_g = current->g_score + step_cost * map_.getResolution();

                    // Check if better path exists
                    std::string key = positionToKey(neighbor_pos);
                    auto it = node_map.find(key);
                    if (it == node_map.end() || tentative_g < it->second->g_score) {
                        // New node or found better path
                        nodes.emplace_back(new Node{
                            neighbor_pos, 
                            tentative_g, 
                            tentative_g + heuristic(neighbor_pos, goal),
                            current
                        });
                        open_set.push(nodes.back().get());
                        node_map[key] = nodes.back().get();
                    }
                }
            }
        }

        return {};  // Open set empty, no path found
    }

    // Parameter setters
    void setMaxVelocity(double max_vel) { max_vel_ = max_vel; }
    void setMaxAcceleration(double max_acc) { max_acc_ = max_acc; }
    void setTimeResolution(double time_res) { time_resolution_ = time_res; }
    void setMinTrajectoryNumber(int min_num) { min_traj_num_ = min_num; }
    void setTrajectoryCutLength(double length) { traj_cut_length_ = length; }
    void setDistanceWeight(double weight) { distance_weight_ = weight; }
    void setYawWeight(double weight) { yaw_weight_ = weight; }

private:
    grid_map::GridMap& map_;
    double safe_threshold_;
    
    // Trajectory parameters
    double max_vel_;
    double max_acc_;
    double time_resolution_;
    int min_traj_num_;
    double traj_cut_length_;
    double distance_weight_;
    double yaw_weight_;

    // Euclidean heuristic
    double heuristic(const Eigen::Vector2d& a, const Eigen::Vector2d& b) const {
        return (a - b).norm();
    }

    // Position to hash key
    std::string positionToKey(const Eigen::Vector2d& pos) const {
        return std::to_string(static_cast<int>(pos.x() * 100)) + "," + 
               std::to_string(static_cast<int>(pos.y() * 100));
    }

    // Reconstruct path
    std::vector<Eigen::Vector2d> reconstructPath(Node* node) const {
        std::vector<Eigen::Vector2d> path;
        while (node) {
            path.push_back(node->position);
            node = node->parent;
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    bool checkCollision(const Eigen::Vector2d& pos) const {
        // Points outside map are considered collision-free
        if (!map_.isInsideMap(pos)) return false;
        
        // Points inside map use safety distance check
        return map_.getDistance(pos) < safe_threshold_;
    }

    std::vector<Eigen::Vector2d> optimizePath(const std::vector<Eigen::Vector2d>& path) {
        if (path.size() < 2) return path;

        std::vector<Eigen::Vector2d> optimized_path;
        optimized_path.push_back(path[0]);
        Eigen::Vector2d prev_pose = path[0];
        double cost1, cost2, cost3;

        // Check first path segment
        if (!checkLineCollision(path[0], path[1]))
            cost1 = (path[0] - path[1]).norm();
        else
            cost1 = std::numeric_limits<double>::infinity();

        for (size_t i = 1; i < path.size() - 1; ++i) {
            const auto& pose1 = path[i];
            const auto& pose2 = path[i + 1];
            
            // Calculate current segment cost
            if (!checkLineCollision(pose1, pose2))
                cost2 = (pose1 - pose2).norm();
            else
                cost2 = std::numeric_limits<double>::infinity();

            // Calculate skip current point cost
            if (!checkLineCollision(prev_pose, pose2))
                cost3 = (prev_pose - pose2).norm();
            else
                cost3 = std::numeric_limits<double>::infinity();

            // 决定是否保留当前中间点 path[i]
            // 若 prev_pose -> pose2 直连更短，则说明 pose1 是冗余点。
            // 此时这里不将 path[i] 放入 optimized_path，
            // 等价于“隐式跳过/删除当前点”。
            if (cost3 < cost1 + cost2) {
                // 更新合并后路径段的代价，供下一轮继续与后续点比较
                cost1 = cost3;
            } else {
                // 直连不划算或不可行，保留当前点作为新的关键点
                optimized_path.push_back(path[i]);
                cost1 = (pose1 - pose2).norm();
                prev_pose = pose1;
            }
        }

        optimized_path.push_back(path.back());
        return optimized_path;
    }

    bool checkLineCollision(const Eigen::Vector2d& start, const Eigen::Vector2d& end) const {
        Eigen::Vector2i start_idx, end_idx;
        map_.posToIndex(start, start_idx);
        map_.posToIndex(end, end_idx);
        
        int dx = abs(end_idx.x() - start_idx.x());
        int dy = abs(end_idx.y() - start_idx.y());
        int sx = start_idx.x() < end_idx.x() ? 1 : -1;
        int sy = start_idx.y() < end_idx.y() ? 1 : -1;
        int err = dx - dy;

        while (true) {
            Eigen::Vector2d pos;
            map_.indexToPos(start_idx, pos);
            if (checkCollision(pos)) return true;
            if (start_idx == end_idx) break;
            
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; start_idx.x() += sx; }
            if (e2 < dx) { err += dx; start_idx.y() += sy; }
        }
        return false;
    }

    std::vector<PathState> samplePathStates(const std::vector<Eigen::Vector2d>& path) {
        std::vector<PathState> states;
        if (path.empty()) return states;

        // Initial state
        states.push_back({path[0], 0, 0, 0});
        
        for (size_t i = 1; i < path.size(); ++i) {
            const auto& prev = states.back();
            const auto& curr_pos = path[i];
            
            // Calculate orientation angle
            double theta = atan2(curr_pos.y() - prev.position.y(), 
                               curr_pos.x() - prev.position.x());
            
            // Normalize angle
            normalizeAngle(prev.theta, theta);
            
            // Calculate segment length
            double delta_s = (curr_pos - prev.position).norm();
            
            states.push_back({
                curr_pos,
                theta,
                theta - prev.theta,
                delta_s
            });
        }
        return states;
    }

    void normalizeAngle(double ref_angle, double& angle) const {
        while (ref_angle - angle > M_PI) angle += 2 * M_PI;
        while (ref_angle - angle < -M_PI) angle -= 2 * M_PI;
    }
    
    // 对整条路径中的每一段路径应用梯形/三角加减速模型得到整条路径的执行总时长 total_time 
    // 根据 total_time 计算出时间采样点之间的间隔 sample_time
    // 在整条路径中按照 sample_time 使用梯形加减速模型进行插值获取每一个 sample_time 在轨迹上的对应状态
    // 将插值点组成的轨迹送入后端进行优化
    void assignTrajectoryTiming(Trajectory& traj) {
        if (traj.path_states.empty()) return;

        // Calculate weighted path length (considering angle changes)
        traj.total_length = 0;
        traj.weighted_length = 0;
        for (const auto& state : traj.path_states) {
            traj.total_length += state.delta_s;
            traj.weighted_length += state.delta_s * distance_weight_ + 
                                   fabs(state.delta_theta) * yaw_weight_;
        }

        // Calculate total time (trapezoidal velocity profile)
        traj.total_time = evaluateDuration(traj.weighted_length, 0, 0, max_vel_, max_acc_);
        
        // 计算相邻两个时间采样点之间的间隔 sample_time。
        // 这里不是直接令 sample_time = time_resolution_，
        // 而是先用 total_time / time_resolution_ 估算“整条轨迹大约应采多少个点”。
        //
        // 其中 time_resolution_ 表示期望的时间采样分辨率，单位是秒：
        // 例如 time_resolution_ = 0.2，表示希望大致每 0.2s 采一个轨迹点。
        //
        // 但如果 total_time 很短，按这个分辨率算出来的采样点数可能太少，
        // 所以这里还用 min_traj_num_ 约束最少采样点数。
        // 最终 sample_time = 总时长 / 实际采样段数。
        double sample_time = traj.total_time / 
                           std::max(static_cast<int>(traj.total_time / time_resolution_ + 0.5), 
                                   min_traj_num_);

        // Generate timed trajectory points
        double accumulated_s = 0;
        size_t state_idx = 0;
        
        for (double t = sample_time; t < traj.total_time - 1e-3; t += sample_time) {
            // 根据当前时刻 t 反推出此时沿路径累计前进了多远
            double s = evaluateLength(t, traj.weighted_length, traj.total_time, 
                                    0, 0, max_vel_, max_acc_);
            
            // 在离散路径段中查找累计路程 s 落在哪一段
            // 在将当前段纳入累计长度后，必须大于按照当前时间 t 查询出的累计前进距离 s
            while (state_idx < traj.path_states.size() - 1 && 
                   accumulated_s + traj.path_states[state_idx].delta_s < s) {
                // 若当前段还不足以覆盖 s，则继续向后推进到下一段
                accumulated_s += traj.path_states[state_idx].delta_s;
                state_idx++;
            }
            
            if (state_idx > 0) {
                // 计算 s 在当前路径段内部的相对位置比例，用于线性插值
                double ratio = (s - accumulated_s) / traj.path_states[state_idx].delta_s;
                
                // 在相邻两个路径状态之间插值得到当前位置
                Eigen::Vector2d pos = traj.path_states[state_idx-1].position + 
                                     ratio * (traj.path_states[state_idx].position - 
                                             traj.path_states[state_idx-1].position);
                // 对航向角也做同样的插值，得到当前时刻的姿态
                double theta = traj.path_states[state_idx-1].theta + 
                             ratio * (traj.path_states[state_idx].delta_theta);
                
                // 将带时间戳的轨迹点 (x, y, theta, t) 保存下来
                traj.timed_trajectory.push_back({
                    Eigen::Vector3d(pos.x(), pos.y(), theta),
                    t
                });
            }
        }
    }

    void fillAdditionalTrajectoryInfo(Trajectory& traj, 
                                    const Eigen::Vector2d& start,
                                    const Eigen::Vector2d& goal) {
        // Set start/goal states
        traj.start_state_XYTheta << start.x(), start.y(), 0;
        traj.final_state_XYTheta << goal.x(), goal.y(), traj.path_states.back().theta;
        
        // Generate unoccupied positions sequence
        // 将位置序列从轨迹序列中提取出来
        for (const auto& state : traj.path_states) {
            traj.UnOccupied_positions.emplace_back(
                state.position.x(), 
                state.position.y(), 
                state.theta
            );
        }
        
        // 以“总时长 / 采样点数”的方式估计相邻轨迹点之间的平均时间间隔，
        // 这里的 UnOccupied_initT 更准确地说是表示了初始时间步长。
        traj.UnOccupied_initT = traj.total_time / traj.timed_trajectory.size();
        
        // Check if needs truncation
        traj.if_cut = (traj.total_length > traj_cut_length_);
        
        // Set default velocity/acceleration (can be adjusted as needed)
        traj.start_state << 0, 0, 0;  // Initial velocity 0
        traj.final_state << 0, 0, 0;   // Final velocity 0
    }

    double evaluateDuration(double length, double start_vel, double end_vel, 
                           double max_vel, double max_acc) const {
        // 先计算达到 max_vel 所需的最小路径长度：
        // 前半段用于从 start_vel 加速到 max_vel，
        // 后半段用于从 max_vel 减速到 end_vel。
        double critical_len = (max_vel*max_vel - start_vel*start_vel)/(2*max_acc) + 
                            (max_vel*max_vel - end_vel*end_vel)/(2*max_acc);
        
        if (length >= critical_len) {
            // 路径足够长：采用梯形速度曲线
            // 总时间 = 加速时间 + 匀速时间 + 减速时间
            return (max_vel - start_vel)/max_acc + 
                  (max_vel - end_vel)/max_acc + 
                  (length - critical_len)/max_vel;
        } else {
            // 路径不够长：达不到 max_vel，采用三角速度曲线
            // 先求能达到的峰值速度 tmp_vel，再计算加减速总时间
            double tmp_vel = sqrt(0.5*(start_vel*start_vel + end_vel*end_vel + 2*max_acc*length));
            return (tmp_vel - start_vel)/max_acc + (tmp_vel - end_vel)/max_acc;
        }
    }

    // 给定时刻 t，按照梯形/三角速度曲线估计从起点累计走过的路径长度。
    //
    // 这个函数与 evaluateDuration() 是配套使用的：
    // 1. evaluateDuration() 先根据总路径长度估计整条轨迹总时长；
    // 2. evaluateLength() 再根据某个时刻 t，反推出该时刻已经走过的累计路程 s。
    //
    // total_length 表示整条路径的总长度（这里实际上传入的是 weighted_length），
    // t 表示当前采样时刻，返回值则表示该时刻对应的累计路径长度。
    double evaluateLength(double t, double total_length, double total_time, 
                         double start_vel, double end_vel, 
                         double max_vel, double max_acc) const {
        // 与 evaluateDuration() 中相同，critical_len 表示：
        // 从 start_vel 加速到 max_vel，再从 max_vel 减速到 end_vel
        // 至少需要多少路径长度。
        double critical_len = (max_vel*max_vel - start_vel*start_vel)/(2*max_acc) + 
                            (max_vel*max_vel - end_vel*end_vel)/(2*max_acc);
        
        if (total_length >= critical_len) {
            // 路径足够长：采用梯形速度曲线
            // t1: 加速段结束时刻
            // t2: 匀速段结束、减速段开始时刻
            double t1 = (max_vel - start_vel)/max_acc;
            double t2 = t1 + (total_length - critical_len)/max_vel;
            
            if (t <= t1) {
                // 加速阶段：s = v0 * t + 1/2 * a * t^2
                return start_vel*t + 0.5*max_acc*t*t;
            } else if (t <= t2) {
                // 匀速阶段：
                // 累计长度 = 加速段长度 + 匀速段长度
                return start_vel*t1 + 0.5*max_acc*t1*t1 + (t - t1)*max_vel;
            } else {
                // 减速阶段：
                // 累计长度 = 加速段长度 + 匀速段长度 + 当前减速段内走过的长度
                return start_vel*t1 + 0.5*max_acc*t1*t1 + (t2 - t1)*max_vel + 
                      max_vel*(t - t2) - 0.5*max_acc*(t - t2)*(t - t2);
            }
        } else {
            // 路径不够长：达不到 max_vel，采用三角速度曲线
            // tmp_vel: 在该路径长度约束下能够达到的峰值速度
            // tmp_t: 加速到峰值速度所需时间
            double tmp_vel = sqrt(0.5*(start_vel*start_vel + end_vel*end_vel + 2*max_acc*total_length));
            double tmp_t = (tmp_vel - start_vel)/max_acc;
            
            if (t <= tmp_t) {
                // 加速阶段：公式与梯形速度曲线的加速段相同
                return start_vel*t + 0.5*max_acc*t*t;
            } else {
                // 减速阶段：
                // 累计长度 = 前半段加速长度 + 后半段减速过程中已走过的长度
                return start_vel*tmp_t + 0.5*max_acc*tmp_t*tmp_t + 
                      tmp_vel*(t - tmp_t) - 0.5*max_acc*(t - tmp_t)*(t - tmp_t);
            }
        }
    }
};

} // namespace path_planning
