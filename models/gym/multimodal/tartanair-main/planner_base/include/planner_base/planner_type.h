#ifndef PLANNER_TYPE_H
#define PLANNER_TYPE_H

#include <iostream>
#include <limits>
#include <cmath>
#include <data_type/point3i.h>

namespace planner{

const double INF = data_type::INF;
typedef data_type::Point3i Point3i;


class Node{

public:
    Point3i idx;
    Point3i pre;
    double g, h;

public:
    inline Node():idx(-1, -1, -1), pre(-1, -1, -1), g(INF), h(INF){}
    inline Node(const Point3i& idx):idx(idx), pre(-1, -1, -1), g(INF), h(INF){}
    inline Node(const Point3i& idx, const Point3i& pre):idx(idx), pre(pre), g(INF), h(INF){}
    inline void set_heuristic(double h){this->h = h;}

    inline bool operator <(const Node& n2){
        return (g+h) < (n2.g+n2.h);
    }
    inline bool operator ==(const Node& n2){
        return idx == n2.idx;
    }
};

}; // namespace end

#endif // PLANNER_TYPE_H
