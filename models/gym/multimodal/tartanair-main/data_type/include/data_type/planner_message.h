#ifndef PLANNER_MESSAGE_H
#define PLANNER_MESSAGE_H

#include <iostream>
#include <set>
#include <vector>
#include <boost/smart_ptr.hpp>
#include "data_type/point3i.h"
#include "data_type/point3d.h"

namespace data_type{

typedef std::shared_ptr<std::vector<Point3i>> point3i_vector_ptr;

class PlannerMessage
{

public:
    // information got from frontier detection
    point3i_vector_ptr m_frontier;              // frontier positions in local map
    std::shared_ptr<Point3i> m_local_size;      // local map size with three dims
    std::shared_ptr<Point3d> m_robot_pose;      // current pose of robot in global map
    std::shared_ptr<Point3d> m_local_orig;      // origin of local map in global map
    boost::shared_ptr<double[]> cost_map_ptr;   // cost map
    boost::shared_ptr<int[]> info_map_ptr;   // info map


    // to be initialized according exploration strategy
    std::shared_ptr<Point3i> m_init;            // init position
    std::shared_ptr<Point3i> m_goal;            // selected goal

    // planning result
    point3i_vector_ptr m_feasible_path;         // result


public:
    inline bool check_info_from_frontier() const {
        if(m_local_size == nullptr){std::cout<<"local size is not given!"; return false;}
        if(m_local_orig == nullptr){std::cout<<"local orig is not given!"; return false;}
        if(m_robot_pose == nullptr){std::cout<<"robot pose is not given!"; return false;}
        if(cost_map_ptr == nullptr){std::cout<<"cost map is not given!"; return false;}
        if(info_map_ptr == nullptr){std::cout<<"info map is not given!"; return false;}
        if(m_frontier == nullptr){std::cout<<"frontiers are not given!"; return false;}
        return true;
    }

    inline bool check_info_from_planner() const {
        if(m_init == nullptr){std::cout<<"init position is not given!"; return false;}
        if(m_goal == nullptr){std::cout<<"goal position is not given!"; return false;}
        if(m_feasible_path == nullptr){std::cout<<"feasible path is not instantialized!"; return false;}

        return true;
    }

    inline bool check_plan_parameter() const {
        if(m_init == nullptr){std::cout<<"init position is not given!"; return false;}
        if(m_goal == nullptr){std::cout<<"goal position is not given!"; return false;}
        if(m_local_size == nullptr){std::cout<<"cost map size is not given!"; return false;}
        if(cost_map_ptr == nullptr){std::cout<<"cost map is not given!"; return false;}
        if(info_map_ptr == nullptr){std::cout<<"info map is not given!"; return false;}
        return true;
    }


};  // END_CLASS

};  // END_NAME_SAPCE

#endif // PLANNER_MESSAGE_H
