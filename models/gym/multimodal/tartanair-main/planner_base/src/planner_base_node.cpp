#include <iostream>
#include <data_type/point3i.h>
#include <data_type/planner_message.h>
#include "planner_base/planner_base.h"
#include "planner_base/planner_baseline.h"

typedef data_type::Point3i Point3i;
typedef data_type::Point3d Point3d;


void detection_result(data_type::PlannerMessage &result)
{
    // local map size
    result.m_local_size = std::make_shared<Point3i>(7, 6, 2);

    // local map origin
    result.m_local_orig = std::make_shared<Point3d>(0, 0, 0);

    // init position
    result.m_init = std::make_shared<Point3i>(0, 0, 0);


    // local frontiers
    result.m_frontier = std::make_shared<std::vector<Point3i>>();
    result.m_frontier->push_back(Point3i(6, 2, 0));
    result.m_frontier->push_back(Point3i(6, 2, 3));
    result.m_frontier->push_back(Point3i(6, 3, 3));


    // local cost-map
    int map_len = result.m_local_size->x()*result.m_local_size->y()*result.m_local_size->z();

    Point3i &size = *result.m_local_size;
    result.cost_map_ptr = boost::make_shared<double[]>(map_len);
    for(int i=0; i<map_len; ++i){
        result.cost_map_ptr[i] = planner::INF;
    }
    result.cost_map_ptr[0*size.y()*size.z() + 0*size.z() + 0] = 1;
    result.cost_map_ptr[1*size.y()*size.z() + 0*size.z() + 0] = 1;
    result.cost_map_ptr[1*size.y()*size.z() + 1*size.z() + 0] = 1;
    result.cost_map_ptr[1*size.y()*size.z() + 2*size.z() + 0] = 1;
    result.cost_map_ptr[2*size.y()*size.z() + 1*size.z() + 0] = 1;
    result.cost_map_ptr[2*size.y()*size.z() + 3*size.z() + 0] = 1;
    result.cost_map_ptr[3*size.y()*size.z() + 1*size.z() + 0] = 1;
    result.cost_map_ptr[3*size.y()*size.z() + 3*size.z() + 0] = 1;
    result.cost_map_ptr[4*size.y()*size.z() + 2*size.z() + 0] = 1;
    result.cost_map_ptr[4*size.y()*size.z() + 3*size.z() + 0] = 1;
    result.cost_map_ptr[4*size.y()*size.z() + 4*size.z() + 0] = 1;
    result.cost_map_ptr[5*size.y()*size.z() + 2*size.z() + 0] = 1;
    result.cost_map_ptr[5*size.y()*size.z() + 3*size.z() + 0] = 1;
    result.cost_map_ptr[6*size.y()*size.z() + 2*size.z() + 0] = 1;

    result.cost_map_ptr[1*size.y()*size.z() + 3*size.z() + 0] = 1;
    result.cost_map_ptr[4*size.y()*size.z() + 1*size.z() + 0] = 1;



    // local info-map
    result.info_map_ptr = boost::make_shared<int[]>(map_len);
    std::memcpy(result.info_map_ptr.get(), result.cost_map_ptr.get(), map_len*sizeof(int));

    // init feasible path
    result.m_feasible_path = std::make_shared<std::vector<Point3i>>();
}

int main(){

    data_type::PlannerMessage plan_msg;
    detection_result(plan_msg);
    std::shared_ptr<planner::PlannerBaseline> planner = std::make_shared<planner::PlannerBaseline>();
    planner->setup_problem(plan_msg);
    planner->plan();
    planner->plan_result(plan_msg);
    if(plan_msg.m_feasible_path->size() == 0){
        std::cout << "No feasible path is found!" << std::endl;
        return 0;
    }

    for(auto it = plan_msg.m_feasible_path->begin(); it != plan_msg.m_feasible_path->end(); ++it){
        printf("(%d, %d, %d)", it->x(), it->y(), it->z());
    }

    return 0;
}
