#include <functional>
#include "planner_base/planner_baseline.h"
namespace planner {

PlannerBaseline::PlannerBaseline():
    setup_finished(false),
    m_init(nullptr),
    m_goal(nullptr),
    m_size(nullptr),
    m_local_cost_map(nullptr),
    m_local_info_map(nullptr)
{
    // planner initialization
}

bool PlannerBaseline::plan()
{
    if(setup_finished){
        m_feasible_path.clear();
        this->AStar_algorithm(m_feasible_path);
        setup_finished = false;
        return true;
    }else{
        ROS_WARN("problem setting is not updated, using old path!");
        return false;
    }
}

void PlannerBaseline::plan_result(const data_type::PlannerMessage &plan_msg)
{
    if(plan_msg.m_feasible_path == nullptr){
        ROS_WARN("feasible path vector is not initialized!");
        return;
    }

    std::vector<NodePtr>::iterator it = m_feasible_path.begin();
    for(; it!=m_feasible_path.end(); ++it){
        plan_msg.m_feasible_path->push_back(Point3i((*it)->idx));
    }
}

void PlannerBaseline::setup_problem(const data_type::PlannerMessage &plan_msg)
{
    // setup init and goal
    m_init = plan_msg.m_init;
    m_goal = plan_msg.m_goal;
    m_local_frontier = plan_msg.m_frontier;

    // setup cost and info map
    m_size = plan_msg.m_local_size;
    m_local_cost_map = plan_msg.cost_map_ptr;
    m_local_info_map = plan_msg.info_map_ptr;

    // finished
    setup_finished = true;

    std::cout << "planner: init is: " << *m_init << std::endl;
    std::cout << "planner: goal is: " << *m_goal << std::endl;
}

void PlannerBaseline::get_successor(const NodePtr &pre, std::vector<NodePtr> &successor)
{
    const Point3i &idx = pre->idx;

    // x-axis neighbor
    if(idx.x()-1 >= 0){
        auto item = std::make_shared<Node>(Point3i(idx.x()-1, idx.y(), idx.z()), idx);
        successor.push_back(item);
    }
    if(idx.x()+1 < m_size->x()){
        auto item = std::make_shared<Node>(Point3i(idx.x()+1, idx.y(), idx.z()), idx);
        successor.push_back(item);
    }

    // y-axis neighbot
    if(idx.y()-1 >= 0){
        auto item = std::make_shared<Node>(Point3i(idx.x(), idx.y()-1, idx.z()), idx);
        successor.push_back(item);
    }
    if(idx.y()+1 < m_size->y()){
        auto item = std::make_shared<Node>(Point3i(idx.x(), idx.y()+1, idx.z()), idx);
        successor.push_back(item);
    }

    // z-axis neighbot
    if(idx.z()-1 >= 0){
        auto item = std::make_shared<Node>(Point3i(idx.x(), idx.y(), idx.z()-1), idx);
        successor.push_back(item);
    }
    if(idx.z()+1 < m_size->z()){
        auto item = std::make_shared<Node>(Point3i(idx.x(), idx.y(), idx.z()+1), idx);
        successor.push_back(item);
    }
}

bool PlannerBaseline::AStar_algorithm(std::vector<NodePtr> &path)
{
    // initialize the open list
    auto cmp = [](const NodePtr& n1, const NodePtr& n2) {return (n1->g+n1->h) > (n2->g+n2->h);};
    std::priority_queue<NodePtr, std::vector<NodePtr>, decltype (cmp)> open_list(cmp);

    // initialize th closed list
    NodePtr closed_list[m_size->x()][m_size->y()][m_size->z()];

    // put the start state into openlist
    auto init_ptr = std::make_shared<Node>(Node(*m_init));
    init_ptr->g = 0;
    open_list.push(init_ptr);

    // carry out search process
    NodePtr goal_found = nullptr;
    while (!open_list.empty()) {

        // get node with smallest f-value, top() does not actually remove the node
        NodePtr cur = open_list.top();

        // already the closed list, which means current node is the old version
        if(closed_list[cur->idx.x()][cur->idx.y()][cur->idx.z()] != nullptr){
            open_list.pop();
            continue;
        }

        // back track to get the optimal path
        if(cur->idx == (*m_goal)){
            goal_found = cur;
            break;
        }

        // remove the node and insert to closed list
        open_list.pop();
        closed_list[cur->idx.x()][cur->idx.y()][cur->idx.z()] = cur;
        //std::cout<< open_list.size() <<  std::endl;
        //std::cout << "Pre:" << cur->idx << " F: " << cur->g + cur->h << std::endl;

        // get successors the current node
        std::vector<NodePtr> successor;
        this->get_successor(cur, successor);

        // for every successor that is not in the closed list
        for(std::vector<NodePtr>::iterator it=successor.begin(); it!=successor.end(); ++it){

            NodePtr &child = *it;
            Point3i &idx = child->idx;

            // node already in the closed list
            if(closed_list[idx.x()][idx.y()][idx.z()] != nullptr){
                continue;
            }

            // node not in the closed list
            double cost_to_go = m_local_cost_map[idx.x()*m_size->y()*m_size->z()+idx.y()*m_size->z()+idx.z()];
            if(child->g > cur->g + cost_to_go){
                child->g = cur->g + cost_to_go;
                //child->set_heuristic((child->idx-(*m_goal)).norm()); // for 8-connected graph
                child->set_heuristic((child->idx-(*m_goal)).manhattan_dist()); // for 4-connected graph
                //std::cout << "Suc:" << child->idx << " F: " << child->g + child->h << std::endl;
                open_list.push(child);
            }
        }
    }

    // back track the optimal path
    if(goal_found != nullptr){
        NodePtr next = goal_found;
        while (next->pre != Point3i(-1, -1, -1)) {
            path.push_back(next);
            next = closed_list[next->pre.x()][next->pre.y()][next->pre.z()];
        }
        std::cout << "path length is " <<path.size()<< std::endl;
        return true;
    }else{
        std::cout<<"solution not found!"<<std::endl;
        return false;
    }
}


}; // end namespace
