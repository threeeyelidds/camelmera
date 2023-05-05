#include "frontier_base/frontier_baseline.h"
#include <cassert>

// // for debug
// #include <opencv2/opencv.hpp>
// #include <opencv2/highgui.hpp>

namespace frontier {

FrontierBaseline::FrontierBaseline():
    m_nh(),
    m_bbx_depth(15),
    m_bbx_upper_orig(10),
    m_bbx_below_orig(10),
    m_map_x_max(100.0), 
    m_map_x_min(-100.0),
    m_map_y_max(100.0), 
    m_map_y_min(-100.0),
    m_map_z_max(5.0), 
    m_map_z_min(-5.0),
    m_exterior_para(false),
    m_sensor_range(-1.0),
    m_tree_max_depth(16),
    m_tree_resolution(-1.0),
    neighbor_count_thresh(3),
    m_world_frame_id("/map"),
    global_map_ptr(nullptr),
    temp_free_map_ptr(nullptr),
    local_frontier_size(0),
    local_frontier_ptr(nullptr),
    global_frontier_ptr(nullptr),
    m_local_cost_map(nullptr),
    m_local_info_map(nullptr),
    cost_inflation_iter(1),
    cost_inflation_val(5.0),
    frontier_update_lock(false)
{
    // parse ros parameters
    ros::NodeHandle private_nh = ros::NodeHandle("~");
    private_nh.param("bbx_depth", m_bbx_depth, m_bbx_depth);
    // private_nh.param("bbx_upper_orig", m_bbx_upper_orig, m_bbx_upper_orig);
    // private_nh.param("bbx_below_orig", m_bbx_below_orig, m_bbx_below_orig);

    private_nh.param("map_x_max", m_map_x_max, m_map_x_max);
    private_nh.param("map_x_min", m_map_x_min, m_map_x_min);
    private_nh.param("map_y_max", m_map_y_max, m_map_y_max);
    private_nh.param("map_y_min", m_map_y_min, m_map_y_min);
    private_nh.param("map_z_max", m_map_z_max, m_map_z_max);
    private_nh.param("map_z_min", m_map_z_min, m_map_z_min);

    private_nh.param("neighbor_count_thresh", neighbor_count_thresh, neighbor_count_thresh);

    private_nh.param("cost_inflation_iter", cost_inflation_iter, cost_inflation_iter);
    private_nh.param("cost_inflation_val", cost_inflation_val, cost_inflation_val);

    // check ros parameters
    m_bbx_upper_orig = ceil(-m_map_z_min);
    m_bbx_below_orig = ceil(m_map_z_max);
    assert((m_bbx_depth > 10 && m_bbx_depth < 16));
    assert((m_bbx_upper_orig > 0 && m_bbx_below_orig > 0));

    // setup publisher
    m_free_pub = m_nh.advertise<visualization_msgs::Marker>("bbx_free", true);
    m_unknown_pub = m_nh.advertise<visualization_msgs::Marker>("bbx_unknown", true);
    m_local_frontier_pub = m_nh.advertise<visualization_msgs::Marker>("bbx_local_frontier", true);
    m_global_frontier_pub = m_nh.advertise<visualization_msgs::Marker>("bbx_global_frontier", true);

    m_cost_pub = m_nh.advertise<visualization_msgs::Marker>("bbx_cost", true);

    // initialize detection
    this->set_exterior_para();
    this->set_interior_para();
}

//FrontierBaseline::~FrontierBaseline()
//{
//    m_local_cost_map.reset();
//    m_local_info_map.reset();
//}

void FrontierBaseline::set_exterior_para()
{
    // read private parameter
    ros::NodeHandle private_nh = ros::NodeHandle("~");
    private_nh.param("sensor_model/max_range", m_sensor_range, m_sensor_range);
    private_nh.param("resolution", m_tree_resolution, m_tree_resolution);
    private_nh.param("frame_id", m_world_frame_id, m_world_frame_id);

    // check if illegal
    if(m_sensor_range <= 0 || m_tree_resolution <=0 || m_tree_max_depth <=0){
        ROS_ERROR("sensor range or tree resolution is illegal!");
        exit(EXIT_FAILURE);
    }
    m_exterior_para = true;

    ROS_INFO("sensor range is %f", m_sensor_range);
    ROS_INFO("tree reslolution is %f", m_tree_resolution);
    ROS_INFO("tree max depth is %f", m_tree_max_depth);

}

void FrontierBaseline::set_interior_para()
{
    if(!m_exterior_para){
        ROS_ERROR("set_interior_para must be called after set_exterior_para");
        exit(EXIT_FAILURE);
    }

    // calculate the cell size beforehand, the m_bbx_depth should be at most 12
    m_bbx_cell_size = m_tree_resolution * pow(2, m_tree_max_depth - m_bbx_depth); // amigo: 0.25*(2^(16-13)) = 2
    if(m_bbx_cell_size > m_sensor_range){
        ROS_ERROR("the bbx_cell_size is bigger than range_sensor, try a samller bbx_depth");
        exit(EXIT_FAILURE);
    }

    // the x and y grid numbers
    //amigo: make the local map larger to make sure there's one layer of unknown cells around at first: ceil(19.99/2)+1 = 11
    // otherwise, there will be no frontiers at the boarder at the begining
    int grid_xy_number = ceil(m_sensor_range/m_bbx_cell_size) + 1; 
    m_offset.x() = grid_xy_number; // 11
    m_offset.y() = grid_xy_number; // 11
    m_bbx_size.x() = grid_xy_number * 2 + 1; // 22 + 1
    m_bbx_size.y() = grid_xy_number * 2 + 1; // 22 + 1

    // the z grid numbers
    int grid_upper_number = ceil(m_bbx_upper_orig/m_bbx_cell_size) + 1; // ceil(10/2) + 1 = 6
    int grid_below_number = ceil(m_bbx_below_orig/m_bbx_cell_size) + 1; // ceil(2/2) + 1 = 2
    m_offset.z() = grid_upper_number; //6
    m_bbx_size.z() = grid_upper_number + grid_below_number; // 8

    // previous key
    const int INF = std::numeric_limits<int>::infinity();
    m_last_key.x() = INF;
    m_last_key.y() = INF;
    m_last_key.z() = INF;

    // generate local map
    m_local_cost_map = boost::make_shared<double[]>(m_bbx_size.x()*m_bbx_size.y()*m_bbx_size.z());
    m_local_info_map = boost::make_shared<int[]>(m_bbx_size.x()*m_bbx_size.y()*m_bbx_size.z());
    // generate global frontier
    global_frontier_ptr = std::make_shared<Node_List>();

    // visulization message
    ROS_INFO("cell size %f", m_bbx_cell_size);
    ROS_INFO("detection level %d", m_bbx_depth);
    ROS_INFO("map bound x (%f, %f)", m_map_x_min, m_map_x_max);
    ROS_INFO("map bound y (%f, %f)", m_map_y_min, m_map_y_max);
    ROS_INFO("x-y-z grids number (%d, %d, %d)", m_bbx_size.x(), m_bbx_size.y(), m_bbx_size.z());
    ROS_INFO("z upper-below grids number (%d, %d)", grid_upper_number, grid_below_number);
}

void FrontierBaseline::detect(
    const std::shared_ptr<OcTree> &global_map, 
    const std::shared_ptr<OcTree> &temp_free_map,
    const octomap::point3d &robot_pose)
{
    ROS_INFO("frontier_detector::robot pose is (%f, %f, %f)", robot_pose.x(), robot_pose.y(), robot_pose.z());
    // this is a write thread
    WriteLock write_lock(m_rw_lock);

    // Todo: only check cells that are changed
    // float max_entropy = 0.6931471805599453;
    Point3i curr_key(floor(robot_pose.x()/m_bbx_cell_size),
                     floor(robot_pose.y()/m_bbx_cell_size),
                     floor(robot_pose.z()/m_bbx_cell_size));
    //if(curr_key == m_last_key){return;}
    //m_last_key = curr_key;

    // convert robot_position at level m_bbx_depth
    double bot_x = curr_key.x()*m_bbx_cell_size;
    double bot_y = curr_key.y()*m_bbx_cell_size;
    double bot_z = curr_key.z()*m_bbx_cell_size;
    Point3d orig_offset(bot_x, bot_y, 0.0);

    // update local variable
    curr_pose = Point3d(robot_pose.x(), robot_pose.y(), robot_pose.z());
    curr_orig = orig_offset;
    global_map_ptr = global_map;
    temp_free_map_ptr = temp_free_map;
    ROS_INFO("frontier_detector::origin is (%f, %f, %f)", curr_orig.x(), curr_orig.y(), curr_orig.z());

    // Save robot_pose.
    valid_poses.push_back(robot_pose);

    // detect free voxels, among which the frontiers will be selected furthmore
    std::vector<BBXNode> key_list_free;
    this->detect_local_free_node(orig_offset, key_list_free);

    // detect frontiers in local map
    auto local_frontier = std::make_shared<Node_List>();
    this->detect_local_frontier(key_list_free, local_frontier, valid_poses.size()-1);

    if(!frontier_update_lock)
    {
        // merge local frontiers into global frontiers
        this->merge_local_frontier(local_frontier);

    }
    frontier_update_lock = false;
}

/*
m_local_info_map: -1: unknow
                   0: obstacle
                   1: free
*/
void FrontierBaseline::detect_local_free_node(const Point3d& local_orig,
                                              std::vector<BBXNode> &vector_free)
{
    // reset local information map and local cost-to-go map
    int cell_num = m_bbx_size.x()*m_bbx_size.y()*m_bbx_size.z();
    for(int i=0; i < cell_num; ++i){
        m_local_info_map[i] = -1; 
        m_local_cost_map[i] = COST_INF;
    }

    //ROS_INFO("detect_free::bounding box min is (%f, %f, %f)", bbx_min.x(), bbx_min.y(), bbx_min.z());
    //ROS_INFO("detect_free::bounding box max is (%f, %f, %f)", bbx_max.x(), bbx_max.y(), bbx_max.z());

    // get all the free numbers
    int explored_voxel = 0;

    for(int i=0; i<m_bbx_size.x(); i++){
        for(int j=0; j<m_bbx_size.y(); j++){
            for(int k=0; k<m_bbx_size.z(); k++){

                // get global coor
                Point3i local_key(i,j,k);
                Point3d local_coor;
                this->key2coor(local_key, local_coor);
                Point3d global_coor = local_coor + local_orig;

                Point3i confirm;
                this->coor2key(local_coor, confirm);
                assert(confirm == local_key);

                // query node according to key
                OcTreeNode* node = global_map_ptr->search(global_coor.x(), global_coor.y(), global_coor.z(), m_bbx_depth);
                const OcTreeNode* node_temp_free = temp_free_map_ptr->search(global_coor.x(), global_coor.y(), global_coor.z(), m_bbx_depth);

                // for unknown cells
                // if(node == nullptr || node_temp_free == nullptr){
                if(node == nullptr){
                    continue;
                }

                // Global index.
                const octomap::point3d g_coor(global_coor.x(), global_coor.y(), global_coor.z());
                octomap::OcTreeKey g_key;
                if (!global_map_ptr->coordToKeyChecked(g_coor, g_key) ) {
                    continue;
                }

                explored_voxel ++;

                int nodeind = i*m_bbx_size.y()*m_bbx_size.z() + j* m_bbx_size.z() + k;
                // // for occupied and free cells
                // double info = node->getEntropyChild();
                // m_local_info_map[i*m_bbx_size.y()*m_bbx_size.z() + j* m_bbx_size.z() + k] = info;

                // ROS_INFO("Node (%d, %d, %d) value %f, info %f", i, j, k, node->getValue(), info);
                // free space
                if(node_temp_free != nullptr && node_temp_free->getValue() < -0.05) {
                    if(node->getValue() < -0.05){
                        // curr_pose specify the history trajectory
                        BBXNode free_node(local_key, global_coor, curr_pose);
                        free_node.global_key = g_key;
                        vector_free.push_back(free_node);
                    }
                }

                if(node->getValue() < -0.05){
                    // curr_pose specify the history trajectory
                    // BBXNode free_node(local_key, global_coor, curr_pose);
                    // vector_free.push_back(free_node);
                    m_local_cost_map[nodeind] = 1;
                    m_local_info_map[nodeind] = 1; // free
                }
                else
                {
                    m_local_info_map[nodeind] = 0; // obstacle
                }

            } // end k
        } //end j
    } // end i

    ROS_INFO("free_detector:: (free, explored, unknown) is (%d, %d, %d)",
             (int)vector_free.size(), explored_voxel, cell_num-explored_voxel);

    cost_inflation(cost_inflation_iter, cost_inflation_val);

    if(m_cost_pub.getNumSubscribers() > 0){
        this->publish_cost_node();
    }

    // visualize the free space
    if(m_free_pub.getNumSubscribers() > 0){
        this->publish_free_node(vector_free);
    }

    // visualize the unknown space
    if(m_unknown_pub.getNumSubscribers() > 0){
        this->publish_unknown_node();
    }

}

void FrontierBaseline::cost_inflation(int iter_count, int inflate_val)
{
    int cell_num = m_bbx_size.x()*m_bbx_size.y()*m_bbx_size.z();
    int multi1 = m_bbx_size.y()*m_bbx_size.z();
    int multi2 = m_bbx_size.z();

    int x_off[18] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1};
    int y_off[18] = {0, 0,-1, 1,-1, 1,-1, 1, 0, 0, 0,-1, 1, 0, 0, 0,-1, 1};
    int z_off[18] = {1,-1, 0, 0, 1,-1,-1, 1, 0, 1,-1, 0, 0, 0, 1,-1, 0, 0};

    // ROS_INFO("inflate bbx size (%d, %d, %d)", m_bbx_size.x(), m_bbx_size.y(), m_bbx_size.z());

    for(int s=0; s<iter_count; s++) {

        boost::shared_ptr<double[]> old_local_cost_map;  // cost-to-go map
        old_local_cost_map = boost::make_shared<double[]>(cell_num);
        std::memcpy(old_local_cost_map.get(), m_local_cost_map.get(), cell_num*sizeof(double));

        for(int i=0; i<m_bbx_size.x(); i++){
            for(int j=0; j<m_bbx_size.y(); j++){
                for(int k=0; k<m_bbx_size.z(); k++){
                    int nodeind = i*multi1 + j*multi2 + k;
                    if(m_local_info_map[nodeind]<=0)  // obstacle or unknow
                        continue;
                    for(int w=0; w<18; w++){ // check whether it is close to an obstacle node
                        int adjnodeind = (i+x_off[w])*multi1 + (j+y_off[w])*multi2 + (k+z_off[w]);
                        if(i+x_off[w]<0 || j+y_off[w]<0 || k+z_off[w]<0 || 
                            i+x_off[w]>=m_bbx_size.x() || j+y_off[w]>=m_bbx_size.y() || k+z_off[w]>=m_bbx_size.z()) // adjacent node is out of boundary 
                            continue;
                        if(old_local_cost_map[adjnodeind]>1.5){ // find an adjacent node with cost
                            // ROS_INFO("inflate node (%d, %d, %d), adj (%d, %d, %d) cost %f", i, j, k, i+x_off[w], j+y_off[w], k+z_off[w], m_local_cost_map[adjnodeind]);

                            m_local_cost_map[nodeind] += inflate_val;
                            break; // each node inflate only once each iteration
                        }
                    }
                }
            }
        }
    }

    // // debug and show the cost as images
    // uint8_t greyArr[338][508]; // hardcode the size
    // int nodeind, arrind_x, arrind_y, row, col;
    // for(int k=0; k<m_bbx_size.z(); k++) {
    //     for(int i=0; i<m_bbx_size.x(); i++){
    //         for(int j=0; j<m_bbx_size.y(); j++){
    //             nodeind = i*multi1 + j*multi2 + k;  
    //             arrind_x = k % 6;
    //             arrind_y = k / 6;
    //             row = arrind_y * 85 + i;
    //             col = arrind_x * 85 + j;
    //             if (m_local_cost_map[nodeind] > 25)
    //                 greyArr[row][col] = 0;
    //             else
    //                 greyArr[row][col] = 255 - uint8_t(m_local_cost_map[nodeind]*10);
    //         }
    //         // ROS_INFO("%d, %d", row, col);
    //     }
    // }
    // cv::Mat greyImg = cv::Mat(338, 508, CV_8UC1, &greyArr);
    // cv::resize(greyImg, greyImg, cv::Size(), 2.0, 2.0, cv::INTER_NEAREST);
    // std::string greyArrWindow = "Costmap";
    // cv::namedWindow(greyArrWindow, cv::WINDOW_AUTOSIZE);
    // cv::imshow(greyArrWindow, greyImg);
    // cv::waitKey(0);


}

void FrontierBaseline::detect_local_frontier(std::vector<BBXNode>& vector_free,
                                             node_list_ptr local_frontier,
                                             std::size_t valid_pose_index)
{
    int count = 0;
    for(std::vector<BBXNode>::iterator it=vector_free.begin(); it!=vector_free.end(); ++it){

        // short reference
        Point3i &size = m_bbx_size;
        Point3i &key = it->local_index;
        Point3d &coor = it->global_coor;


        // 1. map boundary constraints
        // if(abs(coor.x()) > m_map_bound_x || abs(coor.y()) > m_map_bound_y){
        if(coor.x() > m_map_x_max || coor.x() < m_map_x_min || coor.y() > m_map_y_max || coor.y() < m_map_y_min){
            continue;
        }

        // 2. local closure constraints
        if(key.x()<=0||key.x()>=size.x()-1 || key.y()<=0||key.y()>=size.y()-1 || key.z()<=0||key.z()>=size.z()-1){
            continue;
        }

        // check the exterior
        int x_left = key.x()-1, x_rigt = key.x()+1;
        int y_left = key.y()-1, y_rigt = key.y()+1;
        int z_left = key.z()-1, z_rigt = key.z()+1;

        int multi1 = size.y()*size.z();
        int multi2 = size.z();
        int multi_x = key.x()*multi1;
        int multi_xl = x_left*multi1;
        int multi_xr = x_rigt*multi1;
        int multi_y = key.y()*multi2;
        int multi_yl = y_left*multi2;
        int multi_yr = y_rigt*multi2;

        int keyind = multi_x + multi_y + key.z();

        // share face
        int x_leftind = multi_xl + multi_y + key.z();
        int x_rightind = multi_xr + multi_y + key.z();
        int y_leftind = multi_x + multi_yl + key.z();
        int y_rightind = multi_x + multi_yr + key.z();
        int z_leftind = multi_x + multi_y + z_left;
        int z_rightind = multi_x + multi_y + z_rigt;

        // do not add frontier which is close to obstacle
        if (m_local_info_map[y_leftind] == 0 or m_local_info_map[y_rightind] == 0 or
            m_local_info_map[x_leftind] == 0 or m_local_info_map[x_rightind] == 0 or 
            m_local_info_map[z_leftind] == 0 or m_local_info_map[z_rightind] == 0)
        {
            // m_local_cost_map[keyind] = 5; // close to obstacle, add cost for planning
            continue;
        }

        // share edge
        int x_left_y_left_ind = multi_xl + multi_yl + key.z();
        int x_right_y_left_ind = multi_xr + multi_yl + key.z();
        int x_left_y_right_ind = multi_xl + multi_yr + key.z();
        int x_right_y_right_ind = multi_xr + multi_yr + key.z();
        int y_left_z_left_ind = key.x()*multi1 + multi_yl + z_left;
        int y_right_z_left_ind = key.x()*multi1 + multi_yr + z_left;
        int y_left_z_right_ind = key.x()*multi1 + multi_yl + z_rigt;
        int y_right_z_right_ind = key.x()*multi1 + multi_yr + z_rigt;
        int z_left_x_left_ind = multi_xl + multi_y + z_left;
        int z_right_x_left_ind = multi_xl + multi_y + z_rigt;
        int z_left_x_right_ind = multi_xr + multi_y + z_left;
        int z_right_x_right_ind = multi_xr + multi_y + z_rigt;

        if (m_local_info_map[x_left_y_left_ind] == 0 or m_local_info_map[x_right_y_left_ind] == 0 or m_local_info_map[x_left_y_right_ind] == 0 or m_local_info_map[x_right_y_right_ind] == 0 or
            m_local_info_map[y_left_z_left_ind] == 0 or m_local_info_map[y_right_z_left_ind] == 0 or m_local_info_map[y_left_z_right_ind] == 0 or m_local_info_map[y_right_z_right_ind] == 0 or 
            m_local_info_map[z_left_x_left_ind] == 0 or m_local_info_map[z_right_x_left_ind] == 0 or m_local_info_map[z_left_x_right_ind] == 0 or m_local_info_map[z_right_x_right_ind] == 0)
        {
            // m_local_cost_map[keyind] = 3; // close to obstacle, add cost for planning
            continue;
        }

        // six faces
        if (m_local_info_map[y_leftind] == -1 or m_local_info_map[y_rightind] == -1 or
            m_local_info_map[x_leftind] == -1 or m_local_info_map[x_rightind] == -1 or 
            m_local_info_map[z_leftind] == -1 or m_local_info_map[z_rightind] == -1)
        {
            it->valid_pose_index = valid_pose_index;
            // local_frontier->push_front(*it);
            
            // m_local_cost_map[keyind] = 2;
            // continue;

            local_frontier->insert(*it);
            count++;
        }

//        bool loop_break = false;
//        for(int i=key.x()-1; i<=key.x()+1 && (!loop_break); i+=2){
//            for(int j=key.y()-1; j<=key.y()+1 && (!loop_break); j+=2){
//                for(int k=key.z()-1; k<=key.z()+1 && (!loop_break); k+=2){
//                    if(m_local_info_map[i*size.y()*size.z()+j*size.z()+k] == -1){
////                        if(key==Point3i(i,j,k)){
////                            continue;
////                        }
//                        local_frontier->push_front(*it);
//                        m_local_cost_map[key.x()*size.y()*size.z()+key.y()*size.z()+key.z()] = 2;
//                        loop_break = true;
//                    }

//                } // end loop k
//            } // end loop j
//        } // end loop i

    } // finish detection

    ROS_INFO("detect_local_frontier: count = %d", count);

    // add frontier clustering using graph cut with path finding at the same time

    local_frontier_ptr = local_frontier;
    if(m_local_frontier_pub.getNumSubscribers() > 0){
        this->publish_local_frontier_node();
    }



}

void FrontierBaseline::merge_local_frontier(node_list_ptr local_frontier)
{
    // remove previously detected frontier in sensor range
    for(auto it = global_frontier_ptr->begin(); it != global_frontier_ptr->end();){
        // see if this condition is sufficient?
        const Point3d &ref = it->global_coor;
        
        // ROS_INFO("local distance is (%f, %f)",
        //      ref.x()-curr_orig.x(), ref.y()-curr_orig.y());
        // if((Point3d(ref.x(), ref.y(), 0.0)-curr_orig).norm() < m_sensor_range){ // this condition is wrong

        // if(abs(ref.x()-curr_orig.x()) < m_sensor_range && abs(ref.y()-curr_orig.y()) < m_sensor_range){
        //     it = global_frontier_ptr->erase(it);
        // }else{
        //     // other code
        //     it++;
        // }

        const OcTreeNode* node_temp_free = temp_free_map_ptr->search(ref.x(), ref.y(), ref.z(), m_bbx_depth);
        if ( node_temp_free ) {
            it = global_frontier_ptr->erase(it);
        } else {
            it++;
        }

    }

    // record the positions of local frontiers
    local_frontier_size = local_frontier->size();

    // merge current local frontier (insert before std::begin)
    // global_frontier_ptr->splice(std::begin(*global_frontier_ptr), *local_frontier); // amigo: how to avoid duplication??
    global_frontier_ptr->insert( local_frontier->begin(), local_frontier->end() );
    ROS_INFO("merge::number of local-global frontier: (%d, %d)",
             (int)local_frontier_size, (int)global_frontier_ptr->size());

    // visualize the frontier
    if(m_global_frontier_pub.getNumSubscribers() > 0){
        this->publish_global_frontier_node();
    }


}

void FrontierBaseline::publish_free_node(std::vector<BBXNode> &vector_free)
{
    visualization_msgs::Marker coor_list_free;
    for(std::vector<BBXNode>::iterator it=vector_free.begin(); it != vector_free.end(); ++it){
        geometry_msgs::Point cube_center;
        cube_center.x = it->global_coor.x();
        cube_center.y = it->global_coor.y();
        cube_center.z = it->global_coor.z();
        coor_list_free.points.push_back(cube_center);
    }
    coor_list_free.color.r = 0.8;
    coor_list_free.color.g = 0.0;
    coor_list_free.color.b = 0.8;
    coor_list_free.color.a = 1.0;
    this->setup_visualizer(coor_list_free);
    m_free_pub.publish(coor_list_free);
}

void FrontierBaseline::publish_cost_node()
{
    visualization_msgs::Marker coor_list_cost;
    // int count = 0;
    // int count1 = 0;
    // int count2 = 0;
    int ignore_shell = 0;
    for(int i=0; i < m_bbx_size.x()*m_bbx_size.y()*m_bbx_size.z(); ++i){

        double color_val = log(m_local_cost_map[i])/5.0;
        if(color_val>1)
        {
            // count2 ++;
            color_val = 1; 
        }

        if (color_val < 0.01) // cost =1, free space
        {
            // count1 ++;
            continue;
        }

        // x*size.y*size.z + y*size.z + z = (x*size.y+y)*size.z+z
        int z = i % m_bbx_size.z();
        int r = i / m_bbx_size.z();
        int y = r % m_bbx_size.y();
        int x = r / m_bbx_size.y();

        if(x<ignore_shell || x>=m_bbx_size.x()-ignore_shell || y<ignore_shell || y>=m_bbx_size.y()-ignore_shell || z<2 || z>=m_bbx_size.z()-2 )
        {
            // ROS_INFO("(%d, %d, %d), (%d, %d, %d) ",x, y, z, m_bbx_size.x(), m_bbx_size.y(), m_bbx_size.z());
            continue;            
        }

        // count += 1;
        // get global coor
        Point3i local_key(x,y,z);
        Point3d local_coor;
        this->key2coor(local_key, local_coor);
        Point3d global_coor = local_coor + curr_orig;

        geometry_msgs::Point cube_center;
        cube_center.x = global_coor.x();
        cube_center.y = global_coor.y();
        cube_center.z = global_coor.z();
        coor_list_cost.points.push_back(cube_center); 

        std_msgs::ColorRGBA cube_color;
        float color_z = z/25.0;
        if(color_z > 1) color_z=1;
        cube_color.r = 0.8;
        cube_color.g = color_z;
        cube_color.b = 0.0;
        cube_color.a = color_val;
        coor_list_cost.colors.push_back(cube_color);

    }
    // ROS_INFO("Visualize cost count: have cost %d, low cost %d, high cost %d", count, count1, count2);

    coor_list_cost.color.r = 0.5;
    coor_list_cost.color.g = 0.5;
    coor_list_cost.color.b = 0.0;
    coor_list_cost.color.a = 0.5;
    this->setup_visualizer(coor_list_cost, 0.5);
    m_cost_pub.publish(coor_list_cost);
}


void FrontierBaseline::publish_unknown_node()
{
    visualization_msgs::Marker coor_list_unknown;
    for(int i=0; i < m_bbx_size.x()*m_bbx_size.y()*m_bbx_size.z(); ++i){
        if(m_local_info_map[i] == -1){

            // x*size.y*size.z + y*size.z + z = (x*size.y+y)*size.z+z
            int z = i % m_bbx_size.z();
            int r = i / m_bbx_size.z();
            int y = r % m_bbx_size.y();
            int x = r / m_bbx_size.y();

            Point3d local_coor;
            this->key2coor(Point3i(x, y, z), local_coor);
            local_coor += curr_orig;
            OcTreeNode* node = global_map_ptr->search(local_coor.x(), local_coor.y(), local_coor.z(), m_bbx_depth);
            if(node != nullptr){
                ROS_ERROR("unknown::unconsitant case found!");
                continue;
            }

            geometry_msgs::Point cube_center;
            cube_center.x = local_coor.x();
            cube_center.y = local_coor.y();
            cube_center.z = local_coor.z();
            coor_list_unknown.points.push_back(cube_center);
        }
    }

    coor_list_unknown.color.r = 0.5;
    coor_list_unknown.color.g = 0.5;
    coor_list_unknown.color.b = 0.5;
    coor_list_unknown.color.a = 1.0;
    this->setup_visualizer(coor_list_unknown);
    m_unknown_pub.publish(coor_list_unknown);
}

void FrontierBaseline::publish_global_frontier_node()
{
    visualization_msgs::Marker coor_list_frontier;
    for(const auto& g_frontier : *global_frontier_ptr){
        geometry_msgs::Point cube_center;
        cube_center.x = g_frontier.global_coor.x();
        cube_center.y = g_frontier.global_coor.y();
        cube_center.z = g_frontier.global_coor.z();
        coor_list_frontier.points.push_back(cube_center);
    }
    coor_list_frontier.color.r = 0.0;
    coor_list_frontier.color.g = 0.0;
    coor_list_frontier.color.b = 1.0;
    coor_list_frontier.color.a = 0.9;
    this->setup_visualizer(coor_list_frontier);
    m_global_frontier_pub.publish(coor_list_frontier);
}

void FrontierBaseline::publish_local_frontier_node()
{
    visualization_msgs::Marker coor_list_frontier;
    for(const auto& l_frontier : *local_frontier_ptr){
        geometry_msgs::Point cube_center;
        cube_center.x = l_frontier.global_coor.x();
        cube_center.y = l_frontier.global_coor.y();
        cube_center.z = l_frontier.global_coor.z();
        coor_list_frontier.points.push_back(cube_center);
    }
    coor_list_frontier.color.r = 1.0;
    coor_list_frontier.color.g = 0.0;
    coor_list_frontier.color.b = 1.0;
    coor_list_frontier.color.a = 0.9;
    this->setup_visualizer(coor_list_frontier);
    m_local_frontier_pub.publish(coor_list_frontier);
}

void FrontierBaseline::coor2key(const Point3d &coor, Point3i &key)
{
    // m_offset.z store the negtive z coordinate
    key.x() = int(floor(coor.x()/m_bbx_cell_size) + m_offset.x());
    key.y() = int(floor(coor.y()/m_bbx_cell_size) + m_offset.y());
    key.z() = int(floor(coor.z()/m_bbx_cell_size) + m_offset.z());
}

void FrontierBaseline::key2coor(const Point3i &key, Point3d &coor) // returns center of the cell
{
    coor.x() = (key.x() - m_offset.x() + 0.5f) * m_bbx_cell_size;
    coor.y() = (key.y() - m_offset.y() + 0.5f) * m_bbx_cell_size;
    coor.z() = (key.z() - m_offset.z() + 0.5f) * m_bbx_cell_size;
}

void FrontierBaseline::setup_visualizer(visualization_msgs::Marker &marker, double scale)
{
    // marker.color is supposed to set outside this function
    marker.id = 1;
    marker.type = visualization_msgs::Marker::CUBE_LIST;
    marker.header.frame_id = m_world_frame_id;
    marker.header.stamp = ros::Time::now();
    marker.ns = "map";
    marker.scale.x = m_bbx_cell_size * scale;
    marker.scale.y = m_bbx_cell_size * scale;
    marker.scale.z = m_bbx_cell_size * scale;
    marker.action = visualization_msgs::Marker::ADD;
}

bool FrontierBaseline::check_front_wave(const Point3d &origin, const Point3i &neibor_key)
{
    Point3d neibor_coor;
    this->key2coor(neibor_key, neibor_coor);
    neibor_coor += origin;
    OcTreeNode* node = global_map_ptr->search(neibor_coor.x(), neibor_coor.y(), neibor_coor.z(), m_bbx_depth);
    if(node != nullptr){
        return true;
    }
    return false;
}

// float FrontierBaseline::check_goal_cost(const Point3d goal_pt)
// {
//     Point3d neibor_coor;
//     this->key2coor(neibor_key, neibor_coor);
//     neibor_coor += origin;
//     OcTreeNode* node = global_map_ptr->search(neibor_coor.x(), neibor_coor.y(), neibor_coor.z(), m_bbx_depth);
//     if(node != nullptr){
//         return true;
//     }
//     return false;
// }


void FrontierBaseline::detection_result(data_type::PlannerMessage &result)
{
    ReadLock read_lock(m_rw_lock);

    // local map size
    result.m_local_size = std::make_shared<Point3i>(m_bbx_size);

    // local map origin
    result.m_local_orig = std::make_shared<Point3d>(curr_orig);

    // current robot pose
    result.m_robot_pose = std::make_shared<Point3d>(curr_pose);

    // local frontiers
    result.m_frontier = std::make_shared<std::vector<Point3i>>();
    // auto it = global_frontier_ptr->begin();
    
    for(auto it = local_frontier_ptr->begin(); it != local_frontier_ptr->end(); ++it){
        result.m_frontier->push_back(it->local_index);
    }

    // local cost-map
    int map_len = m_bbx_size.x()*m_bbx_size.y()*m_bbx_size.z();
    result.cost_map_ptr = boost::make_shared<double[]>(map_len);
    std::memcpy(result.cost_map_ptr.get(), m_local_cost_map.get(), map_len*sizeof(double));

    // local info-map
    result.info_map_ptr = boost::make_shared<int[]>(map_len);
    std::memcpy(result.info_map_ptr.get(), m_local_info_map.get(), map_len*sizeof(int));
}

bool FrontierBaseline::detection_ready()
{
    // ROS_WARN("call detection ready !");
    ReadLock read_lock(m_rw_lock);
    // ROS_WARN("wait detection finished !");
    return true;
}

bool FrontierBaseline::get_point_state(Point3d global_point)
{
    OcTreeNode* node = global_map_ptr->search(global_point.x(), global_point.y(), global_point.z(), m_bbx_depth);

    // for unknown cells
    if(node == nullptr){
        return false;
    }


    if(node->getValue() < -0.05){
        return true; // free
    }
    else
    {
        return false; // obstacle
    }

}

int FrontierBaseline::count_neighbor_frontiers(Point3d center_pos, double thresh)
{
    int count = 0;
    double dist;
    for(const auto& g_frontier : *global_frontier_ptr){
        dist = (g_frontier.global_coor - center_pos).norm(); 
        if(dist<=thresh)
        {
            count ++;
        }
    }
    return count;
}


void FrontierBaseline::nearest_global_frontier(Point3d robot_pos, Point3d &res_frontier_coor, Point3d &res_campose)
{
    double min_dist, dist;
    std::size_t campose_ind;

    if(global_frontier_ptr->size()==0)
        return;

    min_dist = COST_INF;
    std::vector<Node_List::iterator> deletelist; // mark for deletion of isolated global frontiers
    int ind = 0;
    for(auto it = global_frontier_ptr->begin(); it != global_frontier_ptr->end(); ++it){
        dist = (it->global_coor - robot_pos).norm(); 
        if(dist<min_dist)
        {
            // delete the frontier if it is isolated
            int neighbor_count;
            neighbor_count = count_neighbor_frontiers(it->global_coor, 2.3 * m_bbx_cell_size);
            if(neighbor_count <= neighbor_count_thresh)
            {
                deletelist.push_back(it);
                ROS_INFO(" ==> Delete isolated frontiers %d, count %d", ind, neighbor_count);
            }
            else
            {
                min_dist = dist;
                res_frontier_coor = it->global_coor;    
                campose_ind = it->valid_pose_index; 
            }
        }

        ind++;
    }

    // delete the global frontiers in the deletelist
    for ( const auto& it : deletelist ) {
        global_frontier_ptr->erase(it);
    }

    if(min_dist == COST_INF)
    {
        ROS_INFO("No nearest global frontier found! ");
        return;
    }

    octomap::point3d campose = valid_poses[campose_ind];
    res_campose = Point3d(campose.x(), campose.y(), campose.z()); 

    ROS_INFO("Find nearest point %f, %f, %f", res_frontier_coor.x(), res_frontier_coor.y(), res_frontier_coor.z());
}

bool FrontierBaseline::find_global_frontier(Point3d frontier_coor, bool delete_after_find)
{

    const octomap::OcTreeKey key = global_map_ptr->coordToKey(frontier_coor.x(), frontier_coor.y(), frontier_coor.z());
    BBXNode frontier_node;
    frontier_node.global_key = key;
    auto it = global_frontier_ptr->find(frontier_node);

    if (it != global_frontier_ptr->end())
    {
        if(delete_after_find)
        {
            global_frontier_ptr->erase(it);
            ROS_WARN("Deleted global frontier (%f, %f, %f), remaining size %d!", frontier_coor.x(), frontier_coor.y(), frontier_coor.z(), global_frontier_ptr->size());
            if(m_global_frontier_pub.getNumSubscribers() > 0){
                this->publish_global_frontier_node();
            }
        }
        return true;
    }
    else
    {
        ROS_WARN("Cannot find the global frontier (%f, %f, %f) in the global set!", frontier_coor.x(), frontier_coor.y(), frontier_coor.z());
        return false;
    }

}

void FrontierBaseline::set_frontier_update_lock(bool flag)
{
    frontier_update_lock = flag;
}



}; // end frontier namespace



