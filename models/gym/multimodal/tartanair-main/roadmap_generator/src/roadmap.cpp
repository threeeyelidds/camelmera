#include "roadmap_generator/roadmap.h"

#include <omp.h>

namespace roadmap {

Roadmap::Roadmap():
    m_nh(),
    PLANNING_TIME(20),
    RRT_STAR_RANGE(0.2),
    CHECKER_RESOLUTION(0.001),
    OMPL_X_BOUND(10.0), OMPL_Y_BOUND(10.0), OMPL_Z_BOUND(3.0),
    DYNAMIC_EDT_MAXDIST(4.0),
    CLEARANCE_WEIGHT(5.0),
    m_octree_ptr(nullptr),
    m_edtmap_ptr(nullptr),
    m_map_setup(false),
    SAMPLE_QUERY_DEPTH(13),
    CHECKER_FREE_VALUE_THRESH(-1.0),
    CHECKER_DIST_THRESH(1.5),
    MAP_X_MIN(-70), MAP_X_MAX(70),
    MAP_Y_MIN(-60), MAP_Y_MAX(60),
    MAP_Z_MIN(-6), MAP_Z_MAX(6),
    USE_HEIGHT_MAP(true),
    GRID_SIZE(8),
    GRID_SMOOTH_STHENGTH(3),
    PATH_SMOOTH_STEP(20),
    PATH_SMOOTH_RATIO(0.1),
    OPEN_HEIGHT(2),
    SAMPLE_PROB_DROP(0.8),
    SAMPLE_PROB_LOW_THRESH(0.1)
{
    ros::NodeHandle private_nh = ros::NodeHandle("~");

    // planning related parameters
    private_nh.param<double>("ompl_x_bound", OMPL_X_BOUND, OMPL_X_BOUND);
    private_nh.param<double>("ompl_y_bound", OMPL_Y_BOUND, OMPL_Y_BOUND);
    private_nh.param<double>("ompl_z_bound", OMPL_Z_BOUND, OMPL_Z_BOUND);
    private_nh.param<double>("planning_time", PLANNING_TIME, PLANNING_TIME);
    private_nh.param<double>("rrt_star_range", RRT_STAR_RANGE, RRT_STAR_RANGE);
    private_nh.param<double>("checker_resolution", CHECKER_RESOLUTION, CHECKER_RESOLUTION);
    private_nh.param<double>("dynamic_edt_maxdist", DYNAMIC_EDT_MAXDIST, DYNAMIC_EDT_MAXDIST);
    private_nh.param<double>("clearance_weight", CLEARANCE_WEIGHT, CLEARANCE_WEIGHT);

    // define how to sample the endpoints
    private_nh.param<double>("grid_size", GRID_SIZE, GRID_SIZE);
    private_nh.param<int>("grid_smooth_strength", GRID_SMOOTH_STHENGTH, GRID_SMOOTH_STHENGTH);
    private_nh.param<double>("open_height", OPEN_HEIGHT, OPEN_HEIGHT);
    private_nh.param<double>("sample_prob_drop", SAMPLE_PROB_DROP, SAMPLE_PROB_DROP);
    private_nh.param<double>("sample_prob_low_thresh", SAMPLE_PROB_LOW_THRESH, SAMPLE_PROB_LOW_THRESH);

    // smooth the path
    private_nh.param<int>("path_smooth_step", PATH_SMOOTH_STEP, PATH_SMOOTH_STEP);
    private_nh.param<double>("grid_smooth_ratio", PATH_SMOOTH_RATIO, PATH_SMOOTH_RATIO);    

    // environment related patameters
    private_nh.param<int>("sample_query_depth", SAMPLE_QUERY_DEPTH, SAMPLE_QUERY_DEPTH);
    private_nh.param<double>("checker_free_value_thresh", CHECKER_FREE_VALUE_THRESH, CHECKER_FREE_VALUE_THRESH);
    private_nh.param<double>("checker_dist_thresh", CHECKER_DIST_THRESH, CHECKER_DIST_THRESH);

    private_nh.param<double>("map_x_min", MAP_X_MIN, MAP_X_MIN);
    private_nh.param<double>("map_x_max", MAP_X_MAX, MAP_X_MAX);
    private_nh.param<double>("map_y_min", MAP_Y_MIN, MAP_Y_MIN);
    private_nh.param<double>("map_y_max", MAP_Y_MAX, MAP_Y_MAX);
    private_nh.param<double>("map_z_min", MAP_Z_MIN, MAP_Z_MIN);
    private_nh.param<double>("map_z_max", MAP_Z_MAX, MAP_Z_MAX);
    private_nh.param<bool>("use_height_map", USE_HEIGHT_MAP, USE_HEIGHT_MAP);

    // initialize publisher and subscriber
    m_roadmap_pub = m_nh.advertise<visualization_msgs::Marker>("roadmap", 1, true);
    m_octomap_sub = m_nh.subscribe("octomap_full", 1,  &Roadmap::octomap_callback, this);
    m_plan_service = m_nh.advertiseService("roadmap_srv", &Roadmap::roadmap_service, this); // sample path
    m_plan_paths_service = m_nh.advertiseService("plan_paths_srv", &Roadmap::plan_paths_service, this); // sample path
    m_dist_service = m_nh.advertiseService("distmap_srv", &Roadmap::distmap_service, this); // unused
    m_sample_nodes_service = m_nh.advertiseService("sample_nodes_service", &Roadmap::sample_nodes_service, this); // sample nodes
    m_path_smooth_service = m_nh.advertiseService("path_smooth_service", &Roadmap::path_smooth_service, this);

    xgridnum = 0;
    ygridnum = 0;
    accProbMean = 1;
}

bool Roadmap::plan_paths_service(roadmap_generator::roads::Request &req, roadmap_generator::roads::Response &res)
{
    if(m_octree_ptr == nullptr){
        ROS_WARN("plan_paths_service: Octomap is not set!");
        return false;
    }

    auto goallist = req.goallist.poses;
    const int n_goallist = goallist.size();

    // Prepare the response.
    res.roadmaplist.resize(n_goallist);

    // OMP debugging use.
    // const int omp_max_threads = omp_get_max_threads();

    // Generate roadmap in parallel.
#pragma omp parallel for
    for(int i = 0; i < n_goallist; i++ )
    {
        geometry_msgs::PoseArray planres;
        
        const int succ = this->generate_geometry_roadmap(req.init, goallist[i], planres);
        ROS_INFO("planner ind %d, %d",i, succ);
        if(succ < 0) 
            planres.poses.clear();
        res.roadmaplist[i] = planres;

        // OMP debugging use.
        // int tid = omp_get_thread_num();
        // if ( tid == 0 ) {
        //     int threads = omp_get_num_threads();
        //     ROS_INFO("threads = %d, n_goallist = %d", 
        //         threads, n_goallist);
        // }
    }

    return true;
}

bool Roadmap::roadmap_service(roadmap_generator::road::Request &req, roadmap_generator::road::Response &res)
{
    //    ExclusiveLock lock(m_mutex);
    if(m_octree_ptr == nullptr){
        ROS_WARN("roadmap service: Octomap is not set!");
        return false;
    }
    // octomap::OcTreeNode* node = m_octree_ptr->search(req.init.position.x,
    //                                                  req.init.position.y,
    //                                                  req.init.position.z,
    //                                                  SAMPLE_QUERY_DEPTH);

//    if(node == nullptr){ROS_ERROR("init is not free");}
//    else{ROS_ERROR("node value is %f", node->getValue());}

    // call generation function
    int succ = this->generate_geometry_roadmap(req.init, req.goal, res.roadmap);
    if(succ < 0) return false;
    //this->generate_control_roadmap(req.init, req.goal, res.roadmap);

    // publish roadmap
    if(m_roadmap_pub.getNumSubscribers() > 0){
        visualization_msgs::Marker roadmap_maker;
        for(auto it=res.roadmap.poses.begin(); it != res.roadmap.poses.end(); ++it){
            geometry_msgs::Point cube_center;
            cube_center.x = it->position.x;
            cube_center.y = it->position.y;
            cube_center.z = it->position.z;
            roadmap_maker.points.push_back(cube_center);
        }
        roadmap_maker.color.b = 1.0;
        roadmap_maker.color.a = 1.0;
        this->setup_visualizer(roadmap_maker);
        m_roadmap_pub.publish(roadmap_maker);
    }

    return true;
}

void printMat(cv::Mat m, int xlen, int ylen)
{
    for(int i=0; i<xlen; i++)
    {
        for(int j=0; j<ylen; j++)
        {
            std::cout<<m.at<double>(i,j)<<" ";
        }
        std::cout<<std::endl;
    }
    std::cout<< " --- " <<std::endl;

}

void Roadmap::height_range(double range_x_min, double range_x_max, 
                            double range_y_min, double range_y_max, 
                            double range_z_min, double range_z_max )
{
    if(m_octree_ptr == nullptr){
        ROS_WARN("height_range: Octomap is not set!");
        return;
    }

    double height_resolution = 1.0; // TODO: this should be a parameter
    double xmax,ymax,zmax,xmin,ymin,zmin;
    m_octree_ptr->getMetricMax(xmax,ymax,zmax);
    m_octree_ptr->getMetricMin(xmin,ymin,zmin);

    // ROS_INFO("map range: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f",xmax,xmin,ymax,ymin,zmax,zmin);
    ROS_INFO("interested range: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f",range_x_max,range_x_min, range_y_max, range_y_min, range_z_max, range_z_min);
    // calculate how many grids in each direction
    double resolution = m_octree_ptr->getResolution();
    int depth = m_octree_ptr->getTreeDepth();
    int xgridnum_height = ceil((range_x_max - range_x_min)/height_resolution);
    int ygridnum_height = ceil((range_y_max - range_y_min)/height_resolution);
    int zgridnum_height = ceil((zmax - zmin)/height_resolution); // use actual height range instead of interested range
    xgridnum = ceil((range_x_max - range_x_min)/GRID_SIZE); // the final resolution of the height map
    ygridnum = ceil((range_y_max - range_y_min)/GRID_SIZE);
    ROS_INFO("height grids %d, %d, %d", xgridnum_height, ygridnum_height, zgridnum_height);
    // initialize height map
    cv::Mat lowMap_cv(xgridnum_height, ygridnum_height, CV_64F);
    cv::Mat highMap_cv(xgridnum_height, ygridnum_height, CV_64F);
    const double LARGENUM = zmax+1.0;
    const double SMALLNUM = zmin-1.0;
    for (int k=0;k<xgridnum_height;k++)
    {
        for(int j=0;j<ygridnum_height;j++) 
        {
            lowMap_cv.at<double>(k,j) = LARGENUM;
            highMap_cv.at<double>(k,j) = SMALLNUM;
        }
    }
    ROS_INFO("height map initialized");

    int searchDepth_height;
    searchDepth_height = depth - int(log2(height_resolution/resolution));
    ROS_INFO("search height map depth %d", searchDepth_height);
    double xpos,ypos,zpos;
    octomap::OcTreeNode* node;
    for (int i=0;i<xgridnum_height;i++)
    {
        xpos = height_resolution*(i+0.5) + range_x_min;
        for(int j=0;j<ygridnum_height;j++)
        {
            ypos = height_resolution*(j+0.5) + range_y_min;
            for (int k=0;k<zgridnum_height;k++)
            {
                zpos = height_resolution*(k+0.5) + range_z_min;
                // ROS_INFO("Querying node (%.2f, %.2f, %.2f)", xpos, ypos, zpos);
                node = m_octree_ptr->search(xpos,ypos,zpos,searchDepth_height);
                if(node)
                {
                    bool occupied = m_octree_ptr->isNodeOccupied(node); 
                    if (occupied)
                    {
                        if(lowMap_cv.at<double>(i,j)>zpos)
                            lowMap_cv.at<double>(i,j)=zpos;
                        if(highMap_cv.at<double>(i,j)<zpos)
                            highMap_cv.at<double>(i,j)=zpos;
                    }
                }
            }
        }
    }

    // printMat(lowMap_cv, xgridnum_height, ygridnum_height);
    // printMat(highMap_cv, xgridnum_height, ygridnum_height);
    // cv::Mat img_disp(xgridnum_height*2, ygridnum_height, CV_64F);
    // cv::hconcat(lowMap_cv, highMap_cv, img_disp);
    // cv::imshow("height", img_disp);
    // cv::waitKey(0);

    //assign large value to unknown location on the height map
    for (int i=0;i<xgridnum_height;i++)
    {
        for(int j=0;j<ygridnum_height;j++)
        {
            // set range for the unkown region 
            if(lowMap_cv.at<double>(i,j)==LARGENUM)
                lowMap_cv.at<double>(i,j) = range_z_min;
            if(highMap_cv.at<double>(i,j)==SMALLNUM)
                highMap_cv.at<double>(i,j) = range_z_max;
            // clip the value to interested region
            if(lowMap_cv.at<double>(i,j)>range_z_max)
                lowMap_cv.at<double>(i,j) = range_z_max;
            if(highMap_cv.at<double>(i,j)<range_z_min)
                highMap_cv.at<double>(i,j) = range_z_min;
            // fix the open area
            assert(highMap_cv.at<double>(i,j) >= lowMap_cv.at<double>(i,j));
            if(highMap_cv.at<double>(i,j) - lowMap_cv.at<double>(i,j) < resolution * 3) // some thresh says this is an open area
            {
                lowMap_cv.at<double>(i,j) = std::max(range_z_min, lowMap_cv.at<double>(i,j)-OPEN_HEIGHT);
                highMap_cv.at<double>(i,j) = std::min(range_z_max, highMap_cv.at<double>(i,j)+OPEN_HEIGHT);
            }
        }
    }

    // smooth the height map
    cv::blur(lowMap_cv, lowMap_cv, cv::Size(GRID_SMOOTH_STHENGTH, GRID_SMOOTH_STHENGTH));
    cv::blur(highMap_cv, highMap_cv, cv::Size(GRID_SMOOTH_STHENGTH, GRID_SMOOTH_STHENGTH));

    cv::Mat lowMap_small(xgridnum, ygridnum, CV_64F);
    cv::Mat highMap_small(xgridnum, ygridnum, CV_64F);

    cv::resize(lowMap_cv, lowMap_small, cv::Size(ygridnum, xgridnum));
    cv::resize(highMap_cv, highMap_small, cv::Size(ygridnum, xgridnum));

    cv::blur(lowMap_small, lowMap_small, cv::Size(3, 3));
    cv::blur(highMap_small, highMap_small, cv::Size(3, 3));

    printMat(lowMap_small, xgridnum, ygridnum);
    printMat(highMap_small, xgridnum, ygridnum);

    lowMap = lowMap_small.clone();
    highMap = highMap_small.clone();
}


bool Roadmap::distmap_service(roadmap_generator::distmap::Request &req, roadmap_generator::distmap::Response &res)
{

    using namespace octomap;

    if(m_octree_ptr == nullptr){
        ROS_WARN("distmap service: Octomap is not set!");
        return false;
    }

    // tic
    long start = clock();
    point3d low_bound, high_bound;

    if (req.high_bound.x > req.low_bound.x){
        low_bound.x() = req.low_bound.x;
        high_bound.x() =  req.high_bound.x;

    }else{
        low_bound.x() = req.high_bound.x;
        high_bound.x() = req.low_bound.x;
    }
    if (req.high_bound.y > req.low_bound.y){
        low_bound.y() = req.low_bound.y;
        high_bound.y() =  req.high_bound.y;

    }else{
        low_bound.y() = req.high_bound.y;
        high_bound.y() = req.low_bound.y;
    }


    OcTreeKey lw = m_octree_ptr->coordToKey(low_bound.x(), low_bound.y(), -10);
    OcTreeKey hg = m_octree_ptr->coordToKey(high_bound.x(), high_bound.y(), 10);
    if((lw[0]>hg[0]) || (lw[1]>hg[1]) || (lw[2]>hg[2])){
        ROS_WARN("low bound should be smaller than high bound!");
        return false;
    }

    // extend low and high bound by resolution*8 meters
    OcTreeKey lw_ext(lw[0]-8, lw[1]-8, lw[2]);
    OcTreeKey hg_ext(hg[0]+8, hg[1]+8, hg[2]);
    int grid_x = hg_ext[0] - lw_ext[0] + 1;
    int grid_y = hg_ext[1] - lw_ext[1] + 1;
    int grid_z = hg_ext[2] - lw_ext[2] + 1;

    // the origin and size of local distance map
    double resolution = m_octree_ptr->getResolution();
    point3d map_orig = m_octree_ptr->keyToCoord(lw_ext) - point3d(resolution/2, resolution/2, resolution/2);
    point3d map_size(grid_x, grid_y, grid_z);


    auto &dist_map = res.dist_map;
    dist_map.resize(grid_x*grid_y*grid_z);

    // build up the r-tree
    int tree_node_num = 0;
    std::vector<point3d> free_voxel;
    bgi::rtree<CellPoint, bgi::rstar<16>> rtree;
    for(key_type i = lw_ext[0], x = 0; i <= hg_ext[0]; i++, x++){
        for(key_type j = lw_ext[1], y = 0; j<=hg_ext[1]; j++, y++){
            for(key_type k = lw_ext[2], z = 0; k<=hg_ext[2]; k++, z++){

                OcTreeNode* node = m_octree_ptr->search(OcTreeKey(i, j, k));
                point3d coor = m_octree_ptr->keyToCoord(OcTreeKey(i, j, k));

                if(node==nullptr || node->getValue() >= 0){
                    tree_node_num ++;
                    dist_map[x*grid_y*grid_z + y*grid_z + z].data = -4;
                    rtree.insert(CellPoint(coor.x(), coor.y(), coor.z()));
                }else {
                    free_voxel.push_back(coor);
                }
            }
        }
    }

    long finish_insert = clock();
    double duration = (double)(finish_insert - start) / CLOCKS_PER_SEC;
    ROS_WARN("TIME OF BUILDING TREE IS %f for %d nodes", duration, tree_node_num);


    // query the nearest obstacle or void voxel
    std::vector<CellPoint> result;
    for(auto it = free_voxel.begin(); it != free_voxel.end(); ++it){

        // calculate the distance between query target and result
        OcTreeKey tgt_key = m_octree_ptr->coordToKey(*it);

        if(rtree.empty()){
            dist_map[(tgt_key[0]-lw_ext[0])*grid_y*grid_z +
                    (tgt_key[1]-lw_ext[1])*grid_z +
                    (tgt_key[2]-lw_ext[2])].data = 4;
        }else{
            // query the nearest obstacles
            result.clear();
            rtree.query(bgi::nearest(CellPoint(it->x(), it->y(), it->z()), 1), std::back_inserter(result));
            point3d rst(result[0].get<0>(),result[0].get<1>(), result[0].get<2>());

            dist_map[(tgt_key[0]-lw_ext[0])*grid_y*grid_z +
                    (tgt_key[1]-lw_ext[1])*grid_z +
                    (tgt_key[2]-lw_ext[2])].data = ((*it)-rst).norm();
        }

    }

    long finish_query = clock();
    duration = (double)(finish_query - start) / CLOCKS_PER_SEC;
    ROS_WARN("TIME OF QUERY IS %f for %d nodes", duration, (int)free_voxel.size());

    // test
    //OcTreeKey low_test = m_octree_ptr->coordToKey(req.low_bound.x, req.low_bound.y, req.low_bound.z);
    //OcTreeKey high_test = m_octree_ptr->coordToKey(req.high_bound.x, req.high_bound.y, req.high_bound.z);
    //std::cout << low_test[0]-lw_ext[0] << " "<< low_test[1]-lw_ext[1] << " "<< low_test[2]-lw_ext[2] << std::endl;
    //std::cout << high_test[0]-lw_ext[0] << " "<< high_test[1]-lw_ext[1] << " "<< high_test[2]-lw_ext[2] << std::endl;


    // give responese
    res.resolution.data = resolution;

    res.map_orig.x = map_orig.x();
    res.map_orig.y = map_orig.y();
    res.map_orig.z = map_orig.z();

    res.map_size.x = grid_x;
    res.map_size.y = grid_y;
    res.map_size.z = grid_z;

    return true;

}

bool Roadmap::sample_nodes_service(roadmap_generator::endpoints::Request &req, roadmap_generator::endpoints::Response &res)
{
    if(m_octree_ptr == nullptr){
        ROS_WARN("SampleNodes: Octomap is not set!");
        return false;
    }

    geometry_msgs::PoseArray nodes;
    int pointnum = req.pointnum.data;
    Point3d point;
    ROS_INFO("Sample_nodes_service start, sample num %d", pointnum);
    double range_x_min, range_x_max, range_y_min, range_y_max, range_z_min, range_z_max;
    range_x_min = req.range_minx.data;
    range_x_max = req.range_maxx.data;
    range_y_min = req.range_miny.data;
    range_y_max = req.range_maxy.data;
    range_z_min = req.range_minz.data;
    range_z_max = req.range_maxz.data;


    ROS_INFO("Calculate the height map..");
    height_range(range_x_min, range_x_max, range_y_min, range_y_max, range_z_min, range_z_max);

    cv::Mat accProb_cv(xgridnum, ygridnum, CV_64F);
    for (int k=0;k<xgridnum;k++)
        for(int j=0;j<ygridnum;j++) 
            accProb_cv.at<double>(k,j) = 1.0;
    accProb = accProb_cv.clone();  
    accProbMean = 1.0;
    printMat(accProb, xgridnum, ygridnum);


    for(int i=0; i<pointnum; i++)
    {
        bool suc = sample_node(point, range_x_min, range_x_max, range_y_min, range_y_max, range_z_min, range_z_max);
        geometry_msgs::Pose point_msg;
        point_msg.position.x = point.x();
        point_msg.position.y = point.y();
        point_msg.position.z = point.z();        
        if(suc)
        {
            nodes.poses.push_back(point_msg);
            // ROS_WARN("Added %d pt.. ", i);
        }
        else
        {
            ROS_WARN("Node sample fail.. ");
            res.status.data = 0;
            return false;        
        }
        // maintain the accProb values
        if(accProbMean < SAMPLE_PROB_LOW_THRESH) 
        {
            ROS_WARN("AccPorb multiplied by 2.. ");
            accProb = accProb * 2;
            cv::threshold( accProb, accProb, 1.0, 1.0, 2 ); //Threshold Truncated to 1.0
        }

    }
    res.nodes = nodes;
    res.status.data = 1;
    return true;

}
// assume octomap is ready
// balance the node sampling by accProb
bool Roadmap::sample_node(Point3d & resnode, double range_x_min, double range_x_max, 
                            double range_y_min, double range_y_max, double range_z_min, double range_z_max)
{
    double x,y,z, hmax, hmin, accRate, randnum, dist; 
    int MaxTrail = 1000;
    int xind, yind;
    ompl::RNG rng_;
    octomap::OcTreeNode* node;

   // ROS_WARN("sample_node started.. ");

    printMat(accProb, xgridnum, ygridnum);
    // printMat(lowMap, xgridnum, ygridnum);
    // printMat(highMap, xgridnum, ygridnum);

    for(int i=0;i<MaxTrail;i++)
    {
        x = rng_.uniformReal(range_x_min, range_x_max);
        y = rng_.uniformReal(range_y_min, range_y_max);

        xind = (x - range_x_min)/GRID_SIZE;
        yind = (y - range_y_min)/GRID_SIZE;

        // ROS_INFO("random node (%.2f, %.2f) - (%d, %d) ", x, y, xind, yind);

        accRate = accProb.at<double>(xind, yind);
        accProbMean = 0.9 * accProbMean + 0.1 * accRate; // running average of the accProb value sampled
        randnum = rng_.uniform01();
        if (randnum>accRate)
        {
            ROS_INFO("  drop node (%.2f, %.2f) - (%d, %d), rate %.2f, rand %.2f, mean_prob %.2f ", x, y, xind, yind, accRate, randnum, accProbMean);
            continue;
        }

        if(USE_HEIGHT_MAP)
        {
            hmax = highMap.at<double>(xind, yind);
            hmin = lowMap.at<double>(xind, yind);            
        }
        else
        {
            hmax = range_z_max;
            hmin = range_z_min;
        }        
        // ROS_INFO("height range %.2f, %.2f", hmin, hmax);
        z = rng_.uniformReal(hmin, hmax);
        ROS_INFO("  (%.2f, %.2f, %.2f) , prob %.2f, ave prob %.2f ", x, y,z, accRate, accProbMean);
        node = m_octree_ptr->search(x, y, z, SAMPLE_QUERY_DEPTH);
        octomap::point3d pt_loc(x, y, z);
        dist = m_edtmap_ptr->getDistance(pt_loc);
        if(node != nullptr)
            ROS_INFO(" node value %.2f, dist %.2f", node->getValue(), dist);
        if(node == nullptr || node->getValue() > CHECKER_FREE_VALUE_THRESH || dist < CHECKER_DIST_THRESH)
        {
            continue;
        }
        accProb.at<double>(xind, yind) *= SAMPLE_PROB_DROP; // drop the accept probability by dropParam
        resnode = Point3d(x, y, z);
        ROS_INFO("  sample_node success (%.2f, %.2f, %.2f)", x, y, z);
        return true;
    }
    ROS_WARN("  No available node after %d trials.", MaxTrail);
    return false;
}


void path_ompl2ros(ompl::geometric::PathGeometric ompl_path, geometry_msgs::PoseArray &ros_path)
{
    int ompl_pathlen;
    ompl_pathlen = ompl_path.getStateCount();
    // ROS_INFO("ompl path len %d", ompl_pathlen);
    ros_path.poses.clear();
    for(int i = 0; i < ompl_pathlen; i++){
        geometry_msgs::Pose point;
        point.position.x = ompl_path.getState(i)->as<ob::RealVectorStateSpace::StateType>()->values[0];
        point.position.y = ompl_path.getState(i)->as<ob::RealVectorStateSpace::StateType>()->values[1];
        point.position.z = ompl_path.getState(i)->as<ob::RealVectorStateSpace::StateType>()->values[2];
        // ROS_INFO("point (%f, %f, %f)", point.position.x, point.position.y, point.position.z);
        ros_path.poses.push_back(point);
    }
}

void path_ros2ompl(geometry_msgs::PoseArray ros_path, ompl::geometric::PathGeometric &ompl_path)
{
    int ros_pathlen;
    ros_pathlen = ros_path.poses.size();
    // ompl_path.clear();
    // ROS_INFO("ros path len %d", ros_pathlen);
    for(int i=0; i<ros_pathlen; i++)
    {
        double point[3];
        point[0] = ros_path.poses[i].position.x;
        point[1] = ros_path.poses[i].position.y;
        point[2] = ros_path.poses[i].position.z;
        // ROS_INFO("point (%f, %f, %f)", point[0], point[1], point[2]);
        ompl::base::RealVectorStateSpace::StateType state;
        state.values = point;
        ompl_path.append(&state);
        // ompl_path.append();
    }
}

bool Roadmap::path_smooth_service(roadmap_generator::smooth::Request &req, roadmap_generator::smooth::Response &res)
{
    if(m_octree_ptr == nullptr){
        ROS_WARN("roadmap service: Octomap is not set!");
        return false;
    }

    namespace ob = ompl::base;
    namespace og = ompl::geometric;

    og::PathGeometric ompl_path(m_spaceinfo_ptr);
    ROS_INFO("Init ompl path..");
    path_ros2ompl(req.roadmap, ompl_path);
    // ROS_INFO("ros path to ompl path");
    bool suc = m_pathsimplifier_ptr->shortcutPath(ompl_path, PATH_SMOOTH_STEP, 0, PATH_SMOOTH_RATIO);
    ROS_INFO("reduce vertices to %d", ompl_path.getStateCount());
    m_pathsimplifier_ptr->smoothBSpline(ompl_path, 3);
    ROS_INFO("smooth bspline %d", ompl_path.getStateCount());
    
    path_ompl2ros(ompl_path, res.smooth_roadmap);
    // ROS_INFO("ompl path to ros path");
    // if(suc)
    ROS_INFO("Path smooth from %d to %d", req.roadmap.poses.size(), res.smooth_roadmap.poses.size());

    return true;
}


double Roadmap::generate_geometry_roadmap(geometry_msgs::Pose &init_pose, geometry_msgs::Pose &goal_pose, geometry_msgs::PoseArray &path)
{
    namespace ob = ompl::base;
    namespace og = ompl::geometric;

//    ROS_WARN("init is (%f, %f, %f)", init_pose.position.x, init_pose.position.y, init_pose.position.z);
    // ROS_WARN("goal is (%f, %f, %f)", goal_pose.position.x, goal_pose.position.y, goal_pose.position.z);

    //Setting up the state space
    ob::StateSpacePtr xyz_space(new ob::RealVectorStateSpace(3));

    // set bounds for state space
    ob::RealVectorBounds bounds(3);
    double low, hgh;
    low = std::min(init_pose.position.x, goal_pose.position.x)-OMPL_X_BOUND;
    hgh = std::max(init_pose.position.x, goal_pose.position.x)+OMPL_X_BOUND;
    bounds.setLow(0, (low > MAP_X_MIN)?low:MAP_X_MIN);
    bounds.setHigh(0, (hgh < MAP_X_MAX)?hgh:MAP_X_MAX);

    low = std::min(init_pose.position.y, goal_pose.position.y)-OMPL_Y_BOUND;
    hgh = std::max(init_pose.position.y, goal_pose.position.y)+OMPL_Y_BOUND;
    bounds.setLow(1, (low > MAP_Y_MIN)?low:MAP_Y_MIN);
    bounds.setHigh(1, (hgh < MAP_Y_MAX)?hgh:MAP_Y_MAX);

    low = std::min(init_pose.position.z, goal_pose.position.z)-OMPL_Z_BOUND;
    hgh = std::max(init_pose.position.z, goal_pose.position.z)+OMPL_Z_BOUND;
    bounds.setLow(2, (low > MAP_Z_MIN)?low:MAP_Z_MIN);
    bounds.setHigh(2, (hgh < MAP_Z_MAX)?hgh:MAP_Z_MAX);

    xyz_space->as<ob::RealVectorStateSpace>()->setBounds(bounds);
    xyz_space->as<ob::RealVectorStateSpace>()->setStateSamplerAllocator(
                boost::bind(&Roadmap::alloc_octomap_sampler, this, _1));


    // define a space information class
    ob::SpaceInformationPtr si(new ob::SpaceInformation(xyz_space));
    // set state validity checker for this xyz_space
    si->setStateValidityChecker(ob::StateValidityCheckerPtr(new OctomapChecker(si, m_octree_ptr, m_edtmap_ptr)));
    // the RRTStar defaultly use a DiscreteMotionValidator
    si->setStateValidityCheckingResolution(CHECKER_RESOLUTION);

    // set valide state sampler for xyz_space
    // inside SpaceInfomation, the function will be called with
    //    si->setValidStateSamplerAllocator();
    //    si->printSettings();
    //    si->printProperties();

    // define a planning problem on the sapce
    ob::ProblemDefinitionPtr pdef(new ob::ProblemDefinition(si));

    // set the start and goal states
    ob::ScopedState<> init_state(xyz_space);
    ob::ScopedState<> goal_state(xyz_space);
    init_state->as<ob::RealVectorStateSpace::StateType>()->values[0] = init_pose.position.x;
    init_state->as<ob::RealVectorStateSpace::StateType>()->values[1] = init_pose.position.y;
    init_state->as<ob::RealVectorStateSpace::StateType>()->values[2] = init_pose.position.z;
    goal_state->as<ob::RealVectorStateSpace::StateType>()->values[0] = goal_pose.position.x;
    goal_state->as<ob::RealVectorStateSpace::StateType>()->values[1] = goal_pose.position.y;
    goal_state->as<ob::RealVectorStateSpace::StateType>()->values[2] = goal_pose.position.z;
    pdef->setStartAndGoalStates(init_state, goal_state);

    // Setting up the objective cost functions
    boost::shared_ptr<ob::MultiOptimizationObjective> obj(new ob::MultiOptimizationObjective(si));
    ob::OptimizationObjectivePtr len_obj(new ob::PathLengthOptimizationObjective(si));
    ob::OptimizationObjectivePtr clear_obj(new ClearanceObjective(si));
    obj->addObjective(len_obj, 1.0);
    obj->addObjective(clear_obj, CLEARANCE_WEIGHT);
    pdef->setOptimizationObjective(ob::OptimizationObjectivePtr(obj));

    // define optimal planner
    ob::PlannerPtr planner(new og::RRTstar(si));
    planner->as<og::RRTstar>()->setRange(RRT_STAR_RANGE);
    planner->setProblemDefinition(pdef);
    planner->setup();
    //    planner->printProperties(std::cout);

    ob::PlannerStatus solved = planner->ob::Planner::solve(PLANNING_TIME);
    if (solved){

        //Defining the planning problem
        boost::shared_ptr<og::PathGeometric> ompl_path = boost::static_pointer_cast<og::PathGeometric>(
                    planner->getProblemDefinition()->getSolutionPath());

        // int inter_num = ompl_path->length() / 0.1;
        // ompl_path->interpolate(inter_num); // the total number of the path
        //pdef->getSolutionPath()->print(std::cout);

        ob::Cost total_cost = ompl_path->cost(obj);
        ROS_INFO("total cost = %f", total_cost.value());
        ob::Cost clearance_cost = ompl_path->cost(clear_obj);
        ROS_INFO("clearance cost = %f", clearance_cost.value());
        ob::Cost length_cost = ompl_path->cost(len_obj);
        ROS_INFO("length cost = %f", length_cost.value());

        double smoothness = ompl_path->smoothness();
        ROS_INFO("smoothness = %f", smoothness);
        ROS_INFO("total cost = %f", total_cost.value());

        path.poses.clear();
        int num_frames_inpath = (int) ompl_path->getStates().size();
        ROS_INFO("path size = %d", num_frames_inpath);
        if (num_frames_inpath < 1) return -1;

        for(int i = 0; i < num_frames_inpath; i++){
            geometry_msgs::Pose point;
            point.position.x = ompl_path->getState(i)->as<ob::RealVectorStateSpace::StateType>()->values[0];
            point.position.y = ompl_path->getState(i)->as<ob::RealVectorStateSpace::StateType>()->values[1];
            point.position.z = ompl_path->getState(i)->as<ob::RealVectorStateSpace::StateType>()->values[2];
            path.poses.push_back(point);
        }
        int goalind = num_frames_inpath-1;
        octomap::point3d goaldist(goal_pose.position.x-path.poses[goalind].position.x,
                                  goal_pose.position.y-path.poses[goalind].position.y,
                                  goal_pose.position.z-path.poses[goalind].position.z);
        double goaldist_norm;
        goaldist_norm = goaldist.norm();
        ROS_INFO("target goal is (%.3f, %.3f, %.3f)", goal_pose.position.x, goal_pose.position.y, goal_pose.position.z);
        ROS_INFO("planed goal is (%.3f, %.3f, %.3f)", path.poses[goalind].position.x, path.poses[goalind].position.y, path.poses[goalind].position.z);
        ROS_INFO("goal dist is %.3f",goaldist_norm);
        if (goaldist_norm > 0.1) return -1;

        if (total_cost.value() > 1000) return -1;

        octomap::point3d dist(init_pose.position.x-goal_pose.position.x,
                              init_pose.position.y-goal_pose.position.y,
                              init_pose.position.z-goal_pose.position.z);

        return total_cost.value() / dist.norm();

    }else{
        ROS_ERROR("No solution found");
        return -1;
    }

}

double Roadmap::generate_control_roadmap(geometry_msgs::Pose &init_pose,
                                         geometry_msgs::Pose &goal_pose,
                                         geometry_msgs::PoseArray &path)
{
    namespace ob = ompl::base;
    namespace og = ompl::geometric;

    //Setting up the state space
    boost::shared_ptr<ob::CompoundStateSpace> xyzyaw_space(new ompl::base::CompoundStateSpace());
    ob::StateSpacePtr real(new ob::RealVectorStateSpace(3));
    ob::StateSpacePtr so2(new ob::SO2StateSpace());
    xyzyaw_space->addSubspace(real, 1.0);
    xyzyaw_space->addSubspace(so2, 1.0);

    // set bounds for state space
    ob::RealVectorBounds bounds(3);
    if (goal_pose.position.x > init_pose.position.x){
        bounds.setLow(0,init_pose.position.x-OMPL_X_BOUND);
        bounds.setHigh(0,goal_pose.position.x+OMPL_X_BOUND);
    }else{
        bounds.setLow(0,goal_pose.position.x-OMPL_X_BOUND);
        bounds.setHigh(0,init_pose.position.x+OMPL_X_BOUND);
    }
    if (goal_pose.position.y > init_pose.position.y){
        bounds.setLow(1,init_pose.position.y-OMPL_Y_BOUND);
        bounds.setHigh(1,goal_pose.position.y+OMPL_Y_BOUND);
    }else{
        bounds.setLow(1,goal_pose.position.y-OMPL_Y_BOUND);
        bounds.setHigh(1,init_pose.position.y+OMPL_Y_BOUND);
    }
    if (goal_pose.position.z > init_pose.position.z){
        bounds.setLow(2,init_pose.position.z-OMPL_Z_BOUND);
        bounds.setHigh(2,goal_pose.position.z+OMPL_Z_BOUND);
    }else{
        bounds.setLow(2,goal_pose.position.z-OMPL_Z_BOUND);
        bounds.setHigh(2,init_pose.position.z+OMPL_Z_BOUND);
    }
    xyzyaw_space->as<ob::RealVectorStateSpace>(0)->setBounds(bounds);
    xyzyaw_space->as<ob::RealVectorStateSpace>(0)->setStateSamplerAllocator(
                boost::bind(&Roadmap::alloc_octomap_sampler, this, _1));


    // define control space [linear vel and yaw rate wrt body frame and z]
    auto cspace = boost::make_shared<oc::RealVectorControlSpace>(xyzyaw_space, 3);

    // set the bounds for the control space
    ob::RealVectorBounds cbounds(3);
    cbounds.setLow(-0.5);
    cbounds.setHigh(0.5);
    cbounds.setLow(1, -3.14); // yaw rate
    cbounds.setHigh(1, 3.14);
    cspace->setBounds(cbounds);

    // simple set up
    oc::SimpleSetup ss(cspace);

    // define a space information class
    oc::SpaceInformationPtr si = ss.getSpaceInformation();

    // set state validity checker for this xyz_space
    si->setStateValidityChecker(ob::StateValidityCheckerPtr(new OctomapChecker(si, m_octree_ptr, m_edtmap_ptr)));

    // the RRTStar defaultly use a DiscreteMotionValidator
    si->setStateValidityCheckingResolution(CHECKER_RESOLUTION);

    // Setting the propagation routine for this space: does NOT use ODESolver
    oc::StatePropagatorPtr vehicle_model = boost::make_shared<KinematicModel>(si);
    si->setStatePropagator(vehicle_model);
    //    si->setValidStateSamplerAllocator();



    // set the start and goal states
    ob::ScopedState<> init_state(real+so2);
    ob::ScopedState<> goal_state(real+so2);

    init_state->as<ob::CompoundState>()->as<ob::RealVectorStateSpace::StateType>(0)->values[0] = init_pose.position.x;
    init_state->as<ob::CompoundState>()->as<ob::RealVectorStateSpace::StateType>(0)->values[1] = init_pose.position.y;
    init_state->as<ob::CompoundState>()->as<ob::RealVectorStateSpace::StateType>(0)->values[2] = init_pose.position.z;
    init_state->as<ob::CompoundState>()->as<ob::SO2StateSpace::StateType>(1)->value = 0;

    goal_state->as<ob::CompoundState>()->as<ob::RealVectorStateSpace::StateType>(0)->values[0] = goal_pose.position.x;
    goal_state->as<ob::CompoundState>()->as<ob::RealVectorStateSpace::StateType>(0)->values[1] = goal_pose.position.y;
    goal_state->as<ob::CompoundState>()->as<ob::RealVectorStateSpace::StateType>(0)->values[2] = goal_pose.position.z;
    goal_state->as<ob::CompoundState>()->as<ob::SO2StateSpace::StateType>(1)->value = 0;

    ss.setStartAndGoalStates(init_state, goal_state, 0.05);


    // Setting up the objective cost functions
    boost::shared_ptr<ob::MultiOptimizationObjective> obj(new ob::MultiOptimizationObjective(si));
    ob::OptimizationObjectivePtr len_obj(new ob::PathLengthOptimizationObjective(si));
    //    ob::OptimizationObjectivePtr clear_obj(new ClearanceObjective(si));
    obj->addObjective(len_obj, 1.0);
    //    obj->addObjective(clear_obj, 2.0);

    ss.getProblemDefinition()->setOptimizationObjective(ob::OptimizationObjectivePtr(obj));
    ss.setup();
    ss.getPlanner()->printSettings(std::cout);


    //    // define optimal planner
    //    ob::PlannerPtr planner(new og::RRTstar(si));
    //    planner->as<og::RRTstar>()->setRange(RRT_STAR_RANGE);
    //    planner->setProblemDefinition(pdef);
    //    planner->setup();
    //    //    planner->printProperties(std::cout);
    //    ob::PlannerStatus solved = planner->ob::Planner::solve(PLANNING_TIME);
    ob::PlannerStatus solved = ss.solve(PLANNING_TIME);


    if (solved){

        //Defining the planning problem
        boost::shared_ptr<og::PathGeometric> ompl_path = boost::static_pointer_cast<og::PathGeometric>(
                    ss.getProblemDefinition()->getSolutionPath());
        //        ompl_path->interpolate(10);
        //        pdef->getSolutionPath()->print(std::cout);
        //  ss.getSolutionPath().asGeometric().printAsMatrix(std::cout);

        ob::Cost total_cost = ompl_path->cost(obj);
        ROS_INFO("total cost = %f", total_cost.value());
        //        ob::Cost clearance_cost = ompl_path->cost(clear_obj);
        //        ROS_INFO("clearance cost = %f", clearance_cost.value());
        ob::Cost length_cost = ompl_path->cost(len_obj);
        ROS_INFO("length cost = %f", length_cost.value());

        double smoothness = ompl_path->smoothness();
        ROS_INFO("smoothness = %f", smoothness);
        ROS_INFO("total cost = %f", total_cost.value());
        ROS_INFO("goal pose = (%f,%f,%f)", goal_pose.position.x, goal_pose.position.y, goal_pose.position.z);

        path.poses.clear();
        int num_frames_inpath = (int) ompl_path->getStates().size();
        ROS_INFO("path size = %d", num_frames_inpath);
        for(int i = 0; i < num_frames_inpath; i++){
            geometry_msgs::Pose point;
            point.position.x = ompl_path->getState(i)->as<ob::CompoundState>()
                    ->as<ob::RealVectorStateSpace::StateType>(0)->values[0];
            point.position.y = ompl_path->getState(i)->as<ob::CompoundState>()
                    ->as<ob::RealVectorStateSpace::StateType>(0)->values[1];
            point.position.z = ompl_path->getState(i)->as<ob::CompoundState>()
                    ->as<ob::RealVectorStateSpace::StateType>(0)->values[2];
            path.poses.push_back(point);
        }
        if (total_cost.value() > 1000) return -1;
        octomap::point3d dist(init_pose.position.x-goal_pose.position.x,
                              init_pose.position.y-goal_pose.position.y,
                              init_pose.position.z-goal_pose.position.z);

        return total_cost.value() / dist.norm();

    }else{
        ROS_ERROR("No solution found");
        return -1;
    }

}

ob::StateSamplerPtr Roadmap::alloc_octomap_sampler(const ob::StateSpace *state)
{
    return boost::make_shared<OctomapSampler>(state, m_octree_ptr, SAMPLE_QUERY_DEPTH);
}


void Roadmap::octomap_callback(const octomap_msgs::OctomapPtr &octomap_msg)
{
    using namespace octomap;
    if(m_map_setup){
        return;
    }

    // ExclusiveLock lock(m_mutex);
    m_octree_ptr = boost::make_shared<OcTree>(*dynamic_cast<OcTree*>(octomap_msgs::fullMsgToMap(*octomap_msg)));
    ROS_INFO("Octomap is received and start Euclidean Distance Transformation ...");
    octomap::point3d bbxMin(MAP_X_MIN, MAP_Y_MIN, MAP_Z_MIN);
    octomap::point3d bbxMax(MAP_X_MAX, MAP_Y_MAX, MAP_Z_MAX);
    m_edtmap_ptr = boost::make_shared<DynamicEDTOctomap>(DYNAMIC_EDT_MAXDIST, dynamic_cast<OcTree*>(octomap_msgs::fullMsgToMap(*octomap_msg)),
                                                         bbxMin, bbxMax, true);
    // if false, square root will be omitted thus being faster, but can only query squared dist
    m_edtmap_ptr->update(true);
    ROS_INFO("Distance transform finished!");
    m_map_setup = true;
    m_octomap_sub.shutdown();

    ROS_INFO("Start to calculate query index bound at specific tree depth!");
    // extract query level index
    m_tree_max_depth = m_octree_ptr->getTreeDepth();
    m_tree_resolution = m_octree_ptr->getResolution();
    m_bbx_cell_size = m_tree_resolution * pow(2, m_tree_max_depth - SAMPLE_QUERY_DEPTH);
    assert(m_tree_resolution > 0 && m_tree_max_depth <=16);
    assert(m_bbx_cell_size >= m_tree_resolution);

    // get index bound in the specified query level
    // [####|####] when the bound is located exactly at cell boundary,
    // we get the stract interior by substracting tree_resolution/2
    // [#|####|#] for such case, it turns [+|####|+] after substraction,
    // we get an extra grid at one or two side, which will be regarded as
    // unfree since it contains unknown area
    // m_query_x_min = (int)floor((MAP_X_MIN+m_tree_resolution/2)/m_bbx_cell_size);
    // m_query_x_max = (int)floor((MAP_X_MAX-m_tree_resolution/2)/m_bbx_cell_size);
    // m_query_y_min = (int)floor((MAP_Y_MIN+m_tree_resolution/2)/m_bbx_cell_size);
    // m_query_y_max = (int)floor((MAP_Y_MAX-m_tree_resolution/2)/m_bbx_cell_size);
    // m_query_z_min = (int)floor((MAP_Z_MIN+m_tree_resolution/2)/m_bbx_cell_size);
    // m_query_z_max = (int)floor((MAP_Z_MAX-m_tree_resolution/2)/m_bbx_cell_size);

    ROS_INFO("Create a pathsimplifier..");
    namespace ob = ompl::base;
    namespace og = ompl::geometric;
    //Setting up the state space
    ob::StateSpacePtr xyz_space(new ob::RealVectorStateSpace(3));
    // set bounds for state space
    ob::RealVectorBounds bounds(3);
    bounds.setLow(0, MAP_X_MIN); bounds.setHigh(0, MAP_X_MAX);
    bounds.setLow(1, MAP_Y_MIN); bounds.setHigh(1, MAP_Y_MAX);
    bounds.setLow(2, MAP_Z_MIN); bounds.setHigh(2, MAP_Z_MAX);

    xyz_space->as<ob::RealVectorStateSpace>()->setBounds(bounds);
    xyz_space->as<ob::RealVectorStateSpace>()->setStateSamplerAllocator(
                boost::bind(&Roadmap::alloc_octomap_sampler, this, _1));

    // define a space information class
    // ob::SpaceInformationPtr si(new ob::SpaceInformation(xyz_space));
    m_spaceinfo_ptr = boost::make_shared<ob::SpaceInformation>(xyz_space);
    // set state validity checker for this xyz_space
    m_spaceinfo_ptr->setStateValidityChecker(ob::StateValidityCheckerPtr(new OctomapChecker(m_spaceinfo_ptr, m_octree_ptr, m_edtmap_ptr)));
    // the RRTStar defaultly use a DiscreteMotionValidator
    m_spaceinfo_ptr->setStateValidityCheckingResolution(CHECKER_RESOLUTION);

    if (!m_spaceinfo_ptr->isSetup())
    {
        m_spaceinfo_ptr->setup();
    }    
    m_pathsimplifier_ptr = boost::make_shared<og::PathSimplifier>(m_spaceinfo_ptr);

    ROS_INFO("Finish map initialization! this subscriber will be shutdown!");


}

void Roadmap::setup_visualizer(visualization_msgs::Marker &marker)
{
    // marker.color is supposed to set outside this function
    marker.id = 1;
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.ns = "map";
    marker.scale.x = 0.3;
    marker.scale.y = 0.3;
    marker.scale.z = 0.3;
    marker.action = visualization_msgs::Marker::ADD;
}


};

