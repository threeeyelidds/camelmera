#ifndef rviz_panel_H_
#define rviz_panel_H_

#include <ros/ros.h>
#include <rviz/panel.h>


#include <octomap_msgs/Octomap.h>
#include <octomap/octomap.h>
#include <octomap_msgs/conversions.h>
#include <octomap_msgs/GetOctomap.h>

#include <octomap_server/OctomapServer.h>

/** 
 *  Include header generated from ui file
 *  Note that you will need to use add_library function first
 *  in order to generate the header file from ui.
 */
#include <ui_data_collection_panel_v2.h>

// Other ROS dependencies
#include <std_msgs/Bool.h>


//for ros service from interactive_cuboid_selection
#include "data_collection_panel/LoadGraph.h"
#include "data_collection_panel/ConfirmAndSendBounds.h"
#include "data_collection_panel/SampleNode.h"
#include "data_collection_panel/PanelSimpleService.h"
#include "data_collection_panel/SampleGraph.h"
#include "data_collection_panel/SamplePath.h"
#include "data_collection_panel/GeneratePose.h"
#include "data_collection_panel/SavePath.h"
#include "data_collection_panel/UpdateCuboidBounds.h"
#include "data_collection_panel/LoadPaths.h"

using namespace visualization_msgs;

namespace rviz_panel
{
    /**
     *  Here we declare our new subclass of rviz::Panel. Every panel which
     *  can be added via the Panels/Add_New_Panel menu is a subclass of
     *  rviz::Panel.
     */

    class dataCollectionPanel : public rviz::Panel
    {
        /**
         * This class uses Qt slots and is a subclass of QObject, so it needs
         * the Q_OBJECT macro.
         */
        Q_OBJECT

        public:
            #ifdef UNIT_TEST
                friend class testClass;
            #endif
            /**
             *  QWidget subclass constructors usually take a parent widget
             *  parameter (which usually defaults to 0).  At the same time,
             *  pluginlib::ClassLoader creates instances by calling the default
             *  constructor (with no arguments). Taking the parameter and giving
             *  a default of 0 lets the default constructor work and also lets
             *  someone using the class for something else to pass in a parent
             *  widget as they normally would with Qt.
             */
            dataCollectionPanel(QWidget * parent = 0);

            /**
             *  Now we declare overrides of rviz::Panel functions for saving and
             *  loading data from the config file.  Here the data is the topic name.
             */
            virtual void save(rviz::Config config) const;
            virtual void load(const rviz::Config & config);



        /**
         *  Next come a couple of public Qt Slots.
         */
        public Q_SLOTS:

        /**
         *  Here we declare some internal slots.
         */
        private Q_SLOTS:

            void load_map_press();
            void load_graph_press();
            void path_to_map_directory();
            void display_sampling_cuboid();
            void hide_sampling_cuboid();
            // void confirm_bounds();
            void sample_node_button();
            void add_node_button();
            void sample_graph_button();
            void sample_path_button_();
            void generate_pose_button_();
            void easy_pose_button_();
            void hard_pose_button_();
            void save_path_button_();
            void reload_graph_button_();
            void delete_node_button();
            void pause_button();
            void load_nodes_button();
            void ok_text_input_button();
            void load_paths_button_();
            void next_path_button_();
            void prev_path_button_();

        /**
         *  Finally, we close up with protected member variables
         */
        protected:
            // UI pointer
            std::shared_ptr<Ui::ui_panel_from_qt> ui_;
            // ROS declaration
            ros::NodeHandle nh_;
            const ros::NodeHandle& private_nh = ros::NodeHandle("~");//octomap_server

            // ros::ServiceClient client_;
            ros::Publisher button_1_pub_;
            ros::Publisher button_2_pub_;
            // ros::Publisher Cuboid_Marker_pub;
            ros::Publisher bounds_from_text_pub_min;
            ros::Publisher bounds_from_text_pub_max;

            ros::ServiceClient load_graph_client;
            ros::ServiceClient reload_graph_client;
            ros::ServiceClient display_cuboid_client;
            ros::ServiceClient hide_cuboid_client;
            ros::ServiceClient confirm_and_send_bounds_client;
            ros::ServiceClient sample_node_client;
            ros::ServiceClient add_node_client;
            ros::ServiceClient sample_graph_client;
            ros::ServiceClient sample_path_client;
            ros::ServiceClient generate_pose_client;
            ros::ServiceClient save_path_client;
            ros::ServiceClient revert_node_client;
            ros::ServiceClient delete_node_client;
            ros::ServiceClient sample_graph_pause_client;
            ros::ServiceClient load_nodes_client;
            ros::ServiceClient load_paths_client;
            ros::ServiceClient next_path_client;
            ros::ServiceClient prev_path_client;

            ros::ServiceServer update_cuboid_text_srv;
            bool update_cuboid_cb(data_collection_panel::UpdateCuboidBounds::Request&,
                  data_collection_panel::UpdateCuboidBounds::Response&);
            
            std_msgs::Bool msg_;

            octomap_server::OctomapServer server_octomap;

            const QString path_name;

            // std::string graph_filename_global;
            std::string launch_filename_global;

            std::string WorkDir;
            // std::string ENVDIR = "/home/wenshan/workspace/ros_tartanair"; // TODO:
            // std::string python_roadmap_script = "python /home/sakura/Airlab_intern_ws/src/sample_pipeline/src/sampling/roadmap_path_sample.py &";


    };
} // namespace rviz_panel

#endif