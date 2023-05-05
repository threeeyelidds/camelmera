#include "data_collection_panel/data_collection_panel.hpp"
#include <pluginlib/class_list_macros.hpp>
#include <visualization_msgs/InteractiveMarkerFeedback.h>


#include <QFileDialog>
#include <QMessageBox>

#include <sstream>
#include <fstream>
#include <iostream>

using namespace visualization_msgs;

PLUGINLIB_EXPORT_CLASS(rviz_panel::dataCollectionPanel, rviz::Panel)

namespace rviz_panel
{
    dataCollectionPanel::dataCollectionPanel(QWidget * parent)
    :   rviz::Panel(parent),
        ui_(std::make_shared<Ui::ui_panel_from_qt>())
    {
        // Extend the widget with all attributes and children from UI file
        ui_->setupUi(this);

        ros::spinOnce(); //octomap_server
        // Define ROS publisher
        button_1_pub_ = nh_.advertise<std_msgs::Bool>("button_1_topic", 1);
        button_2_pub_ = nh_.advertise<std_msgs::Bool>("button_2_topic", 1);
        // Cuboid_Marker_pub = nh_.advertise<visualization_msgs::Marker>("/sampling_cuboid_marker", 2); // TODO: what use? 
        bounds_from_text_pub_min = nh_.advertise<visualization_msgs::InteractiveMarkerFeedback>("/two_marker/feedback", 10);
        bounds_from_text_pub_max = nh_.advertise<visualization_msgs::InteractiveMarkerFeedback>("/two_marker/feedback", 10);


        load_graph_client = nh_.serviceClient<data_collection_panel::LoadGraph>("load_graph");
        reload_graph_client = nh_.serviceClient<data_collection_panel::PanelSimpleService>("reload_graph");
        display_cuboid_client = nh_.serviceClient<data_collection_panel::PanelSimpleService>("display_cuboid");
        hide_cuboid_client = nh_.serviceClient<data_collection_panel::PanelSimpleService>("hide_cuboid");
        confirm_and_send_bounds_client = nh_.serviceClient<data_collection_panel::ConfirmAndSendBounds>("confirm_and_send_bounds");
        sample_node_client = nh_.serviceClient<data_collection_panel::SampleNode>("sample_node");
        add_node_client = nh_.serviceClient<data_collection_panel::PanelSimpleService>("add_node");
        sample_graph_client = nh_.serviceClient<data_collection_panel::SampleGraph>("sample_graph");
        sample_graph_pause_client = nh_.serviceClient<data_collection_panel::PanelSimpleService>("sample_graph_pause");
        sample_path_client = nh_.serviceClient<data_collection_panel::SamplePath>("sample_path");
        generate_pose_client = nh_.serviceClient<data_collection_panel::GeneratePose>("generate_pose");
        save_path_client = nh_.serviceClient<data_collection_panel::SavePath>("save_path");
        delete_node_client = nh_.serviceClient<data_collection_panel::SampleNode>("delete_node");
        load_nodes_client = nh_.serviceClient<data_collection_panel::LoadGraph>("load_nodes_from_graph");
        load_paths_client = nh_.serviceClient<data_collection_panel::LoadPaths>("load_paths");
        next_path_client = nh_.serviceClient<data_collection_panel::PanelSimpleService>("next_path");
        prev_path_client = nh_.serviceClient<data_collection_panel::PanelSimpleService>("prev_path");
        // update the text for cuboid selection
        update_cuboid_text_srv = nh_.advertiseService("update_cuboid_text", &dataCollectionPanel::update_cuboid_cb, this);

        // Declare ROS msg_
        msg_.data = true;

        // connect(ui_->output_path_text, SIGNAL(editingFinished()), this, SLOT(path_to_map_directory()));
        connect(ui_->load_map_button, SIGNAL(clicked()), this, SLOT(load_map_press()));
        connect(ui_->load_graph_button, SIGNAL(clicked()), this, SLOT(load_graph_press()));
        connect(ui_->display_sampling_cuboid, SIGNAL(clicked()), this, SLOT(display_sampling_cuboid()));
        connect(ui_->hide_sampling_cuboid, SIGNAL(clicked()), this, SLOT(hide_sampling_cuboid()));
        // connect(ui_->confirm_bounds, SIGNAL(clicked()), this, SLOT(confirm_bounds()));
        connect(ui_->sample_node_button, SIGNAL(clicked()), this, SLOT(sample_node_button()));
        connect(ui_->add_node_button, SIGNAL(clicked()), this, SLOT(add_node_button()));
        connect(ui_->sample_graph_button, SIGNAL(clicked()), this, SLOT(sample_graph_button()));
        connect(ui_->sample_path_button_, SIGNAL(clicked()), this, SLOT(sample_path_button_())); 
        connect(ui_->generate_pose_button_, SIGNAL(clicked()), this, SLOT(generate_pose_button_()));
        connect(ui_->easy_pose_button_, SIGNAL(clicked()), this, SLOT(easy_pose_button_()));
        connect(ui_->hard_pose_button_, SIGNAL(clicked()), this, SLOT(hard_pose_button_()));
        connect(ui_->save_path_button_, SIGNAL(clicked()), this, SLOT(save_path_button_()));
        connect(ui_->reload_graph_button_, SIGNAL(clicked()), this, SLOT(reload_graph_button_()));
        connect(ui_->delete_node_button, SIGNAL(clicked()), this, SLOT(delete_node_button()));
        connect(ui_->pause_button, SIGNAL(clicked()), this, SLOT(pause_button()));
        connect(ui_->load_nodes_button, SIGNAL(clicked()), this, SLOT(load_nodes_button()));
        connect(ui_->ok_text_input_button, SIGNAL(clicked()), this, SLOT(ok_text_input_button()));

        // new buttons
        connect(ui_->load_paths_button_, SIGNAL(clicked()), this, SLOT(load_paths_button_()));
        connect(ui_->next_path_button_, SIGNAL(clicked()), this, SLOT(next_path_button_()));
        connect(ui_->prev_path_button_, SIGNAL(clicked()), this, SLOT(prev_path_button_()));

        std::string repodir = __FILE__;
        std::size_t found = repodir.find("data_collection_panel");
        WorkDir = repodir.substr(0,found); // src folder of the ros workspace
        // find a config file which stores the previous output dir
        std::string dirhiddenfile = WorkDir + "/data_collection_panel/resource/.output_dir";
        std::ifstream hiddenfile(dirhiddenfile.c_str());
        std::string output_dir;
        if(hiddenfile.good() && hiddenfile.is_open()) //file exists
        {
            getline(hiddenfile, output_dir);
            hiddenfile.close();
            ROS_INFO("Use output dir from config file: %s", output_dir.c_str());
        } 
        else
        {
            output_dir = WorkDir;
            ROS_INFO("Use default output dir: %s", output_dir.c_str());
        }
        ui_->output_path_text->setText(output_dir.c_str());

        ui_->node_num->setText("50");
        ui_->edge_num_input->setText("10");
        ui_->min_distance_input->setText("5");
        ui_->max_distance_input->setText("30");


        ui_->vel_min->setText("0");
        ui_->vel_max->setText("10");
        ui_->acc_max->setText("20");
        
        ui_->yaw_min->setText("-180");
        ui_->yaw_max->setText("180");
        
        ui_->roll_min->setText("-90");
        ui_->roll_max->setText("90");
        
        ui_->pitch_min->setText("-90");
        ui_->pitch_max->setText("90");
        
        ui_->rand_deg->setText("20");
        ui_->smooth->setText("1");

        ui_->ros_path_dir_->setText("ros_path");

        ui_->x_min_edit->setText("-5");
        ui_->x_max_edit->setText("5");
        ui_->y_min_edit->setText("-5");
        ui_->y_max_edit->setText("5");
        ui_->z_min_edit->setText("-2.5");
        ui_->z_max_edit->setText("2.5");

        ui_->pose_folder->setText("pose");
        ui_->min_node_num_->setText("5");
        ui_->max_node_num_->setText("20");

    }

    bool dataCollectionPanel::update_cuboid_cb(data_collection_panel::UpdateCuboidBounds::Request& req,
          data_collection_panel::UpdateCuboidBounds::Response& res)
    {
        std::stringstream range_xmin, range_xmax, range_ymin, range_ymax, range_zmin, range_zmax;
        range_xmin << std::fixed << std::setprecision(1) << req.node_range_xmin;
        range_xmax << std::fixed << std::setprecision(1) << req.node_range_xmax;
        range_ymin << std::fixed << std::setprecision(1) << req.node_range_ymin;
        range_ymax << std::fixed << std::setprecision(1) << req.node_range_ymax;
        range_zmin << std::fixed << std::setprecision(1) << req.node_range_zmin;
        range_zmax << std::fixed << std::setprecision(1) << req.node_range_zmax;
        ui_->x_min_edit->setText(QString::fromStdString(range_xmin.str()));
        ui_->x_max_edit->setText(QString::fromStdString(range_xmax.str()));
        ui_->y_min_edit->setText(QString::fromStdString(range_ymin.str()));
        ui_->y_max_edit->setText(QString::fromStdString(range_ymax.str()));
        ui_->z_min_edit->setText(QString::fromStdString(range_zmin.str()));
        ui_->z_max_edit->setText(QString::fromStdString(range_zmax.str()));

        // dataCollectionPanel::cuboidpos2text();

        ROS_INFO_STREAM( "Range " << range_xmin.str() << ", " << range_xmax.str() << ", " << range_ymin.str() << ", " << range_ymax.str() << ", " << range_zmin.str() << ", " << range_zmax.str()  );
        // std::cout<<"range_xmin"<<range_xmin.str()<<std::endl;
        return true;
    }

    void dataCollectionPanel::ok_text_input_button()
    {
        visualization_msgs::InteractiveMarkerFeedback msg;

        msg.header.frame_id = "map";
        msg.client_id= "/rviz/InteractiveMarkers";
        msg.marker_name = "simple_6dof_min";
        msg.event_type = 1;
        msg.pose.position.x = ui_->x_min_edit->text().toFloat();
        msg.pose.position.y = ui_->y_min_edit->text().toFloat();
        msg.pose.position.z = ui_->z_min_edit->text().toFloat();
        msg.mouse_point_valid = false;
        ROS_INFO("OK Button min set %f, %f, %f", msg.pose.position.x, msg.pose.position.y, msg.pose.position.z);
        this->bounds_from_text_pub_min.publish(msg);

        // sleep(1.0);

        visualization_msgs::InteractiveMarkerFeedback msg1;

        msg1.header.frame_id = "map";
        msg1.client_id= "/rviz/InteractiveMarkers";
        msg1.marker_name = "simple_6dof_max";
        msg1.event_type = 1;
        msg1.pose.position.x = ui_->x_max_edit->text().toFloat();
        msg1.pose.position.y = ui_->y_max_edit->text().toFloat();
        msg1.pose.position.z = ui_->z_max_edit->text().toFloat();
        ROS_INFO("OK Button max set %f, %f, %f", msg1.pose.position.x, msg1.pose.position.y, msg1.pose.position.z);
        msg1.mouse_point_valid = false;
        this->bounds_from_text_pub_max.publish(msg1);


    }

    void dataCollectionPanel::load_nodes_button()
    {
        ROS_INFO_STREAM("Load nodes pressed.");
        this->button_2_pub_.publish(this->msg_);

        if (ui_->output_path_text->text() != "")
        {
            /* code */
            QString graph_filename;
            data_collection_panel::LoadGraph lg_srv;
            graph_filename = QFileDialog::getOpenFileName(this, "Choose the Graph file",ui_->output_path_text->text(), "*.graph");
            lg_srv.request.graph_filename = graph_filename.toStdString();
            load_nodes_client.call(lg_srv);
        }
    }

    void dataCollectionPanel::pause_button()
    {   
        data_collection_panel::PanelSimpleService pa_srv;
        if (sample_graph_pause_client.call(pa_srv))
        {
            ROS_INFO("Pause graph sample.");
        }
        else
        {   
            QMessageBox::warning(this,"Error","Failed to call sample_graph_pause service.");
        }
    }
    

    void dataCollectionPanel::delete_node_button()
    {   
        data_collection_panel::ConfirmAndSendBounds srv;
        float max_x, min_x, max_y, min_y, max_z, min_z;
        if (confirm_and_send_bounds_client.call(srv))
        {
            max_x = srv.response.max_x; 
            min_x = srv.response.min_x; 
            max_y = srv.response.max_y; 
            min_y = srv.response.min_y; 
            max_z = srv.response.max_z; 
            min_z = srv.response.min_z; 

            ROS_INFO("Bounds are retrived successfully.");
        }


        data_collection_panel::SampleNode sn_srv;
        sn_srv.request.node_range_xmin = min_x;
        sn_srv.request.node_range_xmax = max_x;
        sn_srv.request.node_range_ymin = min_y;
        sn_srv.request.node_range_ymax = max_y;
        sn_srv.request.node_range_zmin = min_z;
        sn_srv.request.node_range_zmax = max_z;

        if (delete_node_client.call(sn_srv))
        {
            ROS_INFO("Node delete successfully.");
        }
        else
        {
            QMessageBox::warning(this,"Error","Node delete service failed.");
        }

    }
    
    void dataCollectionPanel::sample_path_button_()
    {
        data_collection_panel::SamplePath sp_srv;
        // sp_srv.request.graph_filename = graph_filename_global;
        sp_srv.request.output_dir = ui_->output_path_text->text().toStdString();
        sp_srv.request.sample_cycle_mode = ui_->sample_mode_combo_->currentIndex();

        sp_srv.request.ros_path_folder = ui_->ros_path_dir_->text().toStdString();
        sp_srv.request.cycle_min_nodes = ui_->min_node_num_->text().toInt(); 
        sp_srv.request.cycle_max_nodes = ui_->max_node_num_->text().toInt(); 
        // sp_srv.request.interactive = ui_->interactive->isChecked();


        if (sample_path_client.call(sp_srv))
        {
            if(sp_srv.response.success_flag)
                ROS_INFO("The new path is sampled");
            else
                QMessageBox::warning(this,"Error","No path on the current graph!");
        }
        else
        {   
            QMessageBox::warning(this,"Error","Failed to call service sample_path. Please ensure the service is being executed.");
        }

    }

    void dataCollectionPanel::save_path_button_()
    {
        data_collection_panel::SavePath sap_srv;
        sap_srv.request.save_path_flag = 1;
        save_path_client.call(sap_srv);
    }

    void dataCollectionPanel::load_paths_button_()
    {
        data_collection_panel::LoadPaths sap_srv;
        sap_srv.request.output_dir = ui_->output_path_text->text().toStdString();
        sap_srv.request.ros_path_folder = ui_->ros_path_dir_->text().toStdString();
        
        if (load_paths_client.call(sap_srv))
        {
            ROS_INFO("Load paths successfully.");
        }
        else
        {
            QMessageBox::warning(this,"Error","Load paths service failed.");
        }
    }

    void dataCollectionPanel::next_path_button_()
    {
        data_collection_panel::PanelSimpleService sap_srv;
        next_path_client.call(sap_srv);
    }

    void dataCollectionPanel::prev_path_button_()
    {
        data_collection_panel::PanelSimpleService sap_srv;
        prev_path_client.call(sap_srv);
    }

    void dataCollectionPanel::reload_graph_button_()
    {
        ROS_INFO_STREAM("Button reload graph pressed.");

        data_collection_panel::PanelSimpleService rg_srv;
        if(reload_graph_client.call(rg_srv))
        {
            if(rg_srv.response.success_flag)
                ROS_INFO("Reload graph successfully.");
            else
                QMessageBox::warning(this,"Error","Cannot reload the graph!");
        }
        else
            QMessageBox::warning(this,"Error","Failed to call reload graph service.");

    }

    void dataCollectionPanel::generate_pose_button_()
    {
        data_collection_panel::GeneratePose gp_srv;

        gp_srv.request.data_dir = ui_->output_path_text->text().toStdString(); 
        gp_srv.request.input_folder = ui_->ros_path_dir_->text().toStdString();
        gp_srv.request.output_folder = ui_->pose_folder->text().toStdString(); 
        gp_srv.request.vel_min = ui_->vel_min->text().toFloat();
        gp_srv.request.vel_max = ui_->vel_max->text().toFloat();
        gp_srv.request.acc_max = ui_->acc_max->text().toFloat();
        
        gp_srv.request.yaw_min = ui_->yaw_min->text().toFloat();
        gp_srv.request.yaw_max = ui_->yaw_max->text().toFloat();
        
        gp_srv.request.roll_min = ui_->roll_min->text().toFloat();
        gp_srv.request.roll_max = ui_->roll_max->text().toFloat();
        
        gp_srv.request.pitch_min = ui_->pitch_min->text().toFloat();
        gp_srv.request.pitch_max = ui_->pitch_max->text().toFloat();
        
        gp_srv.request.rand_degree = ui_->rand_deg->text().toFloat();
        gp_srv.request.smooth_count = ui_->smooth->text().toInt();
        
        if (generate_pose_client.call(gp_srv))
        {
            if (gp_srv.response.success_flag)
                ROS_INFO("Pose Generated");
            else
                QMessageBox::warning(this,"Error","Pose generation failed. ");
        }

        else
        {   
            QMessageBox::warning(this,"Error","Failed to call service generate_pose. Please ensure the service is being executed.");
        }


    }

    void dataCollectionPanel::easy_pose_button_()
    {
        data_collection_panel::GeneratePose gp_srv;

        gp_srv.request.data_dir = ui_->output_path_text->text().toStdString(); 
        gp_srv.request.input_folder = ui_->ros_path_dir_->text().toStdString();
        gp_srv.request.output_folder = "pose_easy"; 
        gp_srv.request.vel_min = 0.0;
        gp_srv.request.vel_max = 5.0;
        gp_srv.request.acc_max = 10.0;
        
        gp_srv.request.yaw_min = -180.0;
        gp_srv.request.yaw_max = 180.0;
        
        gp_srv.request.roll_min = -30.0;
        gp_srv.request.roll_max = 30.0;
        
        gp_srv.request.pitch_min = -45.0;
        gp_srv.request.pitch_max = 30.0;
        
        gp_srv.request.rand_degree = 10.0;
        gp_srv.request.smooth_count = 3;
        
        if (generate_pose_client.call(gp_srv))
        {
            if (gp_srv.response.success_flag)
                ROS_INFO("Easy Pose Generated");
            else
                QMessageBox::warning(this,"Error","Pose generation failed. ");
        }

        else
        {   
            QMessageBox::warning(this,"Error","Failed to call service generate_pose. Please ensure the service is being executed.");
        }

    }

    void dataCollectionPanel::hard_pose_button_()
    {
        data_collection_panel::GeneratePose gp_srv;

        gp_srv.request.data_dir = ui_->output_path_text->text().toStdString(); 
        gp_srv.request.input_folder = ui_->ros_path_dir_->text().toStdString();
        gp_srv.request.output_folder = "pose_hard"; 
        gp_srv.request.vel_min = 0.0;
        gp_srv.request.vel_max = 10.0;
        gp_srv.request.acc_max = 20.0;
        
        gp_srv.request.yaw_min = -180.0;
        gp_srv.request.yaw_max = 180.0;
        
        gp_srv.request.roll_min = -60.0;
        gp_srv.request.roll_max = 60.0;
        
        gp_srv.request.pitch_min = -60.0;
        gp_srv.request.pitch_max = 45.0;
        
        gp_srv.request.rand_degree = 15.0;
        gp_srv.request.smooth_count = 1;
        
        if (generate_pose_client.call(gp_srv))
        {
            if (gp_srv.response.success_flag)
                ROS_INFO("Hard Pose Generated");
            else
                QMessageBox::warning(this,"Error","Pose generation failed. ");
        }

        else
        {   
            QMessageBox::warning(this,"Error","Failed to call service generate_pose. Please ensure the service is being executed.");
        }

    }

    void dataCollectionPanel::sample_graph_button()
    {
        data_collection_panel::SampleGraph sg_srv;

        sg_srv.request.output_dir = ui_->output_path_text->text().toStdString();

        sg_srv.request.edge_num = ui_->edge_num_input->text().toInt();
        sg_srv.request.min_dist_thresh = ui_->min_distance_input->text().toFloat();
        sg_srv.request.max_dist_thresh = ui_->max_distance_input->text().toFloat();

        if (sample_graph_client.call(sg_srv))
        {
            if (sg_srv.response.success_flag)
                ROS_INFO("The new graph is sampled.");
            else
                QMessageBox::warning(this,"Error","Please check if the output folder exists. ");

        }
        else
        {
            QMessageBox::warning(this,"Error","Failed to call service sample_graph. Please ensure the service is being executed.");
        }
    }

    void dataCollectionPanel::add_node_button()
    {
        data_collection_panel::PanelSimpleService an_srv;
        if (add_node_client.call(an_srv))
        {
            ROS_INFO("The new nodes are added.");
        }
        else
        {
            QMessageBox::warning(this,"Error","Failed to call service add_node. Please ensure the service is being executed.");
        }
    }
    void dataCollectionPanel::display_sampling_cuboid()
    {
        data_collection_panel::PanelSimpleService srv;
        if (display_cuboid_client.call(srv))
        {
        ROS_INFO("The selection cubiod is now displayed in rviz");
        }
        else
        {
            QMessageBox::warning(this,"Error","Failed to call service display_cuboid. Please ensure the service is being executed.");
        }
    }
    void dataCollectionPanel::sample_node_button()
    {   data_collection_panel::ConfirmAndSendBounds srv;
        float max_x, min_x, max_y, min_y, max_z, min_z;
        if (confirm_and_send_bounds_client.call(srv))
        {
            max_x = srv.response.max_x; 
            min_x = srv.response.min_x; 
            max_y = srv.response.max_y; 
            min_y = srv.response.min_y; 
            max_z = srv.response.max_z; 
            min_z = srv.response.min_z; 

            ROS_INFO("Bounds are retrived successfully.");
        }


        data_collection_panel::SampleNode sn_srv;
        sn_srv.request.node_num = ui_->node_num->text().toInt();
        sn_srv.request.node_range_xmin = min_x;
        sn_srv.request.node_range_xmax = max_x;
        sn_srv.request.node_range_ymin = min_y;
        sn_srv.request.node_range_ymax = max_y;
        sn_srv.request.node_range_zmin = min_z;
        sn_srv.request.node_range_zmax = max_z;

        if (sample_node_client.call(sn_srv))
        {
            ROS_INFO("Node sampling initiated successfully.");
        }
        else
        {
            QMessageBox::warning(this,"Error","Node sampling service failed.");
        }

    }

    void dataCollectionPanel::hide_sampling_cuboid()
    {   
        data_collection_panel::PanelSimpleService srv;
        if (hide_cuboid_client.call(srv))
        {
            ROS_INFO("The selection cubiod is now hidden.");
        }
        else
        {
            QMessageBox::warning(this,"Error","Failed to call service hide_cuboid. Please ensure the service is being executed.");
        }
    }

    // void dataCollectionPanel::confirm_bounds()
    // {
    //     data_collection_panel::ConfirmAndSendBounds srv;
    //     float max_x, min_x, max_y, min_y, max_z, min_z
    //     if (confirm_and_send_bounds_client.call(srv))
    //     {
    //         max_x_global = srv.response.max_x; 
    //         min_x_global = srv.response.min_x; 
    //         max_y_global = srv.response.max_y; 
    //         min_y_global = srv.response.min_y; 
    //         max_z_global = srv.response.max_z; 
    //         min_z_global = srv.response.min_z; 

    //         ROS_INFO("Bounds are retrived successfully.");
    //     }
    // }

    void dataCollectionPanel::load_map_press()
    {
        ROS_INFO_STREAM("Button map pressed.");
        this->button_1_pub_.publish(this->msg_);

        QString file_name;
        QString launch_filename;

        // if (ui_->output_path_text->text() != "")
        // {
        //     file_name = QFileDialog::getOpenFileName(this, "Choose the Octomp file", (WorkDir+"roadmap_generator/launch").c_str(),"*.ot");
        // }
        // else
        // {
        //     QMessageBox::warning(this,"Error","Directory path is incorrect");
        // }

        // std::string mapFilename(""), mapFilenameParam("");//octomap_server

        // mapFilename = std::string(file_name.toStdString());

        // if (mapFilename != "") 
        // {
        //     if (!this->server_octomap.openFile(mapFilename.c_str()))
        //     {
        //         ROS_ERROR("Could not open file %s", mapFilename.c_str());
        //         // exit(1);
        //     }
        // }    
    
        
        launch_filename = QFileDialog::getOpenFileName(this, "Choose the Launch file", (WorkDir+"roadmap_generator/launch").c_str(), "*.launch");
        
        if (launch_filename.toStdString() != "")
        {
            this->launch_filename_global = launch_filename.toStdString();
            int index = this->launch_filename_global.find("launch");
            this->launch_filename_global = this->launch_filename_global.substr(index + 7);

            // std::string command = "roslaunch" + " " + "roadmap_generator" + " " + launch_filename.toStdString();
            std::string command = "roslaunch roadmap_generator  " + this->launch_filename_global + " &";

            std::system(command.c_str());
        }
        
        // else
        // {
        //     QMessageBox::warning(this,"Error","Directory path is incorrect");
        // }        
        
    }


    void dataCollectionPanel::load_graph_press()
    {
        ROS_INFO_STREAM("Button graph pressed.");
        this->button_2_pub_.publish(this->msg_);
        std::cout<<"Load the graph"<<std::endl;

        if (ui_->output_path_text->text() != "")
        {
            /* code */
            QString graph_filename;
            data_collection_panel::LoadGraph lg_srv;
            graph_filename = QFileDialog::getOpenFileName(this, "Choose the Graph file",ui_->output_path_text->text(), "*.graph");
            lg_srv.request.graph_filename = graph_filename.toStdString();
            load_graph_client.call(lg_srv);
        }
        // else
        // {
        //     QMessageBox::warning(this,"Error","Directory path is incorrect");
        // }

        // this->graph_filename_global = graph_filename.toStdString();
        
    }


    /**
     *  Save all configuration data from this panel to the given
     *  Config object. It is important here that you call save()
     *  on the parent class so the class id and panel name get saved.
     */
    void dataCollectionPanel::save(rviz::Config config) const
    {
        rviz::Panel::save(config);
    }

    /**
     *  Load all configuration data for this panel from the given Config object.
     */
    void dataCollectionPanel::load(const rviz::Config & config)
    {
        rviz::Panel::load(config);
    }
} // namespace rviz_panel


