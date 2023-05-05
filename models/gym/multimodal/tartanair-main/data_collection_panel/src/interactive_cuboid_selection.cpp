#include <visualization_msgs/Marker.h>


#include <interactive_markers/interactive_marker_server.h>

#include <tf/transform_broadcaster.h>
#include <tf/tf.h>

#include <math.h>

// For ros service used by data collection rviz panel
#include "data_collection_panel/PanelSimpleService.h"
#include "data_collection_panel/ConfirmAndSendBounds.h"
#include "data_collection_panel/UpdateCuboidBounds.h"

using namespace visualization_msgs;

Marker makeCuboidMarker(bool remove_marker);
Marker makeBox( InteractiveMarker &, float);
InteractiveMarkerControl& makeBoxControl( InteractiveMarker &, float);
void processFeedback( const visualization_msgs::InteractiveMarkerFeedbackConstPtr &);
void make6DofMarker(std::string, bool, unsigned int , const tf::Vector3& , bool);

boost::shared_ptr<interactive_markers::InteractiveMarkerServer> server;

tf::Vector3 position_max,position_min,cuboid_position_vector,cuboid_scale;

bool flag_for_while_loop=false;
bool flag_update_text = true;
bool DisplayCuboid(data_collection_panel::PanelSimpleService::Request&,
                  data_collection_panel::PanelSimpleService::Response&);
bool HideCuboid(data_collection_panel::PanelSimpleService::Request&,
                  data_collection_panel::PanelSimpleService::Response&);
bool ConfirmAndSendBounds(data_collection_panel::ConfirmAndSendBounds::Request&,
                  data_collection_panel::ConfirmAndSendBounds::Response&);




int main(int argc, char  **argv)
{
	/* code */
  ros::init(argc, argv, "interactive_cuboid_selection");
	ros::NodeHandle nh_;
	ros::Publisher Cuboid_Marker_pub;
  ros::ServiceServer display_cuboid_srv;
  ros::ServiceServer hide_cuboid_srv;
  ros::ServiceServer confirm_and_send_bounds_srv;

  display_cuboid_srv = nh_.advertiseService("display_cuboid",DisplayCuboid);
  hide_cuboid_srv = nh_.advertiseService("hide_cuboid",HideCuboid);
  confirm_and_send_bounds_srv = nh_.advertiseService("confirm_and_send_bounds",ConfirmAndSendBounds);

  ros::ServiceClient update_cuboid_text_client;
  update_cuboid_text_client = nh_.serviceClient<data_collection_panel::UpdateCuboidBounds>("update_cuboid_text");

	Cuboid_Marker_pub = nh_.advertise<visualization_msgs::Marker>("/sampling_cuboid_marker", 2);


	server.reset( new interactive_markers::InteractiveMarkerServer("two_marker","",false) );

  position_max = tf::Vector3(5, 5, 2.5);
  position_min = tf::Vector3(-5, -5, -2.5);
  cuboid_position_vector = tf::Vector3(0, 0, 0);  
  cuboid_scale = tf::Vector3(10, 10, 5);

  // visualization_msgs::Marker empty_marker;
  // empty_marker.header.frame_id = "map";
	
  while(ros::ok())
    {

        ros::Duration(0.1).sleep();

        if (flag_for_while_loop)
        {
          make6DofMarker("simple_6dof_max", false, visualization_msgs::InteractiveMarkerControl::MOVE_3D, position_max, true);
          make6DofMarker("simple_6dof_min", false, visualization_msgs::InteractiveMarkerControl::MOVE_3D, position_min, true);

          Marker marker = makeCuboidMarker(false);
          Cuboid_Marker_pub.publish( marker );

          server->applyChanges();
          if (flag_update_text == true)
          {
              // // update the text field for the cuboid
              data_collection_panel::UpdateCuboidBounds cb_srv;
              float xmin = position_max[0]>position_min[0]?position_min[0]:position_max[0]; 
              float xmax = position_max[0]<position_min[0]?position_min[0]:position_max[0]; 
              float ymin = position_max[1]>position_min[1]?position_min[1]:position_max[1]; 
              float ymax = position_max[1]<position_min[1]?position_min[1]:position_max[1]; 
              float zmin = position_max[2]>position_min[2]?position_min[2]:position_max[2]; 
              float zmax = position_max[2]<position_min[2]?position_min[2]:position_max[2]; 
              cb_srv.request.node_range_xmin = xmin;
              cb_srv.request.node_range_xmax = xmax;
              cb_srv.request.node_range_ymin = ymin;
              cb_srv.request.node_range_ymax = ymax;
              cb_srv.request.node_range_zmin = zmin;
              cb_srv.request.node_range_zmax = zmax;
              update_cuboid_text_client.call(cb_srv);
              // if (update_cuboid_text_client.call(cb_srv))
              //   ROS_INFO_STREAM( "successful"  );
              // else
              //   ROS_INFO_STREAM( "failed"  );
            flag_update_text = false;
          }

          // ROS_INFO_STREAM( "Range " << xmin << ", " << xmax << ", " << ymin << ", " << ymax << ", " << zmin << ", " << zmax  );

        }
        else
        {
          Marker marker = makeCuboidMarker(true);
          // marker.color.a = 0.0; // to hide the cuboid
          Cuboid_Marker_pub.publish( marker );

          // The corner box and arrows.
          server->clear();
          server->applyChanges();
        }
        
        ros::spinOnce();
    }
     
    server.reset();

        
	return 0;
}

Marker makeCuboidMarker(bool remove_marker)
{
  visualization_msgs::Marker marker;

  cuboid_position_vector = tf::Vector3(0, 0, 0);
  cuboid_position_vector = position_min;
  cuboid_position_vector.operator+=(position_max);
  cuboid_position_vector.operator*=(0.5);
  
  cuboid_scale = tf::Vector3(10, 10, 5);
  cuboid_scale = position_min;
  cuboid_scale.operator-=(position_max);
  cuboid_scale.absolute();

  marker.header.frame_id = "map";
  marker.header.stamp = ros::Time();
  marker.ns = "sampling_cuboid_marker";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::CUBE;
  if (remove_marker)
    marker.action = visualization_msgs::Marker::DELETE;
  else
    marker.action = visualization_msgs::Marker::ADD;

  tf::pointTFToMsg(cuboid_position_vector,marker.pose.position);

  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  tf::vector3TFToMsg(cuboid_scale,marker.scale);

  marker.color.a = 0.2; // Don't forget to set the alpha!
  marker.color.r = 0;
  marker.color.g = 0;
  marker.color.b = 255;
  return marker;
}


Marker makeBox( InteractiveMarker &msg, float a)
{
  Marker marker;

  marker.type = Marker::CUBE;
  marker.scale.x = msg.scale * 0.45;
  marker.scale.y = msg.scale * 0.45;
  marker.scale.z = msg.scale * 0.45;
  marker.color.r = 0.5;
  marker.color.g = 0.5;
  marker.color.b = 0.5;
  marker.color.a = a;

  return marker;
}


InteractiveMarkerControl& makeBoxControl( InteractiveMarker &msg ,float a)
{
  InteractiveMarkerControl control;
  control.always_visible = true;
  control.markers.push_back( makeBox(msg,a) );
  msg.controls.push_back( control );

  return msg.controls.back();
}


void processFeedback(const visualization_msgs::InteractiveMarkerFeedbackConstPtr &feedback )
{
  std::ostringstream s;
  // s << "Feedback from marker '" << feedback->marker_name << "' "
  //     << " / control '" << feedback->control_name << "'";
  s << feedback->marker_name;

  std::ostringstream mouse_point_ss;
  if( feedback->mouse_point_valid )
  {
    mouse_point_ss << " at " << feedback->mouse_point.x
                   << ", " << feedback->mouse_point.y
                   << ", " << feedback->mouse_point.z
                   << " in frame " << feedback->header.frame_id;
  }

  switch ( feedback->event_type )
  {
    case visualization_msgs::InteractiveMarkerFeedback::BUTTON_CLICK:
      ROS_INFO_STREAM( s.str() << ": button click" << mouse_point_ss.str() << "." );
      break;

    case visualization_msgs::InteractiveMarkerFeedback::MENU_SELECT:
      ROS_INFO_STREAM( s.str() << ": menu item " << feedback->menu_entry_id << " clicked" << mouse_point_ss.str() << "." );
      break;

    case visualization_msgs::InteractiveMarkerFeedback::POSE_UPDATE:
      flag_update_text = true;
      if(feedback->marker_name == "simple_6dof_max")
      {
        position_max = tf::Vector3(feedback->pose.position.x, 
                                   feedback->pose.position.y, 
                                   feedback->pose.position.z);
      }

      else if(feedback->marker_name == "simple_6dof_min")
      {
        position_min = tf::Vector3(feedback->pose.position.x, 
                                   feedback->pose.position.y, 
                                   feedback->pose.position.z);
      }

      ROS_INFO_STREAM_THROTTLE(1.0, s.str() << ": pose changed, "
          << "\tposition = "
          << feedback->pose.position.x
          << ", " << feedback->pose.position.y
          << ", " << feedback->pose.position.z);
          // << "\torientation = "
          // << feedback->pose.orientation.w
          // << ", " << feedback->pose.orientation.x
          // << ", " << feedback->pose.orientation.y
          // << ", " << feedback->pose.orientation.z);
          // << "\nframe: " << feedback->header.frame_id
          // << " time: " << feedback->header.stamp.sec << "sec, "
          // << feedback->header.stamp.nsec << " nsec" );
      break;

    case visualization_msgs::InteractiveMarkerFeedback::MOUSE_DOWN:
      ROS_INFO_STREAM(s.str() << ": mouse down" << mouse_point_ss.str() << "." );
      break;

    case visualization_msgs::InteractiveMarkerFeedback::MOUSE_UP:
      ROS_INFO_STREAM( s.str() << ": mouse up" << mouse_point_ss.str() << "." );
      break;
  }

  // this->server.applyChanges();
}


void make6DofMarker(std::string name, bool fixed, unsigned int interaction_mode, const tf::Vector3& position, bool show_6dof)
{

  InteractiveMarker int_marker;
  int_marker.header.frame_id = "map";
  tf::pointTFToMsg(position, int_marker.pose.position);
  int_marker.scale = 3;

  int_marker.name = name;
  // int_marker.description = "Simple 6-DOF Control";

  // insert a box
  makeBoxControl(int_marker, 1.0);
  int_marker.controls[0].interaction_mode = interaction_mode;

  // if(a>0.0)
  //   int_marker.controls[0].markers[0].action = visualization_msgs::Marker::ADD;
  // else
  //   int_marker.controls[0].markers[0].action = visualization_msgs::Marker::DELETE;

  InteractiveMarkerControl control;

  if ( fixed )
  {
    // int_marker.name += "_fixed";
    // int_marker.description += "\n(fixed orientation)";
    control.orientation_mode = InteractiveMarkerControl::FIXED;
  }

  if (interaction_mode != visualization_msgs::InteractiveMarkerControl::NONE)
  {
      std::string mode_text;
      if( interaction_mode == visualization_msgs::InteractiveMarkerControl::MOVE_3D )         mode_text = "MOVE_3D";
      if( interaction_mode == visualization_msgs::InteractiveMarkerControl::ROTATE_3D )       mode_text = "ROTATE_3D";
      if( interaction_mode == visualization_msgs::InteractiveMarkerControl::MOVE_ROTATE_3D )  mode_text = "MOVE_ROTATE_3D";
      // int_marker.name += "_" + mode_text;
      // int_marker.description = std::string("3D Control") + (show_6dof ? " + 6-DOF controls" : "") + "\n" + mode_text;
  }

  if(show_6dof)
  {
    control.orientation.w = 1;
    control.orientation.x = 1;
    control.orientation.y = 0;
    control.orientation.z = 0;
    // control.name = "rotate_x";
    // control.interaction_mode = InteractiveMarkerControl::ROTATE_AXIS;
    // int_marker.controls.push_back(control);
    control.name = "move_x";
    control.interaction_mode = InteractiveMarkerControl::MOVE_AXIS;
    int_marker.controls.push_back(control);

    control.orientation.w = 1;
    control.orientation.x = 0;
    control.orientation.y = 1;
    control.orientation.z = 0;
    // control.name = "rotate_z";
    // control.interaction_mode = InteractiveMarkerControl::ROTATE_AXIS;
    // int_marker.controls.push_back(control);
    control.name = "move_z";
    control.interaction_mode = InteractiveMarkerControl::MOVE_AXIS;
    int_marker.controls.push_back(control);

    control.orientation.w = 1;
    control.orientation.x = 0;
    control.orientation.y = 0;
    control.orientation.z = 1;
    // control.name = "rotate_y";
    // control.interaction_mode = InteractiveMarkerControl::ROTATE_AXIS;
    // int_marker.controls.push_back(control);
    control.name = "move_y";
    control.interaction_mode = InteractiveMarkerControl::MOVE_AXIS;
    int_marker.controls.push_back(control);
  }

  ::server->insert(int_marker);
  ::server->setCallback(int_marker.name, &processFeedback);
  // if (interaction_mode != visualization_msgs::InteractiveMarkerControl::NONE)
  //   menu_handler.apply( *server, int_marker.name );
}

bool DisplayCuboid(data_collection_panel::PanelSimpleService::Request& request,
                  data_collection_panel::PanelSimpleService::Response& response)
{
  ROS_INFO("Service call successful");
  flag_for_while_loop = true;
  return true;
}

bool HideCuboid(data_collection_panel::PanelSimpleService::Request& request,
                  data_collection_panel::PanelSimpleService::Response& response)
{
  ROS_INFO("Service call successful");
  flag_for_while_loop = false;
  return true;
}

bool ConfirmAndSendBounds(data_collection_panel::ConfirmAndSendBounds::Request& request,
                  data_collection_panel::ConfirmAndSendBounds::Response& response)
{
  response.max_x = position_max.getX();
  response.min_x = position_min.getX();
  response.max_y = position_max.getY();
  response.min_y = position_min.getY();
  response.max_z = position_max.getZ();
  response.min_z = position_min.getZ();
  ROS_INFO("Cuboid: %f, %f,  %f, %f,  %f, %f", response.max_x, response.min_x, response.max_y, response.min_y, 
                                               response.max_z, response.min_z );

  return true;
}