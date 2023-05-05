
#include "roadmap_generator/roadmap.h"

int main(int argc, char** argv){
  ros::init(argc, argv, "roadmap_node");
  roadmap::Roadmap ha;

//  ros::MultiThreadedSpinner s;
//  s.spin();
  ros::spin();

  return(0);
}

