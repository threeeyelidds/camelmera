/** * @author: AirLab / Field Robotics Center
 *
 * @attention Copyright (C) 2016
 * @attention Carnegie Mellon University
 * @attention All rights reserved
 *
 * @attention LIMITED RIGHTS:
 * @attention The US Government is granted Limited Rights to this Data.
 *            Use, duplication, or disclosure is subject to the
 *            restrictions as stated in Agreement AFS12-1642.
 */
/* Copyright 2014 Sanjiban Choudhury
 * state_example.cpp
 *
 *  Created on: Jun 4, 2014
 *      Author: Sanjiban Choudhury
 */

#include "ompl/base/State.h"
#include "ompl/base/ScopedState.h"
#include "ompl/base/spaces/SE2StateSpace.h"
#include "ompl/base/spaces/SE3StateSpace.h"
#include "ompl/base/spaces/DubinsStateSpace.h"
#include "ompl/base/StateSpace.h"

namespace ob = ompl::base;
int main(int argc, char **argv) {
  // Lets define a Dubin's space with radius 2.0 m
  ob::StateSpacePtr space(new ob::DubinsStateSpace(2.0));
  // Lets create a state. Note that scoped state allocates and frees so we dont have
  // to get messy with State*
  ob::ScopedState<ob::DubinsStateSpace> state_obj(space);

  // Fill in some values
  state_obj->setXY(1,2);
  state_obj->setYaw(M_PI/6);

  // Now lets say we want only position
  ob::ScopedState<> pos_component(space->as<ob::DubinsStateSpace>()->getSubspace(0));
  pos_component << state_obj;
  //std::cout << pos_component;

//  // Lets create a compunded space of SE2 and 4dim vector space
  ob::SE2StateSpace *space1 = new ob::SE2StateSpace();
  ob::RealVectorBounds bounds(2);
  bounds.setLow(0);
  bounds.setHigh(1);
  space1->setBounds(bounds);

  ob::RealVectorStateSpace *space2 = new ob::RealVectorStateSpace(2);
  space2->setBounds(bounds);

  ob::StateSpacePtr custom_space = ob::StateSpacePtr(space1) + ob::StateSpacePtr(space2);
  custom_space->setLongestValidSegmentFraction(0.001);
  custom_space->setup();
  ob::ScopedState<> custom_obj(custom_space);

  custom_obj[0] = 0.0;
  custom_obj[2] = 0.3;
  std::cout<<"\n"<<custom_obj<<"\n";

  //std::cout<<"\n"<<ompl::base::ScopedState<>(custom_space->as<ompl::base::CompoundStateSpace>()->getSubspace(0))<<"\n";

  typedef std::map<std::string, ob::StateSpace::SubstateLocation> state_map;
  const state_map substate_loc = custom_space->getSubstateLocationsByName();

  for (state_map::const_iterator it = substate_loc.begin(); it != substate_loc.end(); ++it) {
    std::cout<<"\nName:"<<it->first;
    for (size_t i = 0; i < it->second.chain.size(); i++)
      std::cout<<" "<<it->second.chain[i]<<" ";
  }
  std::cout<<"\n";

  ob::StateSpace::SubstateLocation heading_space = substate_loc.at("SO2Space6");
  ob::State *heading =custom_space->getSubstateAtLocation(custom_obj.get(), heading_space);

  std::cout<<"\nHeading"<<heading->as<ob::SO2StateSpace::StateType>()->value<<"\n";
  heading->as<ob::SO2StateSpace::StateType>()->value = 0.5;
  std::cout<<"\n"<<custom_obj<<"\n";

//
//  ob::ScopedState<> se2comp(obs);
//  std::cout<<"\n"<<custom_obj<<"\n";

}




