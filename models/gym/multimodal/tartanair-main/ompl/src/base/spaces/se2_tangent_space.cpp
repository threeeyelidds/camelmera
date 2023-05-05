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
 * se2_tangent_space.cpp
 *
 *  Created on: Jun 25, 2014
 *      Author: Sanjiban Choudhury
 */

#include "ompl/base/spaces/se2_tangent_space.h"
#include "ompl/tools/config/MagicConstants.h"
#include <cstring>

namespace ob = ompl::base;
namespace caob = ca::ompl_base;

ob::State* caob::SE2TangentSpace::allocState(void) const {
  StateType *state = new StateType();
  allocStateComponents(state);
  return state;
}

void caob::SE2TangentSpace::freeState(ob::State *state) const {
  ob::CompoundStateSpace::freeState(state);
}




