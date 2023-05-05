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
 * MotionValidator.cpp
 *
 *  Created on: Nov 30, 2014
 *      Author: Sanjiban Choudhury
 */

#include "ompl/base/MotionValidator.h"
#include "ompl/base/SpaceInformation.h"

namespace ompl
{
namespace base
{

unsigned int MotionValidator::getMotionStates(const State *s1, const State *s2, std::vector<State*> &states, unsigned int count, bool endpoints, bool alloc) const {
  // add 1 to the number of states we want to add between s1 & s2. This gives us the number of segments to split the motion into
  count++;

  if (count < 2)
  {
    unsigned int added = 0;

    // if they want endpoints, then at most endpoints are included
    if (endpoints)
    {
      if (alloc)
      {
        states.resize(2);
        states[0] = si_->allocState();
        states[1] = si_->allocState();
      }
      if (states.size() > 0)
      {
        si_->copyState(states[0], s1);
        added++;
      }

      if (states.size() > 1)
      {
        si_->copyState(states[1], s2);
        added++;
      }
    }
    else
      if (alloc)
        states.resize(0);
    return added;
  }

  if (alloc)
  {
    states.resize(count + (endpoints ? 1 : -1));
    if (endpoints)
      states[0] = si_->allocState();
  }

  unsigned int added = 0;

  if (endpoints && states.size() > 0)
  {
    si_->copyState(states[0], s1);
    added++;
  }

  /* find the states in between */
  for (unsigned int j = 1 ; j < count && added < states.size() ; ++j)
  {
    if (alloc)
      states[added] = si_->allocState();
    si_->getStateSpace()->interpolate(s1, s2, (double)j / (double)count, states[added]);
    added++;
  }

  if (added < states.size() && endpoints)
  {
    if (alloc)
      states[added] = si_->allocState();
    si_->copyState(states[added], s2);
    added++;
  }

  return added;
}

}
}
