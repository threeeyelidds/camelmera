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
/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2010, Rice University
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Rice University nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/* Author: Luis G. Torres */

#include "ompl/base/objectives/MaximizeMinClearanceObjective.h"
#include "ompl/tools/config/MagicConstants.h"
#include <limits>

ompl::base::MaximizeMinClearanceObjective::
MaximizeMinClearanceObjective(const SpaceInformationPtr &si) :
    MinimaxObjective(si)
{
    this->setCostThreshold(Cost(std::numeric_limits<double>::infinity()));
}

ompl::base::Cost ompl::base::MaximizeMinClearanceObjective::stateCost(const State* s) const
{
    return Cost(si_->getStateValidityChecker()->clearance(s));
}

bool ompl::base::MaximizeMinClearanceObjective::isCostBetterThan(Cost c1, Cost c2) const
{
    return c1.v > c2.v + magic::BETTER_PATH_COST_MARGIN;
}

ompl::base::Cost ompl::base::MaximizeMinClearanceObjective::identityCost(void) const
{
    return Cost(std::numeric_limits<double>::infinity());
}

ompl::base::Cost ompl::base::MaximizeMinClearanceObjective::infiniteCost(void) const
{
    return Cost(-std::numeric_limits<double>::infinity());
}
