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

/* Author: Ioan Sucan */

#ifndef OMPL_BASE_SPACES_SE2_STATE_SPACE_
#define OMPL_BASE_SPACES_SE2_STATE_SPACE_

#include "ompl/base/StateSpace.h"
#include "ompl/base/spaces/RealVectorStateSpace.h"
#include "ompl/base/spaces/SO2StateSpace.h"

#include <cmath>
#include <boost/math/constants/constants.hpp>
namespace ompl
{
    namespace base
    {

        /** \brief A state space representing SE(2) */
        class SE2StateSpace : public CompoundStateSpace
        {
        public:

            /** \brief A state in SE(2): (x, y, yaw) */
            class StateType : public CompoundStateSpace::StateType
            {
            public:
                StateType(void) : CompoundStateSpace::StateType()
                {
                }

                /** \brief Get the X component of the state */
                double getX(void) const
                {
                    return as<RealVectorStateSpace::StateType>(0)->values[0];
                }

                /** \brief Get the Y component of the state */
                double getY(void) const
                {
                    return as<RealVectorStateSpace::StateType>(0)->values[1];
                }

                /** \brief Get the yaw component of the state. This is
                    the rotation in plane, with respect to the Z
                    axis. */
                double getYaw(void) const
                {
                    return as<SO2StateSpace::StateType>(1)->value;
                }

                /** \brief Set the X component of the state */
                void setX(double x)
                {
                    as<RealVectorStateSpace::StateType>(0)->values[0] = x;
                }

                /** \brief Set the Y component of the state */
                void setY(double y)
                {
                    as<RealVectorStateSpace::StateType>(0)->values[1] = y;
                }

                /** \brief Set the X and Y components of the state */
                void setXY(double x, double y)
                {
                    setX(x);
                    setY(y);
                }

                /** \brief Set the yaw component of the state. This is
                    the rotation in plane, with respect to the Z
                    axis. */
                void setYaw(double yaw)
                {
                    as<SO2StateSpace::StateType>(1)->value = yaw;
                    double v = fmod(as<SO2StateSpace::StateType>(1)->value, 2.0 * boost::math::constants::pi<double>());
                    if (v <= -boost::math::constants::pi<double>())
                        v += 2.0 * boost::math::constants::pi<double>();
                    else
                        if (v > boost::math::constants::pi<double>())
                            v -= 2.0 * boost::math::constants::pi<double>();
                    as<SO2StateSpace::StateType>(1)->value = v;
                }


            };


            SE2StateSpace(void) : CompoundStateSpace()
            {
                setName("SE2" + getName());
                type_ = STATE_SPACE_SE2;
                addSubspace(StateSpacePtr(new RealVectorStateSpace(2)), 1.0);
                addSubspace(StateSpacePtr(new SO2StateSpace()), 0.5);
                lock();
            }

            virtual ~SE2StateSpace(void)
            {
            }

            /** \copydoc RealVectorStateSpace::setBounds() */
            void setBounds(const RealVectorBounds &bounds)
            {
                as<RealVectorStateSpace>(0)->setBounds(bounds);
            }

            /** \copydoc RealVectorStateSpace::getBounds() */
            const RealVectorBounds& getBounds(void) const
            {
                return as<RealVectorStateSpace>(0)->getBounds();
            }

            virtual State* allocState(void) const;
            virtual void freeState(State *state) const;

            virtual void registerProjections(void);

            void AssignWorkspaceTagSubspace(ca::ompl_base::WorkspaceTags::e workspace_tag_translation, ca::ompl_base::WorkspaceTags::e workspace_tag_rotation) {
              getSubspace(0)->AssignWorkspaceTag(workspace_tag_translation);
              getSubspace(1)->AssignWorkspaceTag(workspace_tag_rotation);
            }
        };
    }
}

#endif
