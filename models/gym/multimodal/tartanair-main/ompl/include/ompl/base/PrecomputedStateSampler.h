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
*  Copyright (c) 2012, Willow Garage
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
*   * Neither the name of the Willow Garage nor the names of its
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

#include "ompl/base/StateSampler.h"
#include <vector>

namespace ompl
{
    namespace base
    {

        /** \brief State space sampler for discrete states */
        class PrecomputedStateSampler : public StateSampler
        {
        public:

            /** \brief Constructor. Takes the state space to be sampled (\e space) and the set of states to draw samples from (\e states) */
            PrecomputedStateSampler(const StateSpace *space, const std::vector<const State*> &states);

            /** \brief Constructor. Takes the state space to be sampled (\e space), the set of states to draw samples from (\e states) and a range to sample from: [\e minIndex, \e maxIndex]*/
            PrecomputedStateSampler(const StateSpace *space, const std::vector<const State*> &states, std::size_t minIndex, std::size_t maxIndex);

            virtual void sampleUniform(State *state);
            virtual void sampleUniformNear(State *state, const State *near, const double distance);
            virtual void sampleGaussian(State *state, const State *mean, const double stdDev);

        protected:

            /** \brief The states to sample from */
            const std::vector<const State*> &states_;

            /** \brief The minimum index to start sampling at */
            std::size_t                      minStateIndex_;

            /** \brief The maximum index to stop sampling at */
            std::size_t                      maxStateIndex_;
        };
    }
}
