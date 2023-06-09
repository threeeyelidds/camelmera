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
*  Copyright (c) 2011, Rice University
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

/* Author: Mark Moll */

#ifndef OMPL_DATASTRUCTURES_GREEDY_K_CENTERS_
#define OMPL_DATASTRUCTURES_GREEDY_K_CENTERS_

#include "ompl/util/RandomNumbers.h"

namespace ompl
{
    /** \brief An instance of this class can be used to greedily select a given
        number of representatives from a set of data points that are all far
        apart from each other. */
    template<typename _T>
    class GreedyKCenters
    {
    public:
        /** \brief The definition of a distance function */
        typedef boost::function<double(const _T&, const _T&)> DistanceFunction;

        GreedyKCenters(void)
        {
        }

        virtual ~GreedyKCenters(void)
        {
        }

        /** \brief Set the distance function to use */
        void setDistanceFunction(const DistanceFunction &distFun)
        {
            distFun_ = distFun;
        }

        /** \brief Get the distance function used */
        const DistanceFunction& getDistanceFunction(void) const
        {
            return distFun_;
        }

        /** \brief Greedy algorithm for selecting k centers
            \param data a vector of data points
            \param k the desired number of centers
            \param centers a vector of length k containing the indices into
                data of the k centers
            \param dists a 2-dimensional array such that dists[i][j] is the distance
                between data[i] and data[center[j]]
        */
        void kcenters(const std::vector<_T>& data, unsigned int k,
            std::vector<unsigned int>& centers, std::vector<std::vector<double> >& dists)
        {
            // array containing the minimum distance between each data point
            // and the centers computed so far
            std::vector<double> minDist(data.size(), std::numeric_limits<double>::infinity());

            centers.clear();
            centers.reserve(k);
            dists.resize(data.size(), std::vector<double>(k));
            // first center is picked randomly
            centers.push_back(rng_.uniformInt(0, data.size() - 1));
            for (unsigned i=1; i<k; ++i)
            {
                unsigned ind;
                const _T& center = data[centers[i - 1]];
                double maxDist = -std::numeric_limits<double>::infinity();
                for (unsigned j=0; j<data.size(); ++j)
                {
                    if ((dists[j][i-1] = distFun_(data[j], center)) < minDist[j])
                        minDist[j] = dists[j][i - 1];
                    // the j-th center is the one furthest away from center 0,..,j-1
                    if (minDist[j] > maxDist)
                    {
                        ind = j;
                        maxDist = minDist[j];
                    }
                }
                // no more centers available
                if (maxDist < std::numeric_limits<double>::epsilon()) break;
                centers.push_back(ind);
            }

            const _T& center = data[centers.back()];
            unsigned i = centers.size() - 1;
            for (unsigned j = 0; j < data.size(); ++j)
                dists[j][i] = distFun_(data[j], center);
        }

    protected:
        /** \brief The used distance function */
        DistanceFunction distFun_;

        /** Random number generator used to select first center */
        RNG              rng_;
    };
}

#endif
