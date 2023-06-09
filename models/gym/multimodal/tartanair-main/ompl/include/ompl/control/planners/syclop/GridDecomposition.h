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

/* Author: Matt Maly */

#ifndef OMPL_CONTROL_PLANNERS_SYCLOP_GRIDDECOMPOSITION_
#define OMPL_CONTROL_PLANNERS_SYCLOP_GRIDDECOMPOSITION_

#include <cstdlib>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>
#include "ompl/base/spaces/RealVectorBounds.h"
#include "ompl/base/State.h"
#include "ompl/control/planners/syclop/Decomposition.h"
#include "ompl/util/RandomNumbers.h"

namespace ompl
{
    namespace control
    {
        /** \brief A GridDecomposition is a Decomposition implemented using a grid. */
        class GridDecomposition : public Decomposition
        {
        public:
            /** \brief Constructor. Creates a GridDecomposition as a hypercube with a given dimension, side length, and bounds.
                The cells of the hypercube are referenced by integer coordinates of the form
                \f$(r_1,\ldots,r_k)\f$, where \f$ 0 \leq r_i < \texttt{len}\f$. */
            GridDecomposition(unsigned int len, unsigned int dim, const base::RealVectorBounds& b);

            virtual ~GridDecomposition()
            {
            }

            virtual double getRegionVolume(unsigned int /*rid*/)
            {
                return cellVolume_;
            }

            virtual void getNeighbors(unsigned int rid, std::vector<unsigned int>& neighbors) const;

            virtual int locateRegion(const base::State* s) const;

            virtual void sampleFromRegion(unsigned int rid, RNG& rng, std::vector<double>& coord) const;

        protected:
            /** \brief Helper method to return the bounds of a given region. */
            virtual const base::RealVectorBounds& getRegionBounds(unsigned int rid) const;

            /** \brief Converts a given region to a coordinate in the grid. */
            void regionToGridCoord(unsigned int rid, std::vector<unsigned int>& coord) const;

            /** \brief Converts the given grid coordinate to its corresponding region ID. */
            unsigned int gridCoordToRegion (const std::vector<unsigned int> &coord) const;

            /** \brief Converts a decomposition space coordinate to the ID of the region that contains iit. */
            unsigned int coordToRegion(const std::vector<double>& coord) const;

            /** \brief Converts a decomposition space coordinate to a grid coordinate. */
            void coordToGridCoord(const std::vector<double>& coord, std::vector<unsigned int>& gridCoord) const;

            /** \brief Computes the neighbors of the given region in a n-dimensional grid */
            void computeGridNeighbors (unsigned int rid, std::vector <unsigned int> &neighbors) const;

            /** Recursive subroutine for grid neighbor computation */
            void computeGridNeighborsSub (const std::vector <unsigned int>&coord, std::vector <unsigned int> &neighbors,
                                          unsigned int dim, std::vector <unsigned int> &candidate) const;

            unsigned int length_;
            double cellVolume_;
            mutable boost::unordered_map<int, boost::shared_ptr<base::RealVectorBounds> > regToBounds_;

        private:
            /** \brief Helper method to return len^dim in call to super-constructor. */
            unsigned int calcNumRegions(unsigned int len, unsigned int dim) const;
        };
    }
}
#endif
