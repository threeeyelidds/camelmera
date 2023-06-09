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
*  Copyright (c) 2008, Willow Garage, Inc.
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

#ifndef OMPL_DATASTRUCTURES_NEAREST_NEIGHBORS_
#define OMPL_DATASTRUCTURES_NEAREST_NEIGHBORS_

#include <vector>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <ompl/base/StateSpace.h>

namespace ompl
{

    /** \brief Abstract representation of a container that can perform nearest neighbors queries */
    template<typename _T>
    class NearestNeighbors
    {
    public:

        /** \brief The definition of a distance function */
        typedef boost::function<double(const _T&, const _T&)> DistanceFunction;

        NearestNeighbors(void)
        {
        }

        virtual ~NearestNeighbors(void)
        {
        }

        /** \brief Set the distance function to use */
        virtual void setDistanceFunction(const DistanceFunction &distFun)
        {
            distFun_ = distFun;
        }

        /** \brief Get the distance function used */
        const DistanceFunction& getDistanceFunction(void) const
        {
            return distFun_;
        }

        /** \brief Clear the datastructure */
        virtual void clear(void) = 0;

        /** \brief Add an element to the datastructure */
        virtual void add(const _T &data) = 0;

        /** \brief Add a vector of points */
        virtual void add(const std::vector<_T> &data)
        {
            for (typename std::vector<_T>::const_iterator elt = data.begin() ; elt != data.end() ; ++elt)
                add(*elt);
        }

        /** \brief Remove an element from the datastructure */
        virtual bool remove(const _T &data) = 0;

        /** \brief Get the nearest neighbor of a point */
        virtual _T nearest(const _T &data) const = 0;

        /** \brief Get the k-nearest neighbors of a point
         *
         * All the nearest neighbor structures currently return the neighbors in
         * sorted order, but this is not required.
         */
        virtual void nearestK(const _T &data, std::size_t k, std::vector<_T> &nbh) const = 0;

        /** \brief Get the nearest neighbors of a point, within a specified radius
         *
         * All the nearest neighbor structures currently return the neighbors in
         * sorted order, but this is not required.
         */
        virtual void nearestR(const _T &data, double radius, std::vector<_T> &nbh) const = 0;

        /** \brief Get the number of elements in the datastructure */
        virtual std::size_t size(void) const = 0;

        /** \brief Get all the elements in the datastructure */
        virtual void list(std::vector<_T> &data) const = 0;

    protected:

        /** \brief The used distance function */
        DistanceFunction distFun_;

    };
}

#endif
