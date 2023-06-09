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

#ifndef OMPL_UTIL_RANDOM_NUMBERS_
#define OMPL_UTIL_RANDOM_NUMBERS_

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <cassert>

#include <ompl/util/ClassForward.h>

//The basic Eigen include
#include <Eigen/Core>

namespace ompl
{
    /// @cond IGNORE
    //A simple forward declaration of the prolate hyperspheroid class
    OMPL_CLASS_FORWARD(ProlateHyperspheroid);
    /// @endcond

    /** \brief Random number generation. An instance of this class
        cannot be used by multiple threads at once (member functions
        are not const). However, the constructor is thread safe and
        different instances can be used safely in any number of
        threads. It is also guaranteed that all created instances will
        have a different random seed. */
    class RNG
    {
    public:

        /** \brief Constructor. Always sets a different random seed */
        RNG();

        /** \brief Constructor. Set to the specified instance seed. */
        RNG(boost::uint32_t localSeed);

        /** \brief Generate a random real between 0 and 1 */
        double uniform01()
        {
            return uni_();
        }

        /** \brief Generate a random real within given bounds: [\e lower_bound, \e upper_bound) */
        double uniformReal(double lower_bound, double upper_bound)
        {
            assert(lower_bound <= upper_bound);
            return (upper_bound - lower_bound) * uni_() + lower_bound;
        }

        /** \brief Generate a random integer within given bounds: [\e lower_bound, \e upper_bound] */
        int uniformInt(int lower_bound, int upper_bound)
        {
            int r = (int)floor(uniformReal((double)lower_bound, (double)(upper_bound) + 1.0));
            return (r > upper_bound) ? upper_bound : r;
        }

        /** \brief Generate a random boolean */
        bool uniformBool()
        {
            return uni_() <= 0.5;
        }

        /** \brief Generate a random real using a normal distribution with mean 0 and variance 1 */
        double gaussian01()
        {
            return normal_();
        }

        /** \brief Generate a random real using a normal distribution with given mean and variance */
        double gaussian(double mean, double stddev)
        {
            return normal_() * stddev + mean;
        }

        /** \brief Generate a random real using a half-normal distribution. The value is within specified bounds [\e
            r_min, \e r_max], but with a bias towards \e r_max. The function is implemended using a Gaussian distribution with
            mean at \e r_max - \e r_min. The distribution is 'folded' around \e r_max axis towards \e r_min.
            The variance of the distribution is (\e r_max - \e r_min) / \e focus. The higher the focus,
            the more probable it is that generated numbers are close to \e r_max. */
        double halfNormalReal(double r_min, double r_max, double focus = 3.0);

        /** \brief Generate a random integer using a half-normal
            distribution. The value is within specified bounds ([\e r_min, \e r_max]), but
            with a bias towards \e r_max. The function is implemented on top of halfNormalReal() */
        int    halfNormalInt(int r_min, int r_max, double focus = 3.0);

        /** \brief Uniform random unit quaternion sampling. The computed value has the order (x,y,z,w) */
        void   quaternion(double value[4]);

        /** \brief Uniform random sampling of Euler roll-pitch-yaw angles, each in the range (-pi, pi]. The computed value has the order (roll, pitch, yaw) */
        void   eulerRPY(double value[3]);

        /** \brief Set the seed used to generate the seeds of each RNG instance. Use this
            function to ensure the same sequence of random numbers is generated across multiple instances of RNG. */
        static void setSeed(boost::uint32_t seed);

        /** \brief Get the seed used to generate the seeds of each RNG instance.
            Passing the returned value to setSeed() at a subsequent execution of the code will ensure deterministic
            (repeatable) behaviour across multiple instances of RNG. Useful for debugging. */
        static boost::uint32_t getSeed();

        /** \brief Set the seed used for the instance of a RNG. Use this function to ensure that an instance of
            an RNG generates the same deterministic sequence of numbers. This function resets the member generators*/
        void setLocalSeed(boost::uint32_t localSeed);

        /** \brief Get the seed used for the instance of a RNG. Passing the returned value to the setInstanceSeed()
            of another RNG will assure that the two objects generate the same sequence of numbers.
            Useful for comparing different settings of a planner while maintaining the same stochastic behaviour,
            assuming that every "random" decision made by the planner is made from the same RNG. */
        boost::uint32_t getLocalSeed() const
        {
            return localSeed_;
        }

        /** \brief Uniform random sampling of a unit-length vector. I.e., the surface of an n-ball */
        void uniformNormalVector(unsigned int n, double value[]);

        /** \brief Uniform random sampling of the content of an n-ball, with a radius appropriately distributed between [0,r) */
        void uniformInBall(double r, unsigned int n, double value[]);

        /** \brief Uniform random sampling of the surface of a prolate hyperspheroid, a special symmetric type of
        n-dimensional ellipse.
        @par J D. Gammell, S. S. Srinivasa, T. D. Barfoot, "Informed RRT*: Optimal Sampling-based
        Path Planning Focused via Direct Sampling of an Admissible Ellipsoidal Heuristic."
        IROS 2014. <a href="http://arxiv.org/abs/1404.2334">arXiv:1404.2334 [cs.RO]</a>.
        <a href="http://www.youtube.com/watch?v=d7dX5MvDYTc">Illustration video</a>.
        <a href="http://www.youtube.com/watch?v=nsl-5MZfwu4">Short description video</a>. */
        void uniformProlateHyperspheroidSurface(ProlateHyperspheroidPtr phsPtr, unsigned int n, double value[]);

        /** \brief Uniform random sampling of a prolate hyperspheroid, a special symmetric type of
        n-dimensional ellipse.
        @par J D. Gammell, S. S. Srinivasa, T. D. Barfoot, "Informed RRT*: Optimal Sampling-based
        Path Planning Focused via Direct Sampling of an Admissible Ellipsoidal Heuristic."
        IROS 2014. <a href="http://arxiv.org/abs/1404.2334">arXiv:1404.2334 [cs.RO]</a>.
        <a href="http://www.youtube.com/watch?v=d7dX5MvDYTc">Illustration video</a>.
        <a href="http://www.youtube.com/watch?v=nsl-5MZfwu4">Short description video</a>. */
        void uniformProlateHyperspheroid(ProlateHyperspheroidPtr phsPtr, unsigned int n, double value[]);

    private:

        /** \brief The seed used for the instance of a RNG */
        boost::uint32_t                                                          localSeed_;
        boost::mt19937                                                           generator_;
        boost::uniform_real<>                                                    uniDist_;
        boost::normal_distribution<>                                             normalDist_;
        // Variate generators must be reset when the seed changes
        boost::variate_generator<boost::mt19937&, boost::uniform_real<> >        uni_;
        boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > normal_;

    };


    /** \brief A class describing a prolate hyperspheroid, a special symmetric type of n-dimensional ellipse,
    for use in direct informed sampling.
    @par J D. Gammell, S. S. Srinivasa, T. D. Barfoot, "Informed RRT*: Optimal Sampling-based
    Path Planning Focused via Direct Sampling of an Admissible Ellipsoidal Heuristic."
    IROS 2014. <a href="http://arxiv.org/abs/1404.2334">arXiv:1404.2334 [cs.RO]</a>.
    <a href="http://www.youtube.com/watch?v=d7dX5MvDYTc">Illustration video</a>.
    <a href="http://www.youtube.com/watch?v=nsl-5MZfwu4">Short description video</a>. */
    class ProlateHyperspheroid
    {
    public:
        /** \brief The description of an n-dimensional prolate hyperspheroid */
        ProlateHyperspheroid(unsigned int n, const double focus1[], const double focus2[]);

        /** \brief Set the transverse diameter of the PHS */
        void setTransverseDiameter(double transverseDiameter);

        /** \brief Transform a point from a sphere to PHS */
        void transform(unsigned int n, const double sphere[], double phs[]);

        /** \brief Check if the given point lies within the PHS */
        bool isInPhs(unsigned int n, const double point[]);

        /** \brief The dimension of the PHS */
        inline unsigned int getPhsDimension(void) { return dim_; };

        /** \brief The measure of the PHS */
        double getPhsMeasure(void);

        /** \brief The measure of the PHS for a given transverse diameter */
        double getPhsMeasure(double tranDiam);

        /** \brief The minimum transverse diameter of the PHS, i.e., the distance between the foci */
        inline double getMinTransverseDiameter(void) { return minTransverseDiameter_; };

        /** \brief Calculate length of a line that originates from one focus, passes through the given point, and terminates at the other focus, i.e., the transverse diameter of the ellipse on which the given sample lies*/
        double getPathLength(unsigned int n, const double point[]);

        /** \brief The "volume" of a unit n-ball */
        static double unitNBallMeasure(unsigned int N);

        /** \brief The "volume" of a general n-PHS given the distance between the foci (minTransverseDiameter) and the actual transverse diameter (transverseDiameter) */
        static double calcPhsMeasure(unsigned int N, double minTransverseDiameter, double transverseDiameter);

    protected:

    private:
        /** \brief The dimension of the prolate hyperspheroid.*/
        unsigned int dim_;
        /** \brief The minimum possible transverse diameter of the PHS. Defined as the distance between the two foci*/
        double minTransverseDiameter_;
        /** \brief The transverse diameter of the PHS. */
        double transverseDiameter_;
        /** \brief The measure of the PHS. */
        double phsMeasure_;
        /** \brief The first focus of the PHS (i.e., the start state of the planning problem)*/
        Eigen::VectorXd xFocus1_;
        /** \brief The second focus of the PHS (i.e., the goal state of the planning problem)*/
        Eigen::VectorXd xFocus2_;
        /** \brief The centre of the PHS. Defined as the average of the foci.*/
        Eigen::VectorXd xCentre_;
        /** \brief The rotation from PHS-frame to world frame. Is only calculated on construction. */
        Eigen::MatrixXd rotationWorldFromEllipse_;
        /** \brief The transformation from PHS-frame to world frame. Is calculated every time the transverse diameter changes. */
        Eigen::MatrixXd transformationWorldFromEllipse_;
        /** \brief Whether the transformation is up to date */
        bool isTransformUpToDate_;

        //Functions
        /** \brief Calculate the rotation from the PHS frame to the world frame via singular-value decomposition using the transverse symmetry of the PHS. */
        void updateRotation(void);

        /** \brief Calculate the hyperspheroid to PHS transformation matrix */
        void updateTransformation(void);
    };
}

#endif
