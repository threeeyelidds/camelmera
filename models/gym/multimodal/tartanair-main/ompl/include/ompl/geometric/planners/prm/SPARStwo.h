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
*  Copyright (c) 2013, Rutgers the State University of New Jersey, New Brunswick
*  All Rights Reserved.
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
*   * Neither the name of Rutgers University nor the names of its
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

/* Author: Andrew Dobson */

#ifndef OMPL_GEOMETRIC_PLANNERS_SPARS_TWO_
#define OMPL_GEOMETRIC_PLANNERS_SPARS_TWO_

#include "ompl/geometric/planners/PlannerIncludes.h"
#include "ompl/datastructures/NearestNeighbors.h"
#include "ompl/geometric/PathSimplifier.h"
#include "ompl/util/Time.h"

#include <boost/range/adaptor/map.hpp>
#include <boost/unordered_map.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/pending/disjoint_sets.hpp>
#include <boost/function.hpp>
#include <boost/thread.hpp>
#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <map>

namespace ompl
{

    namespace geometric
    {

        /**
           @anchor gSPARStwo
           @par Short description
           SPARStwo is a variant of the SPARS algorithm which removes the
           dependency on having the dense graph, D.  It works through similar
           mechanics, but uses a different approach to identifying interfaces
           and computing shortest paths through said interfaces.
           @par External documentation
           A. Dobson, K. Bekris,
           Improving Sparse Roadmap Spanners,
           <em>IEEE International Conference on Robotics and Automation (ICRA)</em> May 2013.
           <a href="http://www.cs.rutgers.edu/~kb572/pubs/spars2.pdf">[PDF]</a>
        */

        /** \brief <b> SPArse Roadmap Spanner Version 2.0 </b> */
        class SPARStwo : public base::Planner
        {
        public:

            /** \brief Enumeration which specifies the reason a guard is added to the spanner. */
            enum GuardType
            {
                START,
                GOAL,
                COVERAGE,
                CONNECTIVITY,
                INTERFACE,
                QUALITY,
            };

            /** \brief The type used internally for representing vertex IDs */
            typedef unsigned long int VertexIndexType;

            /** \brief Pair of vertices which support an interface. */
            typedef std::pair< VertexIndexType, VertexIndexType > VertexPair;

            /** \brief Interface information storage class, which does bookkeeping for criterion four. */
            struct InterfaceData
            {
                /** \brief States which lie inside the visibility region of a vertex and support an interface. */
                base::State* pointA_;
                base::State* pointB_;

                /** \brief States which lie just outside the visibility region of a vertex and support an interface. */
                base::State* sigmaA_;
                base::State* sigmaB_;

                /** \brief Last known distance between the two interfaces supported by points_ and sigmas. */
                double       d_;

                /** \brief Constructor */
                InterfaceData(void) :
                    pointA_(NULL),
                    pointB_(NULL),
                    sigmaA_(NULL),
                    sigmaB_(NULL),
                    d_(std::numeric_limits<double>::infinity())
                {
                }

                /** \brief Clears the given interface data. */
                void clear(const base::SpaceInformationPtr& si)
                {
                    if (pointA_)
                    {
                        si->freeState(pointA_);
                        pointA_ = NULL;
                    }
                    if (pointB_)
                    {
                        si->freeState(pointB_);
                        pointB_ = NULL;
                    }
                    if (sigmaA_)
                    {
                        si->freeState(sigmaA_);
                        sigmaA_ = NULL;
                    }
                    if (sigmaB_)
                    {
                        si->freeState(sigmaB_);
                        sigmaB_ = NULL;
                    }
                    d_ = std::numeric_limits<double>::infinity();
                }

                /** \brief Sets information for the first interface (i.e. interface with smaller index vertex). */
                void setFirst(const base::State *p, const base::State *s, const base::SpaceInformationPtr& si)
                {
                    if (pointA_)
                        si->copyState(pointA_, p);
                    else
                        pointA_ = si->cloneState(p);
                    if (sigmaA_)
                        si->copyState(sigmaA_, s);
                    else
                        sigmaA_ = si->cloneState(s);
                    if (pointB_)
                        d_ = si->distance(pointA_, pointB_);
                }

                /** \brief Sets information for the second interface (i.e. interface with larger index vertex). */
                void setSecond(const base::State *p, const base::State *s, const base::SpaceInformationPtr& si)
                {
                    if (pointB_)
                        si->copyState(pointB_, p);
                    else
                        pointB_ = si->cloneState(p);
                    if (sigmaB_)
                        si->copyState(sigmaB_, s);
                    else
                        sigmaB_ = si->cloneState(s);
                    if (pointA_)
                        d_ = si->distance(pointA_, pointB_);
                }
            };

            /** \brief the hash which maps pairs of neighbor points to pairs of states */
            typedef boost::unordered_map< VertexPair, InterfaceData, boost::hash< VertexPair > > InterfaceHash;

            // The InterfaceHash structure is wrapped inside of this struct due to a compilation error on
            // GCC 4.6 with Boost 1.48.  An implicit assignment operator overload does not compile with these
            // components, so an explicit overload is given here.
            // Remove this struct when the minimum Boost requirement is > v1.48.
            struct InterfaceHashStruct
            {
                InterfaceHashStruct& operator=(const InterfaceHashStruct &rhs) { interfaceHash = rhs.interfaceHash; return *this; }
                InterfaceHash interfaceHash;
            };

            struct vertex_state_t {
                typedef boost::vertex_property_tag kind;
            };

            struct vertex_color_t {
                typedef boost::vertex_property_tag kind;
            };

            struct vertex_interface_data_t {
                typedef boost::vertex_property_tag kind;
            };

            /**
             @brief The underlying roadmap graph.

             @par Any BGL graph representation could be used here. Because we
             expect the roadmap to be sparse (m<n^2), an adjacency_list is more
             appropriate than an adjacency_matrix.

             @par Obviously, a ompl::base::State* vertex property is required.
             The incremental connected components algorithm requires
             vertex_predecessor_t and vertex_rank_t properties.
             If boost::vecS is not used for vertex storage, then there must also
             be a boost:vertex_index_t property manually added.

             @par Edges should be undirected and have a weight property.
             */
            typedef boost::adjacency_list <
                boost::vecS, boost::vecS, boost::undirectedS,
                boost::property < vertex_state_t, base::State*,
                boost::property < boost::vertex_predecessor_t, VertexIndexType,
                boost::property < boost::vertex_rank_t, VertexIndexType,
                boost::property < vertex_color_t, GuardType,
                boost::property < vertex_interface_data_t, InterfaceHashStruct > > > > >,
                boost::property < boost::edge_weight_t, double >
            > Graph;

            /** \brief Vertex in Graph */
            typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;

            /** \brief Edge in Graph */
            typedef boost::graph_traits<Graph>::edge_descriptor   Edge;

            /** \brief Constructor */
            SPARStwo(const base::SpaceInformationPtr &si);

            /** \brief Destructor */
            virtual ~SPARStwo(void);

            virtual void setProblemDefinition(const base::ProblemDefinitionPtr &pdef);

            /** \brief Sets the stretch factor */
            void setStretchFactor(double t)
            {
                stretchFactor_ = t;
            }

            /** \brief Sets vertex visibility range as a fraction of max. extent. */
            void setSparseDeltaFraction( double D )
            {
                sparseDeltaFraction_ = D;
                if (sparseDelta_ > 0.0) // setup was previously called
                    sparseDelta_ = D * si_->getMaximumExtent();
            }

            /** \brief Sets interface support tolerance as a fraction of max. extent. */
            void setDenseDeltaFraction( double d )
            {
                denseDeltaFraction_ = d;
                if (denseDelta_ > 0.0) // setup was previously called
                    denseDelta_ = d * si_->getMaximumExtent();
            }

            /** \brief Sets the maximum failures until termination */
            void setMaxFailures( unsigned int m )
            {
                maxFailures_ = m;
            }

            /** \brief Retrieve the maximum consecutive failure limit. */
            unsigned int getMaxFailures( ) const
            {
                return maxFailures_;
            }

            /** \brief Retrieve the dense graph interface support delta. */
            double getDenseDeltaFraction( ) const
            {
                return denseDeltaFraction_;
            }

            /** \brief Retrieve the sparse graph visibility range delta. */
            double getSparseDeltaFraction( ) const
            {
                return sparseDeltaFraction_;
            }

            /** \brief Retrieve the spanner's set stretch factor. */
            double getStretchFactor( ) const
            {
                return stretchFactor_;
            }

            /** \brief While the termination condition permits, construct the spanner graph */
            void constructRoadmap(const base::PlannerTerminationCondition &ptc);

            /** \brief While the termination condition permits, construct the spanner graph. If \e stopOnMaxFail is true,
                the function also terminates when the failure limit set by setMaxFailures() is reached. */
            void constructRoadmap(const base::PlannerTerminationCondition &ptc, bool stopOnMaxFail);

            /** \brief Function that can solve the motion planning
                problem. This function can be called multiple times on
                the same problem, without calling clear() in
                between. This allows the planner to continue work for
                more time on an unsolved problem, for example. Start
                and goal states from the currently specified
                ProblemDefinition are cached. This means that between
                calls to solve(), input states are only added, not
                removed. When using PRM as a multi-query planner, the
                input states should be however cleared, without
                clearing the roadmap itself. This can be done using
                the clearQuery() function. */
            virtual base::PlannerStatus solve(const base::PlannerTerminationCondition &ptc);

            /** \brief Clear the query previously loaded from the ProblemDefinition.
                Subsequent calls to solve() will reuse the previously computed roadmap,
                but will clear the set of input states constructed by the previous call to solve().
                This enables multi-query functionality for PRM. */
            void clearQuery(void);

            virtual void clear(void);

            /** \brief Set a different nearest neighbors datastructure */
            template<template<typename T> class NN>
            void setNearestNeighbors(void)
            {
                nn_.reset(new NN< Vertex >());
                if (isSetup())
                    setup();
            }

            virtual void setup(void);

            /** \brief Retrieve the computed roadmap. */
            const Graph& getRoadmap(void) const
            {
                return g_;
            }

            /** \brief Get the number of vertices in the sparse roadmap. */
            unsigned int milestoneCount(void) const
            {
                return boost::num_vertices(g_);
            }

            /** \brief Get the number of iterations the algorithm performed */
            long unsigned int getIterations(void) const
            {
                return iterations_;
            }

            virtual void getPlannerData(base::PlannerData &data) const;

        protected:

            /** \brief Free all the memory allocated by the planner */
            void freeMemory(void);

            /** \brief Check that the query vertex is initialized (used for internal nearest neighbor searches) */
            void checkQueryStateInitialization(void);

            /** \brief Checks to see if the sample needs to be added to ensure coverage of the space */
            bool checkAddCoverage(const base::State *qNew, std::vector<Vertex> &visibleNeighborhood);

            /** \brief Checks to see if the sample needs to be added to ensure connectivity */
            bool checkAddConnectivity(const base::State *qNew, std::vector<Vertex> &visibleNeighborhood);

            /** \brief Checks to see if the current sample reveals the existence of an interface, and if so, tries to bridge it. */
            bool checkAddInterface(const base::State *qNew, std::vector<Vertex> &graphNeighborhood, std::vector<Vertex> &visibleNeighborhood);

            /** \brief Checks vertex v for short paths through its region and adds when appropriate. */
            bool checkAddPath( Vertex v );

            /** \brief A reset function for resetting the failures count */
            void resetFailures(void);

            /** \brief Finds visible nodes in the graph near st */
            void findGraphNeighbors(base::State* st, std::vector<Vertex> &graphNeighborhood, std::vector<Vertex> &visibleNeighborhood);

            /** \brief Approaches the graph from a given vertex */
            void approachGraph( Vertex v );

            /** \brief Finds the representative of the input state, st  */
            Vertex findGraphRepresentative(base::State* st);

            /** \brief Finds representatives of samples near qNew_ which are not his representative */
            void findCloseRepresentatives(base::State *workArea, const base::State *qNew, Vertex qRep,
                                          std::map<Vertex, base::State*> &closeRepresentatives, const base::PlannerTerminationCondition &ptc);

            /** \brief High-level method which updates pair point information for repV_ with neighbor r */
            void updatePairPoints(Vertex rep, const base::State *q, Vertex r, const base::State *s);

            /** \brief Computes all nodes which qualify as a candidate v" for v and vp */
            void computeVPP(Vertex v, Vertex vp, std::vector<Vertex> &VPPs);

            /** \brief Computes all nodes which qualify as a candidate x for v, v', and v" */
            void computeX(Vertex v, Vertex vp, Vertex vpp, std::vector<Vertex> &Xs);

            /** \brief Rectifies indexing order for accessing the vertex data */
            VertexPair index( Vertex vp, Vertex vpp );

            /** \brief Retrieves the Vertex data associated with v,vp,vpp */
            InterfaceData& getData( Vertex v, Vertex vp, Vertex vpp );

            /** \brief Performs distance checking for the candidate new state, q against the current information */
            void distanceCheck(Vertex rep, const base::State *q, Vertex r, const base::State *s, Vertex rp);

            /** \brief When a new guard is added at state st, finds all guards who must abandon their interface information and deletes that information */
            void abandonLists(base::State* st);

            /** \brief Construct a guard for a given state (\e state) and store it in the nearest neighbors data structure */
            Vertex addGuard(base::State *state, GuardType type);

            /** \brief Connect two guards in the roadmap */
            void connectGuards( Vertex v, Vertex vp );

            /** \brief Check if there exists a solution, i.e., there exists a pair of milestones such that the first is in \e start and the second is in \e goal, and the two milestones are in the same connected component. If a solution is found, the path is saved. */
            bool haveSolution(const std::vector<Vertex> &start, const std::vector<Vertex> &goal, base::PathPtr &solution);

            /** Thread that checks for solution */
            void checkForSolution(const base::PlannerTerminationCondition &ptc, base::PathPtr &solution);

            /** \brief Returns true if we have reached the iteration failures limit, \e maxFailures_ or if a solution was added */
            bool reachedTerminationCriterion(void) const;

            /** \brief Returns whether we have reached the iteration failures limit, maxFailures_ */
            bool reachedFailureLimit (void) const;

            /** \brief Given two milestones from the same connected component, construct a path connecting them and set it as the solution */
            base::PathPtr constructSolution(const Vertex start, const Vertex goal) const;

            /** \brief Check if two milestones (\e m1 and \e m2) are part of the same connected component. This is not a const function since we use incremental connected components from boost */
            bool sameComponent(Vertex m1, Vertex m2);

            /** \brief Compute distance between two milestones (this is simply distance between the states of the milestones) */
            double distanceFunction(const Vertex a, const Vertex b) const
            {
                return si_->distance(stateProperty_[a], stateProperty_[b]);
            }

            /** \brief Sampler user for generating valid samples in the state space */
            base::ValidStateSamplerPtr                                          sampler_;

            /** \brief Sampler user for generating random in the state space */
            base::StateSamplerPtr                                               simpleSampler_;

            /** \brief Nearest neighbors data structure */
            boost::shared_ptr< NearestNeighbors<Vertex> >                       nn_;

            /** \brief Connectivity graph */
            Graph                                                               g_;

            /** \brief Array of start milestones */
            std::vector<Vertex>                                                 startM_;

            /** \brief Array of goal milestones */
            std::vector<Vertex>                                                 goalM_;

            /** \brief Vertex for performing nearest neighbor queries. */
            Vertex                                                              queryVertex_;

            /** \brief Stretch Factor as per graph spanner literature (multiplicative bound on path quality) */
            double                                                              stretchFactor_;

            /** \brief Maximum visibility range for nodes in the graph as a fraction of maximum extent. */
            double                                                              sparseDeltaFraction_;

            /** \brief Maximum range for allowing two samples to support an interface as a fraction of maximum extent. */
            double                                                              denseDeltaFraction_;

            /** \brief The number of consecutive failures to add to the graph before termination */
            unsigned int                                                        maxFailures_;

            /** \brief Number of sample points to use when trying to detect interfaces. */
            unsigned int                                                        nearSamplePoints_;

            /** \brief Access to the internal base::state at each Vertex */
            boost::property_map<Graph, vertex_state_t>::type                    stateProperty_;

            /** \brief A path simplifier used to simplify dense paths added to the graph */
            PathSimplifierPtr                                                   psimp_;

            /** \brief Access to the weights of each Edge */
            boost::property_map<Graph, boost::edge_weight_t>::type              weightProperty_;

            /** \brief Access to the colors for the vertices */
            boost::property_map<Graph, vertex_color_t>::type                    colorProperty_;

            /** \brief Access to the interface pair information for the vertices */
            boost::property_map<Graph, vertex_interface_data_t>::type           interfaceDataProperty_;

            /** \brief Data structure that maintains the connected components */
            boost::disjoint_sets<
                boost::property_map<Graph, boost::vertex_rank_t>::type,
                boost::property_map<Graph, boost::vertex_predecessor_t>::type >
                                                                                disjointSets_;
            /** \brief Random number generator */
            RNG                                                                 rng_;

            /** \brief A flag indicating that a solution has been added during solve() */
            bool                                                                addedSolution_;

            /** \brief A counter for the number of consecutive failed iterations of the algorithm */
            unsigned int                                                        consecutiveFailures_;

            /** \brief A counter for the number of iterations of the algorithm */
            long unsigned int                                                   iterations_;

            /** \brief Maximum visibility range for nodes in the graph */
            double                                                              sparseDelta_;

            /** \brief Maximum range for allowing two samples to support an interface */
            double                                                              denseDelta_;

            /** \brief Mutex to guard access to the Graph member (g_) */
            mutable boost::mutex                                                graphMutex_;

        };

    }
}

#endif
