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
 * se2_tangent_space.h
 *
 *  Created on: Jun 25, 2014
 *      Author: Sanjiban Choudhury
 */

#ifndef OMPL_INCLUDE_OMPL_BASE_SPACES_SE2_TANGENT_SPACE_H_
#define OMPL_INCLUDE_OMPL_BASE_SPACES_SE2_TANGENT_SPACE_H_

#include "ompl/base/StateSpace.h"
#include "ompl/base/spaces/RealVectorStateSpace.h"

namespace ca {

/**
 * \brief extension of ompl::base but under a slightly different name
 */
namespace ompl_base {

/**
 * \brief Describes velocities in SE2
 */
class SE2TangentSpace : public ompl::base::CompoundStateSpace {
 public:

  /**
   * \brief SE2 velocity state
   */
  class StateType : public ompl::base::CompoundStateSpace::StateType {
   public:

    /**
     * \brief Constructor
     */
    StateType(void)
        : CompoundStateSpace::StateType() {
    }

    /**
     * \brief Returns x velocity component
     * @return x velocity
     */
    double GetXDot(void) const {
      return as<ompl::base::RealVectorStateSpace::StateType>(0)->values[0];
    }

    /**
     * \brief Returns y velocity component
     * @return y velocity
     */
    double GetYDot(void) const {
      return as<ompl::base::RealVectorStateSpace::StateType>(0)->values[1];
    }

    /**
     * \brief Returns the rotational velocity
     * @return yaw velocity
     */
    double GetYawDot(void) const {
      return as<ompl::base::RealVectorStateSpace::StateType>(1)->values[0];
    }

    /**
     * \brief Sets the x velocity component
     * @param x_dot the x velocity
     */
    void SetXDot(double x_dot) {
      as<ompl::base::RealVectorStateSpace::StateType>(0)->values[0] = x_dot;
    }

    /**
     * \brief Sets the y velocity component
     * @param y_dot the y velocity
     */
    void SetYDot(double y_dot) {
      as<ompl::base::RealVectorStateSpace::StateType>(0)->values[1] = y_dot;
    }

    /**
     * \brief Sets the rotational velocity
     * @param yaw_dot the rotational velocity
     */
    void SetYawDot(double yaw_dot) {
      as<ompl::base::RealVectorStateSpace::StateType>(1)->values[0] = yaw_dot;
    }

  };

  /**
   * \brief Constructor
   */
  SE2TangentSpace(void)
      : ompl::base::CompoundStateSpace() {
    setName("SE2Tangent" + getName());
    type_ = ompl::base::TANGENT_SPACE_SE2;
    addSubspace(
        ompl::base::StateSpacePtr(new ompl::base::RealVectorStateSpace(3)),
        1.0);
    addSubspace(
        ompl::base::StateSpacePtr(new ompl::base::RealVectorStateSpace(1)),
        1.0);
    lock();
  }

  /**
   * \brief Destructor
   */
  virtual ~SE2TangentSpace(void) {
  }

  /**
   * \brief Sets bounds for the space
   * @param translation_bounds bounds for translational part
   * @param rotation_bounds bounds for rotational part
   */
  void SetBounds(const ompl::base::RealVectorBounds &translation_bounds,
                 const ompl::base::RealVectorBounds &rotation_bounds) {
    as<ompl::base::RealVectorStateSpace>(0)->setBounds(translation_bounds);
    as<ompl::base::RealVectorStateSpace>(1)->setBounds(rotation_bounds);
  }

  /**
   * \brief Returns the bounds for the translational part of the space
   * @return the bounds
   */
  const ompl::base::RealVectorBounds& GetTranslationBounds(void) const {
    return as<ompl::base::RealVectorStateSpace>(0)->getBounds();
  }

  /**
   * \brief Returns the bounds for the rotational part of the space
   * @return the bounds
   */
  const ompl::base::RealVectorBounds& GetRotationBounds(void) const {
    return as<ompl::base::RealVectorStateSpace>(1)->getBounds();
  }

  /**
   * \brief Allocates memory for a state
   * @return Reference to the new state
   */
  virtual ompl::base::State* allocState(void) const;

  /**
   * \brief Frees state from memory
   * @param state reference to the state to free from memory
   */
  virtual void freeState(ompl::base::State *state) const;

  /**
   * \todo comment
   * @param workspace_tag_translation
   * @param workspace_tag_rotation
   */
  void AssignWorkspaceTagSubspace(
      ca::ompl_base::WorkspaceTags::e workspace_tag_translation,
      ca::ompl_base::WorkspaceTags::e workspace_tag_rotation) {
    getSubspace(0)->AssignWorkspaceTag(workspace_tag_translation);
    getSubspace(1)->AssignWorkspaceTag(workspace_tag_rotation);
  }
};

}
}

#endif  // OMPL_INCLUDE_OMPL_BASE_SPACES_SE2_TANGENT_SPACE_H_ 
