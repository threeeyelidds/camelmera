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
 * se3_tangent_space.h
 *
 *  Created on: Jun 25, 2014
 *      Author: Sanjiban Choudhury
 */

#ifndef OMPL_INCLUDE_OMPL_BASE_SPACES_SE3_TANGENT_SPACE_H_
#define OMPL_INCLUDE_OMPL_BASE_SPACES_SE3_TANGENT_SPACE_H_

#include "ompl/base/StateSpace.h"
#include "ompl/base/spaces/RealVectorStateSpace.h"

namespace ca {
namespace ompl_base {

/**
 * \brief Describes velocities is SE3
 */
class SE3TangentSpace : public ompl::base::CompoundStateSpace {
 public:

  /**
   * \brief SE3 velocity state
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
     * \brief Returns translational x velocity component
     * @return x velocity
     */
    double GetXDot(void) const {
      return as<ompl::base::RealVectorStateSpace::StateType>(0)->values[0];
    }

    /**
     * \brief Returns translational y velocity component
     * @return y velocity
     */
    double GetYDot(void) const {
      return as<ompl::base::RealVectorStateSpace::StateType>(0)->values[1];
    }

    /**
     * \brief Returns translational z velocity component
     * @return z velocity
     */
    double GetZDot(void) const {
      return as<ompl::base::RealVectorStateSpace::StateType>(0)->values[2];
    }

    /**
     * \brief Returns rotational velocity component around x axis
     * @return phi dot
     */
    double GetPhiDot(void) const {
      return as<ompl::base::RealVectorStateSpace::StateType>(1)->values[0];
    }

    /**
     * \brief Returns rotational velocity component around y axis
     * @return theta dot
     */
    double GetThetaDot(void) const {
      return as<ompl::base::RealVectorStateSpace::StateType>(1)->values[1];
    }

    /**
     * \brief Returns rotational velocity component around z axis
     * @return psi dot
     */
    double GetPsiDot(void) const {
      return as<ompl::base::RealVectorStateSpace::StateType>(1)->values[2];
    }

    /**
     * \brief Sets the translational x velocity component
     * @param x_dot
     */
    void SetXDot(double x_dot) {
      as<ompl::base::RealVectorStateSpace::StateType>(0)->values[0] = x_dot;
    }

    /**
     * \brief Sets the translational y velocity component
     * @param y_dot
     */
    void SetYDot(double y_dot) {
      as<ompl::base::RealVectorStateSpace::StateType>(0)->values[1] = y_dot;
    }

    /**
     * \brief Sets the translational z velocity component
     * @param z_dot
     */
    void SetZDot(double z_dot) {
      as<ompl::base::RealVectorStateSpace::StateType>(0)->values[2] = z_dot;
    }

    /**
     * \brief Set the complete translational velocity
     * @param x_dot x velocity
     * @param y_dot y velocity
     * @param z_dot z velocity
     */
    void SetXYZDot(double x_dot, double y_dot, double z_dot) {
      SetXDot(x_dot);
      SetYDot(y_dot);
      SetZDot(z_dot);
    }

    /**
     * \brief Sets the rotational velocity component along the x axis
     * @param phi_dot
     */
    void SetPhiDot(double phi_dot) {
      as<ompl::base::RealVectorStateSpace::StateType>(1)->values[0] = phi_dot;
    }

    /**
     * \brief Sets the rotational velocity component along the y axis
     * @param theta_dot
     */
    void SetThetaDot(double theta_dot) {
      as<ompl::base::RealVectorStateSpace::StateType>(1)->values[1] = theta_dot;
    }

    /**
     * \brief Sets the rotational velocity component along the z axis
     * @param psi_dot
     */
    void SetPsiDot(double psi_dot) {
      as<ompl::base::RealVectorStateSpace::StateType>(1)->values[2] = psi_dot;
    }

    /**
     * \brief Sets the complete rotational velocity
     * @param phi_dot rotation about x axis
     * @param theta_dot rotation about y axis
     * @param psi_dot rotation about z axiss
     */
    void SetPhiThetaPsiDot(double phi_dot, double theta_dot, double psi_dot) {
      SetPhiDot(phi_dot);
      SetThetaDot(theta_dot);
      SetPsiDot(psi_dot);
    }

  };

  /**
   * \brief Constructor
   */
  SE3TangentSpace(void)
      : ompl::base::CompoundStateSpace() {
    setName("SE3Tangent" + getName());
    type_ = ompl::base::TANGENT_SPACE_SE3;
    addSubspace(
        ompl::base::StateSpacePtr(new ompl::base::RealVectorStateSpace(3)),
        1.0);
    addSubspace(
        ompl::base::StateSpacePtr(new ompl::base::RealVectorStateSpace(3)),
        1.0);
    lock();
  }

  /**
   * \brief Destructor
   */
  virtual ~SE3TangentSpace(void) {
  }

  /**
   * \brief Sets bounds for the space
   * @param translation_bounds bounds for the translational part
   * @param rotation_bounds bounds for the rotational part
   */
  void SetBounds(const ompl::base::RealVectorBounds &translation_bounds,
                 const ompl::base::RealVectorBounds &rotation_bounds) {
    as<ompl::base::RealVectorStateSpace>(0)->setBounds(translation_bounds);
    as<ompl::base::RealVectorStateSpace>(1)->setBounds(rotation_bounds);
  }

  /**
   * \brief Retrieve the bounds for the translational portion
   * @return
   */
  const ompl::base::RealVectorBounds& GetTranslationBounds(void) const {
    return as<ompl::base::RealVectorStateSpace>(0)->getBounds();
  }

  /**
   * \brief Retrieves the bounds for the rotational portion
   * @return
   */
  const ompl::base::RealVectorBounds& GetRotationBounds(void) const {
    return as<ompl::base::RealVectorStateSpace>(1)->getBounds();
  }

  /**
   * \brief Allocates memory for a state, of the this space
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

#endif  // OMPL_INCLUDE_OMPL_BASE_SPACES_SE3_TANGENT_SPACE_H_ 
