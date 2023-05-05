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
 * workspace_information.cpp
 *
 *  Created on: Jun 24, 2014
 *      Author: Sanjiban Choudhury
 */

#include "ompl/base/workspace_information.h"
#include "boost/assert.hpp"

namespace ob = ompl::base;
namespace ca {
namespace ompl_base {

void WorkspaceInformation::Setup(const ob::StateSpacePtr &space_ptr) {
  workspace_map_.clear();
  typedef std::map<std::string, ob::StateSpace::SubstateLocation> state_map;
  const state_map substate_loc = space_ptr->getSubstateLocationsByName();
  for (state_map::const_iterator it = substate_loc.begin(); it != substate_loc.end(); ++it) {
    if (it->second.space->workspace_tag() != WorkspaceTags::NONE)
      if (workspace_map_.count(it->second.space->workspace_tag()) == 0)
        workspace_map_[it->second.space->workspace_tag()] = it->first;
  }
  if (workspace_map_.empty())
    DefaultSetup(space_ptr);
  if (!SanityCheck(space_ptr))
    workspace_map_.clear();
}

void WorkspaceInformation::PrintInformation(std::ostream &out) const {
  out << "Workspace information "<< std::endl;
  for (TagMap::const_iterator it = workspace_map_.begin(); it != workspace_map_.end(); ++it)
    out<<"Tag: "<<WorkspaceTagAnnotation[it->first]<<" StateSpace Name: "<<it->second<<std::endl;
}

void WorkspaceInformation::DefaultSetup (const ompl::base::StateSpacePtr &space_ptr) {
  switch (space_ptr->getType()) {
    case ob::STATE_SPACE_REAL_VECTOR:
      if (space_ptr->getDimension() >=2 && space_ptr->getDimension() <= 3)
        workspace_map_[WorkspaceTags::TRANSLATION] = space_ptr->getName();
      break;
    case ob::STATE_SPACE_SE2:
      workspace_map_[WorkspaceTags::TRANSLATION] = space_ptr->as<ob::CompoundStateSpace>()->getSubspace(0)->getName();
      workspace_map_[WorkspaceTags::ROTATION] = space_ptr->as<ob::CompoundStateSpace>()->getSubspace(1)->getName();
      break;
    case ob::STATE_SPACE_SE3:
      workspace_map_[WorkspaceTags::TRANSLATION] = space_ptr->as<ob::CompoundStateSpace>()->getSubspace(0)->getName();
      workspace_map_[WorkspaceTags::ROTATION] = space_ptr->as<ob::CompoundStateSpace>()->getSubspace(1)->getName();
      break;
  }
}

bool WorkspaceInformation::SanityCheck(const ompl::base::StateSpacePtr &space_ptr) {

  if ( IsPresent(WorkspaceTags::TRANSLATION) && (Dimension(WorkspaceTags::TRANSLATION, space_ptr) < 2 || Dimension(WorkspaceTags::TRANSLATION, space_ptr) > 3 ))
    return false;

  if ( IsPresent(WorkspaceTags::ROTATION) && !(Dimension(WorkspaceTags::ROTATION, space_ptr) == 1 || Dimension(WorkspaceTags::ROTATION, space_ptr) ==3 ))
    return false;

  if ( IsPresent(WorkspaceTags::TANGENT_TRANSLATION) && (Dimension(WorkspaceTags::TANGENT_TRANSLATION, space_ptr) < 2 || Dimension(WorkspaceTags::TANGENT_TRANSLATION, space_ptr) > 3 ))
    return false;

  if ( IsPresent(WorkspaceTags::TANGENT_ROTATION) && !(Dimension(WorkspaceTags::TANGENT_ROTATION, space_ptr) == 1 || Dimension(WorkspaceTags::TANGENT_ROTATION, space_ptr) ==3 ))
    return false;

  if ( IsPresent(WorkspaceTags::TRANSLATION) && IsPresent(WorkspaceTags::TANGENT_TRANSLATION) && Dimension(WorkspaceTags::TRANSLATION, space_ptr) != Dimension(WorkspaceTags::TANGENT_TRANSLATION, space_ptr))
    return false;

  if ( IsPresent(WorkspaceTags::ROTATION) && IsPresent(WorkspaceTags::TANGENT_ROTATION) && Dimension(WorkspaceTags::ROTATION, space_ptr) != Dimension(WorkspaceTags::TANGENT_ROTATION, space_ptr))
    return false;

  if ( IsPresent(WorkspaceTags::TIME) && Dimension(WorkspaceTags::TIME, space_ptr) != 1)
    return false;

  // Now check types
  if ( IsPresent(WorkspaceTags::TRANSLATION) && Types(WorkspaceTags::TRANSLATION, space_ptr) != ob::STATE_SPACE_REAL_VECTOR)
    return false;

  if ( IsPresent(WorkspaceTags::ROTATION) && !(Types(WorkspaceTags::ROTATION, space_ptr) == ob::STATE_SPACE_SO2 || Types(WorkspaceTags::ROTATION, space_ptr) == ob::STATE_SPACE_SO3))
    return false;

  if ( IsPresent(WorkspaceTags::TANGENT_TRANSLATION) && Types(WorkspaceTags::TANGENT_TRANSLATION, space_ptr) != ob::STATE_SPACE_REAL_VECTOR)
    return false;

  if ( IsPresent(WorkspaceTags::TANGENT_ROTATION) && Types(WorkspaceTags::TANGENT_ROTATION, space_ptr) != ob::STATE_SPACE_REAL_VECTOR)
    return false;

  return true;
}

bool WorkspaceInformation::IsPresent (WorkspaceTags::e tag) {
  return workspace_map_.count(tag) > 0;
}

unsigned WorkspaceInformation::Dimension (WorkspaceTags::e tag, const ompl::base::StateSpacePtr &space_ptr) {
  return space_ptr->getSubstateLocationsByName().at(workspace_map_.at(tag)).space->getDimension();
}

unsigned int WorkspaceInformation::Types (WorkspaceTags::e tag, const ompl::base::StateSpacePtr &space_ptr) {
  return space_ptr->getSubstateLocationsByName().at(workspace_map_.at(tag)).space->getType();
}

}  // namespace ompl_base
}  // namespace ca

