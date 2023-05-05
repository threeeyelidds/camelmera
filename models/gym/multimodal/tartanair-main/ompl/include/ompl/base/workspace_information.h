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
 * workspace_information.h
 *
 *  Created on: Jun 24, 2014
 *      Author: Sanjiban Choudhury
 */

#ifndef OMPL_INCLUDE_OMPL_BASE_WORKSPACE_INFORMATION_H_
#define OMPL_INCLUDE_OMPL_BASE_WORKSPACE_INFORMATION_H_

#include "ompl/base/StateSpace.h"
#include "ompl/base/workspace_tags.h"
#include "ompl/base/workspace_information.h"
#include <map>

namespace ca {
namespace ompl_base {

/**
 * \brief stores mapping between workspace and statespace names in a compound statespace
 */
class WorkspaceInformation {
 public:
  typedef std::map<WorkspaceTags::e, std::string> TagMap;

  /**
   * \brief Default constructor
   */
  WorkspaceInformation()
      : workspace_map_() {
  }
  ;

  /**
   * \brief Destructor
   */
  ~WorkspaceInformation() {
  }
  ;

  /**
   * \brief Sets up the mapping between workspace tags and their corresponding statespace names for a given compound statespace
   * @param space_ptr reference to the compound state space
   */
  void Setup(const ompl::base::StateSpacePtr &space_ptr);

  /**
   * \brief Pipe the mapping between workspace and states space names in string format
   * @param out the stream to pipe to
   */
  void PrintInformation(std::ostream &out) const;

  /**
   * \brief Retrieve the actual mapping
   * @return Copy of the mapping
   */
  const TagMap workspace_map() const {
    return workspace_map_;
  }

 protected:

  /**
   * \brief Sets up default mappings between some workspace and statespace types
   * @param space_ptr reference to the compound statepace
   */
  void DefaultSetup(const ompl::base::StateSpacePtr &space_ptr);

  /**
   * \brief Checks if the workspace mappings were set correctly from the given statespace
   * @param space_ptr reference to the statespace
   * @return true if setup correctly
   */
  bool SanityCheck(const ompl::base::StateSpacePtr &space_ptr);

  /**
   * \brief Checks if the given workspace tag is set in the mapping
   * @param tag the workspace tag
   * @return true if there is a mapping between the given tag and a statespace
   */
  bool IsPresent(WorkspaceTags::e tag);

  /**
   * \brief Retrieves the dimension of the statespace corresponding to a given workspace
   * @param tag the workspace type
   * @param space_ptr reference to the compound statespace
   */
  unsigned Dimension(WorkspaceTags::e tag,
                     const ompl::base::StateSpacePtr &space_ptr);

  /**
   * \brief Retrieves the statepace type from a given workspace tag
   * @param tag the workspace tag
   * @param space_ptr reference to compound statespace
   * @return the type in the form of an int
   */
  unsigned int Types(WorkspaceTags::e tag,
                     const ompl::base::StateSpacePtr &space_ptr);
  TagMap workspace_map_;
};

}  // namespace ompl_base
}  // namespace ca

#endif  // OMPL_INCLUDE_OMPL_BASE_WORKSPACE_INFORMATION_H_ 
