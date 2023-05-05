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
 * workspace_tags.h
 *
 *  Created on: Jun 24, 2014
 *      Author: Sanjiban Choudhury
 */

#ifndef OMPL_INCLUDE_OMPL_BASE_WORKSPACE_TAGS_H_
#define OMPL_INCLUDE_OMPL_BASE_WORKSPACE_TAGS_H_

namespace ca {
namespace ompl_base {

/**
 * \brief holds possible workspace types
 */
struct WorkspaceTags {
  enum e {
    NONE = 0,
    TIME,
    TRANSLATION,
    ROTATION,
    TANGENT_TRANSLATION,
    TANGENT_ROTATION
  };
};

/**
 * \brief holds possible workspace types in string form
 */
static const char* WorkspaceTagAnnotation[] = { "NONE", "TIME", "TRANSLATION",
    "ROTATION", "TANGENT_TRANSLATION", "TANGENT_ROTATION" };

}  // namespace ompl_base
}  // namespace ca

#endif  // OMPL_INCLUDE_OMPL_WORKSPACE_TAGS_H_ 
