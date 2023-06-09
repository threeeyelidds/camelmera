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
*  Copyright (c) 2011, Willow Garage
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

#ifndef OMPL_BASE_GENERIC_PARAM_
#define OMPL_BASE_GENERIC_PARAM_

#include "ompl/util/Console.h"
#include "ompl/util/ClassForward.h"
#include <boost/function.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <map>

namespace ompl
{
    namespace base
    {

        /// @cond IGNORE
        /** \brief Forward declaration of ompl::base::GenericParam */
        OMPL_CLASS_FORWARD(GenericParam);
        /// @endcond

        /** \brief Motion planning algorithms often employ parameters
            to guide their exploration process. (e.g., goal
            biasing). Motion planners (and some of their components)
            use this class to declare what the parameters are, in a
            generic way, so that they can be set externally. */
        class GenericParam
        {
        public:

            /** \brief The constructor of a parameter takes the name of the parameter (\e name) */
            GenericParam(const std::string &name) : name_(name)
            {
            }

            virtual ~GenericParam(void)
            {
            }

            /** \brief Get the name of the parameter */
            const std::string& getName(void) const
            {
                return name_;
            }

            /** \brief Set the name of the parameter */
            void setName(const std::string &name)
            {
                name_ = name;
            }

            /** \brief Set the value of the parameter. The value is taken in as a string, but converted to the type of that parameter. */
            virtual bool setValue(const std::string &value) = 0;

            /** \brief Retrieve the value of the parameter, as a string. */
            virtual std::string getValue(void) const = 0;

            /** \brief Assignment operator by type. This is just for convenience, as it just calls setValue() */
            template<typename T>
            GenericParam& operator=(const T &value)
            {
                try
                {
                    setValue(boost::lexical_cast<std::string>(value));
                }
                catch (boost::bad_lexical_cast &e)
                {
                    OMPL_WARN("Invalid value format specified for parameter '%s': %s", name_.c_str(), e.what());
                }
                return *this;
            }

            /** \brief Set a suggested range */
            void setRangeSuggestion(const std::string &rangeSuggestion)
            {
                rangeSuggestion_ = rangeSuggestion;
            }

            /** \brief Get the suggested range of values */
            const std::string& getRangeSuggestion(void) const
            {
                return rangeSuggestion_;
            }

        protected:

            /** \brief The name of the parameter */
            std::string name_;

            /** \brief Suggested range for the parameter

                This can be used to provide a hint to, e.g., a GUI. The
                convention used in OMPL is to denote ranges for the
                following types as follows:
                - \c bool: "0,1"
                - \c enum: "<enum_val0>,<enum_val1>,<enum_val2>,..."
                - \c int, \c double: either "first:last" or "first:stepsize:last".
                  In the first case, the stepsize is assumed to be 1. It is
                  important to use floating point representations for double
                  ranges (i.e., "1." instead of "1") to make sure the type is
                  deduced correctly.
            */
            std::string rangeSuggestion_;
        };


        /** \brief This is a helper class that instantiates parameters with different data types. */
        template<typename T>
        class SpecificParam : public GenericParam
        {
        public:

            /** \brief The type for the 'setter' function for this parameter */
            typedef boost::function<void(T)> SetterFn;

            /** \brief The type for the 'getter' function for this parameter */
            typedef boost::function<T()>     GetterFn;

            /** \brief An explicit instantiation of a parameter \e name requires the \e setter function and optionally the \e
                getter function. */
            SpecificParam(const std::string &name, const SetterFn &setter, const GetterFn &getter = GetterFn()) :
                GenericParam(name), setter_(setter), getter_(getter)
            {
                if (!setter_)
                    OMPL_ERROR("Setter function must be specified for parameter");
            }

            virtual ~SpecificParam(void)
            {
            }

            virtual bool setValue(const std::string &value)
            {
                bool result = true;
                try
                {
                    if (setter_)
                        setter_(boost::lexical_cast<T>(value));
                }
                catch (boost::bad_lexical_cast &e)
                {
                    result = false;
                    OMPL_WARN("Invalid value format specified for parameter '%s': %s", name_.c_str(), e.what());
                }

                if (getter_)
                    OMPL_DEBUG("The value of parameter '%s' is now: '%s'", name_.c_str(), getValue().c_str());
                else
                    OMPL_DEBUG("The value of parameter '%s' was set to: '%s'", name_.c_str(), value.c_str());
                return result;
            }

            virtual std::string getValue(void) const
            {
                if (getter_)
                    try
                    {
                        return boost::lexical_cast<std::string>(getter_());
                    }
                    catch (boost::bad_lexical_cast &e)
                    {
                        OMPL_WARN("Unable to parameter '%s' to string: %s", name_.c_str(), e.what());
                        return "";
                    }
                else
                    return "";
            }

        protected:

            /** \brief The setter function for this parameter */
            SetterFn setter_;

            /** \brief The getter function for this parameter */
            GetterFn getter_;
        };

        /** \brief Maintain a set of parameters */
        class ParamSet
        {
        public:

            /** \brief This function declares a parameter \e name, and specifies the \e setter and \e getter functions. */
            template<typename T>
            void declareParam(const std::string &name, const typename SpecificParam<T>::SetterFn &setter,
                              const typename SpecificParam<T>::GetterFn &getter = typename SpecificParam<T>::GetterFn())
            {
                params_[name].reset(new SpecificParam<T>(name, setter, getter));
            }

            /** \brief Add a parameter to the set */
            void add(const GenericParamPtr &param);

            /** \brief Remove a parameter from the set */
            void remove(const std::string &name);

            /** \brief Include the params of a different ParamSet into this one. Optionally include a prefix for each of the parameters */
            void include(const ParamSet &other, const std::string &prefix = "");

            /** \brief Algorithms in OMPL often have parameters that
                can be set externally. While each algorithm will have
                their own getter and setter functions specifically for
                those parameters, this function allows setting
                parameters generically, for any algorithm that
                declares parameters, by specifying the parameter name
                \e key and its value \e value (both as string, but \e
                value is cast to the type desired by the corresponding
                setter). Under the hood, this calls SpecificParam::setValue().
                This ability makes it easy to automatically configure
                using external sources (e.g., a configuration
                file). The function returns true if the parameter was
                parsed and set successfully and false otherwise. */
            bool setParam(const std::string &key, const std::string &value);

            /** \brief Get the value of the parameter named \e key. Store the value as string in \e value and return true if the parameter was found. Return false otherwise. */
            bool getParam(const std::string &key, std::string &value) const;

            /** \brief Set the values for a set of parameters. The parameter names are the keys in the map \e kv.
                The corresponding key values in \e kv are set as the parameter values.
                Return true if all parameters were set successfully. This function simply calls setParam() multiple times.
                If \e ignoreUnknown is true, then no attempt is made to set unknown
                parameters (and thus no errors are reported) */
            bool setParams(const std::map<std::string, std::string> &kv, bool ignoreUnknown = false);

            /** \brief Get the known parameter as a map from names to their values cast as string */
            void getParams(std::map<std::string, std::string> &params) const;

            /** \brief List the names of the known parameters */
            void getParamNames(std::vector<std::string> &params) const;

            /** \brief List the values of the known parameters, in the same order as getParamNames() */
            void getParamValues(std::vector<std::string> &vals) const;

            /** \brief Get the map from parameter names to parameter descriptions */
            const std::map<std::string, GenericParamPtr>& getParams(void) const;

            /** \brief Get the parameter that corresponds to a specified name. An empty shared ptr is returned if the parameter does not exist */
            const GenericParamPtr& getParam(const std::string &key) const;

            /** \brief Check whether this set of parameters includes the parameter named \e key */
            bool hasParam(const std::string &key) const;

            /** \brief Access operator for parameters, by name. If the parameter is not defined, an exception is thrown */
            GenericParam& operator[](const std::string &key);

            /** \brief Get the number of parameters maintained by this instance */
            std::size_t size(void) const
            {
                return params_.size();
            }

            /** \brief Clear all the set parameters */
            void clear(void);

            /** \brief Print the parameters to a stream */
            void print(std::ostream &out) const;

        private:

            std::map<std::string, GenericParamPtr> params_;
        };
    }
}

#endif
