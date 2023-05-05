#ifndef PATH_LENGTH_OBJECTIVE_H
#define PATH_LENGTH_OBJECTIVE_H

#include <ompl/base/objectives/PathLengthOptimizationObjective.h>

namespace roadmap{
namespace ob = ompl::base;
class PathLengthObjective: public ob::PathLengthOptimizationObjective
{
public:
    PathLengthObjective(const ob::SpaceInformationPtr& si):PathLengthOptimizationObjective(si){};

    ob::Cost stateCost(const ob::State* s)const{
        return ob::Cost(1/si_->getStateValidityChecker()->clearance(s));
    }

};


}


#endif // PATH_LENGTH_OBJECTIVE_H
