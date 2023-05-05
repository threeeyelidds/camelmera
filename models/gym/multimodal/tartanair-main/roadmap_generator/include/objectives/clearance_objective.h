#ifndef CLEARANCE_OBJECTIVE_H
#define CLEARANCE_OBJECTIVE_H


#include <ompl/base/objectives/StateCostIntegralObjective.h>

namespace roadmap {
namespace ob = ompl::base;

class ClearanceObjective: public ob::StateCostIntegralObjective
{
public:
    ClearanceObjective(const ob::SpaceInformationPtr& si) :ob::StateCostIntegralObjective(si, true){}

    ob::Cost stateCost(const ob::State* s) const{
        return ob::Cost(1/si_->getStateValidityChecker()->clearance(s));
    }

};

}
#endif // CLEARANCE_OBJECTIVE_H
