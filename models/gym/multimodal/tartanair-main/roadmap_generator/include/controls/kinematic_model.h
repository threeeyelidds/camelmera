#ifndef KINEMATIC_MODEL_H
#define KINEMATIC_MODEL_H

#include <ompl/control/SpaceInformation.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/control/ODESolver.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

namespace roadmap {
namespace ob = ompl::base;
namespace oc = ompl::control;


class KinematicModel : public oc::StatePropagator
{
public:
    KinematicModel(const oc::SpaceInformationPtr &si);

    // state is the starting point of propagation
    void propagate(const ob::State *state, const oc::Control* control,
                   const double duration, ob::State *result) const override;
protected:
    // Explicit Euler Method for numerical integration.
    void EulerIntegration(const ob::State *start, const oc::Control *control,
                          const double duration, ob::State *result) const;
    void ode(const ob::State *state, const oc::Control *control,
             std::valarray<double> &dstate) const;

    void update(ob::State *state, const std::valarray<double> &dstate) const;

    double carLength_;
    double timeStep_;
    ob::StateSpacePtr space_;


};


}


#endif // KINEMATIC_MODEL_H
