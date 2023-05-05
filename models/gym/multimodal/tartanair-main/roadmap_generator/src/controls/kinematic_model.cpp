#include "controls/kinematic_model.h"

namespace roadmap{

roadmap::KinematicModel::KinematicModel(const oc::SpaceInformationPtr &si): oc::StatePropagator(si){
    space_     = si->getStateSpace();
    carLength_ = 0.2;
    timeStep_  = 0.01;
}

void KinematicModel::propagate(const ompl::base::State *state,
                               const ompl::control::Control *control,
                               const double duration, ompl::base::State *result) const
{
    this->EulerIntegration(state, control, 0.05, result);
}

void KinematicModel::EulerIntegration(const ob::State *start, const oc::Control *control,
                                      const double duration, ob::State *init) const
{
    double t = timeStep_;

    std::valarray<double> dstate;
    space_->copyState(init, start);

    while (t < duration + std::numeric_limits<double>::epsilon()) {

        this->ode(init, control, dstate);
        this->update(init, timeStep_*dstate);
        t += timeStep_;
    }

    if (t + std::numeric_limits<double>::epsilon() > duration){
        this->ode(init, control, dstate);
        this->update(init, (t - duration) * dstate);
    }

}

void KinematicModel::ode(const ob::State *start, const oc::Control *control, std::valarray<double> &dstate) const
{
    // refer to http://planning.cs.uiuc.edu/node1.html

    // starting state for propagation
    const double yaw_rot = start->as<ob::CompoundState>()->as<ob::SO2StateSpace::StateType>(1)->value;

    // control signals
    const double *u = control->as<oc::RealVectorControlSpace::ControlType>()->values;
    const double linear_vel = u[0], steer_ang = u[1], z_vel = u[2];
    const double circle_radius = tan(steer_ang) / carLength_;

    // project to geometry space
    dstate.resize(4);
    dstate[0] = linear_vel * cos(yaw_rot);
    dstate[1] = linear_vel * sin(yaw_rot);
    dstate[2] = z_vel;
    dstate[3] = linear_vel * circle_radius;  //agular velocity

}

void KinematicModel::update(ob::State *start, const std::valarray<double> &dstate) const
{
    ob::RealVectorStateSpace::StateType &xyz = *start->as<ob::CompoundState>()->as<ob::RealVectorStateSpace::StateType>(0);
    ob::SO2StateSpace::StateType &yaw = *start->as<ob::CompoundState>()->as<ob::SO2StateSpace::StateType>(1);

    xyz.values[0] += dstate[0];
    xyz.values[1] += dstate[1];
    xyz.values[2] += dstate[2];
    yaw.value += dstate[3];
    space_->enforceBounds(start);
}


}
