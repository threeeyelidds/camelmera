#ifndef POINT_3I_H
#define POINT_3I_H

#include <iostream>
#include <limits>
#include <cmath>

namespace data_type{

const double INF = std::numeric_limits<double>::infinity();

/**
 * @brief data structure, keep being tune with frontier
 */

class Point3i
{
private:
    int x_val, y_val, z_val;

public:
    inline Point3i():x_val(0),y_val(0),z_val(0){}
    inline Point3i(int x, int y, int z):x_val(x),y_val(y),z_val(z){}
    inline Point3i(const Point3i& point):x_val(point.x()),y_val(point.y()),z_val(point.z()){}

public:
    inline int x() const {return x_val;}
    inline int y() const {return y_val;}
    inline int z() const {return z_val;}

    inline int& x() {return x_val;}
    inline int& y() {return y_val;}
    inline int& z() {return z_val;}

    inline bool operator ==(const Point3i& n2){
        return (x_val==n2.x()) && (y_val==n2.y()) && (z_val==n2.z());
    }

    inline bool operator !=(const Point3i& n2){
        return !((x_val==n2.x()) && (y_val==n2.y()) && (z_val==n2.z()));
    }

    inline double norm () const {
      return sqrt(norm_sq());
    }

    inline double norm_sq() const {
      return double(x()*x() + y()*y() + z()*z());
    }

    inline double manhattan_dist() const{
        return double(abs(x()) + abs(y()) + abs(z()));
    }

    inline Point3i operator- (const Point3i &other) const
    {
      Point3i result(*this);
      result.x() -= other.x();
      result.y() -= other.y();
      result.z() -= other.z();
      return result;
    }
};

//! user friendly output in format (x y z)
inline std::ostream& operator<<(std::ostream& out, const data_type::Point3i& v){
    out << "(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
    return out;
}

}; // namespace end

#endif // PLANNER_TYPE_H
