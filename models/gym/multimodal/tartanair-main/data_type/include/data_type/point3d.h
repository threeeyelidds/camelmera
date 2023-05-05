#ifndef POINT3D_H
#define POINT3D_H

#include <iostream>
#include <cmath>
namespace data_type {
/*!
 * \brief This class represents a three-dimensional vector
 *
 * The three-dimensional vector can be used to represent a
 * translation in three-dimensional space or to represent the
 * attitude of an object using Euler angle.
 */
class Point3d {
public:

  /*!
   * \brief Default constructor
   */
  Point3d () { data[0] = data[1] = data[2] = 0.0; }

  /*!
   * \brief Copy constructor
   *
   * @param other a vector of dimension 3
   */
  Point3d (const Point3d& other) {
    data[0] = other(0);
    data[1] = other(1);
    data[2] = other(2);
  }

  /*!
   * \brief Constructor
   *
   * Constructs a three-dimensional vector from
   * three single values x, y, z or roll, pitch, yaw
   */
  Point3d (float x, float y, float z) {
    data[0] = x;
    data[1] = y;
    data[2] = z;
  }


  /* inline Eigen3::Point3df getPoint3df() const { return Eigen3::Point3df(data[0], data[1], data[2]) ; } */
  /* inline Eigen3::Vector4f& getVector4f() { return data; } */
  /* inline Eigen3::Vector4f getVector4f() const { return data; } */

  /*!
   * \brief Assignment operator
   *
   * @param other a vector of dimension 3
   */
  inline Point3d& operator= (const Point3d& other)  {
    data[0] = other(0);
    data[1] = other(1);
    data[2] = other(2);
    return *this;
  }


  /*!
   * \brief Three-dimensional vector (cross) product
   *
   * Calculates the tree-dimensional cross product, which
   * represents the vector orthogonal to the plane defined
   * by this and other.
   * @return this x other
   */
  inline Point3d cross (const Point3d& other) const
  {
    //return (data.start<3> ().cross (other.data.start<3> ()));
    // \note should this be renamed?
    return Point3d(y()*other.z() - z()*other.y(),
                   z()*other.x() - x()*other.z(),
                   x()*other.y() - y()*other.x());
  }

  /// dot product
  inline double dot (const Point3d& other) const
  {
    return x()*other.x() + y()*other.y() + z()*other.z();
  }

  inline const float& operator() (unsigned int i) const
  {
    return data[i];
  }
  inline float& operator() (unsigned int i)
  {
    return data[i];
  }

  inline float& x()
  {
    return operator()(0);
  }

  inline float& y()
  {
    return operator()(1);
  }

  inline float& z()
  {
    return operator()(2);
  }

  inline const float& x() const
  {
    return operator()(0);
  }

  inline const float& y() const
  {
    return operator()(1);
  }

  inline const float& z() const
  {
    return operator()(2);
  }

  inline float& roll()
  {
    return operator()(0);
  }

  inline float& pitch()
  {
    return operator()(1);
  }

  inline float& yaw()
  {
    return operator()(2);
  }

  inline const float& roll() const
  {
    return operator()(0);
  }

  inline const float& pitch() const
  {
    return operator()(1);
  }

  inline const float& yaw() const
  {
    return operator()(2);
  }

  inline Point3d operator- () const
  {
    Point3d result;
    result(0) = -data[0];
    result(1) = -data[1];
    result(2) = -data[2];
    return result;
  }

  inline Point3d operator+ (const Point3d &other) const
  {
    Point3d result(*this);
    result(0) += other(0);
    result(1) += other(1);
    result(2) += other(2);
    return result;
  }

  inline Point3d operator*  (float x) const {
    Point3d result(*this);
    result(0) *= x;
    result(1) *= x;
    result(2) *= x;
    return result;
  }

  inline Point3d operator- (const Point3d &other) const
  {
    Point3d result(*this);
    result(0) -= other(0);
    result(1) -= other(1);
    result(2) -= other(2);
    return result;
  }

  inline void operator+= (const Point3d &other)
  {
    data[0] += other(0);
    data[1] += other(1);
    data[2] += other(2);
  }

  inline void operator-= (const Point3d& other) {
    data[0] -= other(0);
    data[1] -= other(1);
    data[2] -= other(2);
  }

  inline void operator/= (float x) {
    data[0] /= x;
    data[1] /= x;
    data[2] /= x;
  }

  inline void operator*= (float x) {
    data[0] *= x;
    data[1] *= x;
    data[2] *= x;
  }

  inline bool operator== (const Point3d &other) const {
    for (unsigned int i=0; i<3; i++) {
      if (operator()(i) != other(i))
        return false;
    }
    return true;
  }

  /// @return length of the vector ("L2 norm")
  inline double norm () const {
    return sqrt(norm_sq());
  }

  /// @return squared length ("L2 norm") of the vector
  inline double norm_sq() const {
    return (x()*x() + y()*y() + z()*z());
  }

  /// normalizes this vector, so that it has norm=1.0
  inline Point3d& normalize () {
    double len = norm();
    if (len > 0)
      *this /= (float) len;
    return *this;
  }

  /// @return normalized vector, this one remains unchanged
  inline Point3d normalized () const {
    Point3d result(*this);
    result.normalize ();
    return result;
  }

  inline double angleTo(const Point3d& other) const {
    double dot_prod = this->dot(other);
    double len1 = this->norm();
    double len2 = other.norm();
    return acos(dot_prod / (len1*len2));
  }


  inline double distance (const Point3d& other) const {
    double dist_x = x() - other.x();
    double dist_y = y() - other.y();
    double dist_z = z() - other.z();
    return sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z);
  }

  inline double distanceXY (const Point3d& other) const {
    double dist_x = x() - other.x();
    double dist_y = y() - other.y();
    return sqrt(dist_x*dist_x + dist_y*dist_y);
  }

  Point3d& rotate_IP (double roll, double pitch, double yaw);

  //    void read (unsigned char * src, unsigned int size);
  std::istream& read(std::istream &s);
  std::ostream& write(std::ostream &s) const;
  std::istream& readBinary(std::istream &s);
  std::ostream& writeBinary(std::ostream &s) const;


protected:
  float data[3];



};

inline std::ostream& operator<<(std::ostream& out, data_type::Point3d const& v){ return out; }

//! user friendly output in format (x y z)

}

#endif // POINT3D_H
