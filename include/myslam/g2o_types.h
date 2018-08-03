#ifndef G2O_TYPES_H
#define G2O_TYPES_H

#include "myslam/common_include.h"
#include "camera.h"

namespace myslam
{
  class EdgeProjectXYZ2UVPoseOnly:public g2o::BaseUnaryEdge<2,Eigen::Vector2d,g2o::VertexSE3Expmap>
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual void computeError();
    virtual void linearizeOplus();
    
    virtual bool read(std::istream&in){}
    virtual bool write(std::ostream&out)const{};
    
    Vector3d point_;
    Camera::Ptr camera_;
  };
}
#endif