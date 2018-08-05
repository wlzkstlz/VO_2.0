#ifndef VO2_H
#define VO2_H
#include "visual_odometry.h"

namespace myslam
{
  class VO2:public VisualOdometry
  {
  public:
    VO2();
    virtual bool addFrame( Frame::Ptr frame );
    virtual void featureMatching();
    virtual void poseEstimationPnP();
  public:
    vector<myslam::MapPoint> mappoints2add;
    SE3 T_c_w_estimated;
  };
}

#endif