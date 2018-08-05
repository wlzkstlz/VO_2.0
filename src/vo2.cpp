#include "myslam/vo2.h"
#include "myslam/config.h"

#include <opencv2/calib3d/calib3d.hpp>

namespace myslam 
{
VO2::VO2():VisualOdometry()
{
  
}

bool VO2::addFrame(Frame::Ptr frame)
{
    switch ( state_ )
    {
    case INITIALIZING:
    {
        state_ = OK;
        curr_ = ref_ = frame;
        map_->insertKeyFrame ( frame );
        // extract features from first frame 
        extractKeyPoints();
        computeDescriptors();
	// add mappoints to map
	//mappoints2add.clear();
	for(int i=0;i<keypoints_curr_.size();i++)
	{
	  myslam::MapPoint::Ptr mappoint(new myslam::MapPoint);
	  Vector2d pt(keypoints_curr_[i].pt.x,keypoints_curr_[i].pt.y);
	  mappoint->pos_=curr_->camera_->pixel2camera(pt,
		curr_->findDepth(keypoints_curr_[i]));
	  mappoint->norm_=mappoint->pos_;
	  mappoint->descriptor_=descriptors_curr_.row(i).clone();
	  map_->insertMapPoint(mappoint);
	}
        break;
    }
    case OK:
    {
        curr_ = frame;
        extractKeyPoints();
        computeDescriptors();
        featureMatching();
        poseEstimationPnP();
        if ( checkEstimatedPose() == true ) // a good estimation
        {
            curr_->T_c_w_ = T_c_w_estimated;  // T_c_w = T_c_r*T_r_w 
            ref_ = curr_;
            num_lost_ = 0;
            if ( checkKeyFrame() == true ) // is a key-frame
            {
                addKeyFrame();
            }
        }
        else // bad estimation due to various reasons
        {
            num_lost_++;
            if ( num_lost_ > max_num_lost_ )
            {
                state_ = LOST;
            }
            return false;
        }
        break;
    }
    case LOST:
    {
        cout<<"vo has lost."<<endl;
        break;
    }
    }

    return true;
}

void VO2::featureMatching()
{
  //prepare mappoint for match;
  curr_->T_c_w_=ref_->T_c_w_;
  Mat descriptor_map_inrange;
  vector<int>indice;
  pt3d_matching.clear();
  for(auto it=map_->map_points_.begin();it!=map_->map_points_.end();it++)
  {
    if(curr_->isInFrame(it->second->pos_))
    {
      it->second->observed_times_++;
      descriptor_map_inrange.push_back(it->second->descriptor_);
      pt3d_matching.push_back(cv::Point3f(it->second->pos_(0,0),
					  it->second->pos_(1,0),it->second->pos_(2,0)));
      indice.push_back(it->first);
    }
  }
  
  //featureMatching
  cv::FlannBasedMatcher matcher;
  vector<cv::DMatch>matches;
  matcher.match(descriptor_map_inrange,descriptors_curr_,matches);
 
  
    // select the best matches
    float min_dis = std::min_element (matches.begin(), matches.end(),
					[] ( const cv::DMatch& m1, const cv::DMatch& m2 )
					{return m1.distance < m2.distance;} 
				     )->distance;
    feature_matches_.clear();
    for ( cv::DMatch& m : matches )
    {
        if ( m.distance < max<float> ( min_dis*match_ratio_, 30.0 ) )
        {
            feature_matches_.push_back(m);
        }
    }
    
    sort(feature_matches_.begin(),feature_matches_.end(),[](const cv::DMatch&m1,const cv::DMatch&m2)
							  {return m1.distance<m2.distance;});
    vector<cv::DMatch>tmp;
    int sz=min<int>(feature_matches_.size(),myslam::Config::get<int>("minmum_distance_match_num"));
    for(int i=0;i<sz;i++)
      tmp.push_back(feature_matches_[i]);
    swap(tmp,feature_matches_);
    
    cout<<"good matches: "<<feature_matches_.size()<<endl;
}

void VO2::poseEstimationPnP()
{
    // construct the 3d 2d observations
    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;
    
    for ( cv::DMatch m:feature_matches_ )
    {
        pts3d.push_back( pt3d_matching[m.queryIdx] );
        pts2d.push_back( keypoints_curr_[m.trainIdx].pt );
    }
    
    Mat K = ref_->camera_->getK();
    Mat rvec, tvec, inliers;
    
    cv::solvePnPRansac( pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
    num_inliers_ = inliers.rows;
    cout<<"pnp inliers: "<<num_inliers_<<endl;
    T_c_w_estimated = SE3(
        SO3(rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0)), 
        Vector3d( tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0))
    );
    
    //bundle_adjustment；

    if(myslam::Config::get<int>("use_bundleadjustment"))
    {
      g2o::SE3Quat se3_quat(T_c_w_estimated.rotation_matrix(),T_c_w_estimated.translation());
      
      if(myslam::Config::get<int>("use_ba_pose_only"))
	bundle_adjustment_pose_only(pts3d,pts2d,K,se3_quat);
      else
	bundle_adjustment(pts3d,pts2d,K,se3_quat);
      T_c_w_estimated=SE3(se3_quat.rotation(),se3_quat.translation());
    }  
    
    T_c_r_estimated_=T_c_w_estimated*ref_->T_c_w_.inverse();
}

}