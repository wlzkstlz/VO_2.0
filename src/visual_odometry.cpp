/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include "myslam/g2o_types.h"

namespace myslam
{

VisualOdometry::VisualOdometry() :
    state_ ( INITIALIZING ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 )
{
    num_of_features_    = Config::get<int> ( "number_of_features" );
    scale_factor_       = Config::get<double> ( "scale_factor" );
    level_pyramid_      = Config::get<int> ( "level_pyramid" );
    match_ratio_        = Config::get<float> ( "match_ratio" );
    max_num_lost_       = Config::get<float> ( "max_num_lost" );
    min_inliers_        = Config::get<int> ( "min_inliers" );
    max_motion_norm	= Config::get<double>("max_motion_norm");
    key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
    key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
    orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );
}

VisualOdometry::~VisualOdometry()
{

}

bool VisualOdometry::addFrame ( Frame::Ptr frame )
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
        // compute the 3d position of features in ref frame 
        setRef3DPoints();
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
            curr_->T_c_w_ = T_c_r_estimated_ * ref_->T_c_w_;  // T_c_w = T_c_r*T_r_w 
            ref_ = curr_;
            setRef3DPoints();
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

void VisualOdometry::extractKeyPoints()
{
    orb_->detect ( curr_->color_, keypoints_curr_ );
}

void VisualOdometry::computeDescriptors()
{
    orb_->compute ( curr_->color_, keypoints_curr_, descriptors_curr_ );
}

void VisualOdometry::featureMatching()
{
    // match desp_ref and desp_curr, use OpenCV's brute force match 
    vector<cv::DMatch> matches;
    cv::BFMatcher matcher ( cv::NORM_HAMMING );
    matcher.match ( descriptors_ref_, descriptors_curr_, matches );
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

void VisualOdometry::setRef3DPoints()
{
    // select the features with depth measurements 
    pt3d_matching.clear();
    descriptors_ref_ = Mat();
    for ( size_t i=0; i<keypoints_curr_.size(); i++ )
    {
        double d = ref_->findDepth(keypoints_curr_[i]);               
        if ( d > 0)
        {
            Vector3d p_cam = ref_->camera_->pixel2camera(
                Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y), d
            );
            pt3d_matching.push_back( cv::Point3f( p_cam(0,0), p_cam(1,0), p_cam(2,0) ));
            descriptors_ref_.push_back(descriptors_curr_.row(i));
        }
    }
}

void VisualOdometry::poseEstimationPnP()
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
    
    cout<<"debug K="<<endl<<K<<endl;
    
    Mat rvec, tvec, inliers;
    cv::solvePnPRansac( pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
    num_inliers_ = inliers.rows;
    cout<<"pnp inliers: "<<num_inliers_<<endl;
    cout<<"inliers:"<<inliers<<endl;
    T_c_r_estimated_ = SE3(
        SO3(rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0)), 
        Vector3d( tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0))
    );
    
    //bundle_adjustment；

    if(myslam::Config::get<int>("use_bundleadjustment"))
    {
      g2o::SE3Quat se3_quat(T_c_r_estimated_.rotation_matrix(),T_c_r_estimated_.translation());
      
      if(myslam::Config::get<int>("use_ba_pose_only"))
	bundle_adjustment_pose_only(pts3d,pts2d,K,se3_quat);
      else
	bundle_adjustment(pts3d,pts2d,K,se3_quat);
      T_c_r_estimated_=SE3(se3_quat.rotation(),se3_quat.translation());
    }
}

bool VisualOdometry::checkEstimatedPose()
{
    // check if the estimated pose is good
    if ( num_inliers_ < min_inliers_ )
    {
        cout<<"reject because inlier is too small: "<<num_inliers_<<endl;
        return false;
    }
    // if the motion is too large, it is probably wrong
    Sophus::Vector6d d = T_c_r_estimated_.log();
    if ( d.norm() > max_motion_norm)
    {
        cout<<"reject because motion is too large: "<<d.norm()<<endl;
        return false;
    }
    return true;
}

bool VisualOdometry::checkKeyFrame()
{
    Sophus::Vector6d d = T_c_r_estimated_.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}

void VisualOdometry::addKeyFrame()
{
    cout<<"adding a key-frame"<<endl;
    map_->insertKeyFrame ( curr_ );
}

void VisualOdometry::bundle_adjustment(const std::vector<cv::Point3f> pt3d, 
	const std::vector<cv::Point2f> pt2d, const Mat& K, g2o::SE3Quat &se3_quat)
{
    //step1
    typedef g2o::BlockSolver_6_3 Block;
    Block::LinearSolverType*linearSolver=new g2o::LinearSolverCSparse<Block::PoseMatrixType>();
    Block*solver_ptr=new Block(linearSolver);
    g2o::OptimizationAlgorithmLevenberg*solver=new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    
    //vertex
    g2o::VertexSE3Expmap*pose=new g2o::VertexSE3Expmap();
    pose->setId(0);
    pose->setEstimate(se3_quat);
    optimizer.addVertex(pose);
    
    int index=1;
    for(const cv::Point3f p:pt3d)
    {
      g2o::VertexSBAPointXYZ*point=new g2o::VertexSBAPointXYZ();
      point->setId(index++);
      point->setEstimate(Eigen::Vector3d(p.x,p.y,p.z));
      point->setMarginalized(true);
      optimizer.addVertex(point);
    }
    
    //camera intrinsics
    g2o::CameraParameters*camera=new g2o::CameraParameters(
      K.at<double>(0,0),Eigen::Vector2d(K.at<double>(0,2),K.at<double>(1,2)),0
    );
    camera->setId(0);
    optimizer.addParameter(camera);
    
    //edges
    index=1;
    for(const cv::Point2f p:pt2d)
    {
      g2o::EdgeProjectXYZ2UV*edge=new g2o::EdgeProjectXYZ2UV();
      edge->setId(index);
      edge->setVertex(0,dynamic_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(index)));
      edge->setVertex(1,pose);
      edge->setMeasurement(Eigen::Vector2d(p.x,p.y));
      edge->setParameterId(0,0);
      edge->setInformation(Eigen::Matrix2d::Identity());
      optimizer.addEdge(edge);
      index++;
    }
    
    boost::timer timer;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    se3_quat=pose->estimate();
    cout<<"optimization of g2o costs time: "<<timer.elapsed()<<endl;
}

void VisualOdometry::bundle_adjustment_pose_only(const vector< cv::Point3f > pt3d, const vector< cv::Point2f > pt2d, const Mat& K, g2o::SE3Quat& se3_quat)
{
    //step1
    typedef g2o::BlockSolver_6_3 Block;
    Block::LinearSolverType*linearSolver=new g2o::LinearSolverCSparse<Block::PoseMatrixType>();
    Block*solver_ptr=new Block(linearSolver);
    g2o::OptimizationAlgorithmLevenberg*solver=new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    
    //vertex
    g2o::VertexSE3Expmap*pose=new g2o::VertexSE3Expmap();
    pose->setId(0);
    pose->setEstimate(se3_quat);
    optimizer.addVertex(pose);

    //edges
    for(int i=0;i<pt2d.size();i++)
    {
      myslam::EdgeProjectXYZ2UVPoseOnly*edge=new myslam::EdgeProjectXYZ2UVPoseOnly();
      edge->setId(i);
      edge->setVertex(0,pose);
      edge->setMeasurement(Eigen::Vector2d(pt2d[i].x,pt2d[i].y));
      edge->setParameterId(0,0);
      edge->setInformation(Eigen::Matrix2d::Identity());
      edge->point_=Vector3d(pt3d[i].x,pt3d[i].y,pt3d[i].z);
      edge->camera_=curr_->camera_;
      optimizer.addEdge(edge);
    }
    
    boost::timer timer;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    se3_quat=pose->estimate();
    cout<<"optimization of pose only g2o costs time: "<<timer.elapsed()<<endl;
}


}