// -------------- test the visual odometry -------------
#include <fstream>
#include <boost/timer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp> 

#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include "myslam/vo2.h"

int main ( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"usage: run_vo parameter_file"<<endl;
        return 1;
    }

    myslam::Config::setParameterFile ( argv[1] );
    //myslam::VisualOdometry::Ptr vo ( new myslam::VisualOdometry );
    myslam::VO2::Ptr vo ( new myslam::VO2 );
    //add rgb and depth files
    string dataset_dir = myslam::Config::get<string> ( "dataset_dir" );
    cout<<"dataset: "<<dataset_dir<<endl;
    ifstream fin ( dataset_dir+"/associate.txt" );
    if ( !fin )
    {
        cout<<"please generate the associate file called associate.txt!"<<endl;
        return 1;
    }
    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    while ( !fin.eof() )
    {
        string rgb_time, rgb_file, depth_time, depth_file;
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
        rgb_times.push_back ( atof ( rgb_time.c_str() ) );
        depth_times.push_back ( atof ( depth_time.c_str() ) );
        rgb_files.push_back ( dataset_dir+"/"+rgb_file );
        depth_files.push_back ( dataset_dir+"/"+depth_file );

        if ( fin.good() == false )
            break;
    }
    cout<<"read total "<<rgb_files.size() <<" entries"<<endl;
    
    //add groundtruth
    vector<SE3>groundtruths;
    ifstream fin2(dataset_dir+"/rgb_groundtruth.txt");
    if ( !fin2 )
    {
        cout<<"please generate the associate file called rgb_groundtruth.txt!"<<endl;
        return 1;
    }
    while ( !fin2.eof() )
    {
        string tx, ty, tz, qx,qy,qz,qw,temp;
        fin2>>temp>>temp>>temp>>tx>>ty>>tz>>qx>>qy>>qz>>qw;
	groundtruths.push_back(
	  SE3(
	    Eigen::Quaterniond(atof(qw.c_str()),atof(qx.c_str()),atof(qy.c_str()),atof(qz.c_str())),
	      Vector3d(atof(tx.c_str()),atof(ty.c_str()),atof(tz.c_str()))
	  )
	);
	
        if ( fin2.good() == false )
            break;
    }
    SE3 first_ground_truth=groundtruths[0];
    first_ground_truth=first_ground_truth.inverse();
    vector<cv::Affine3d> ground_traj_elements;
    for(int i=0;i<groundtruths.size();i++)
    {
      groundtruths[i]=first_ground_truth*groundtruths[i];
      
      //show the groundtruth
      SE3 ground=groundtruths[i];
      ground_traj_elements.push_back(
	cv::Affine3d(
	    cv::Affine3d::Mat3( 
		ground.rotation_matrix()(0,0), ground.rotation_matrix()(0,1), ground.rotation_matrix()(0,2),
		ground.rotation_matrix()(1,0), ground.rotation_matrix()(1,1), ground.rotation_matrix()(1,2),
		ground.rotation_matrix()(2,0), ground.rotation_matrix()(2,1), ground.rotation_matrix()(2,2)
	    ), 
	    cv::Affine3d::Vec3(
		ground.translation()(0,0), ground.translation()(1,0), ground.translation()(2,0)
	    )
	  )
      );
    }
    cv::viz::WTrajectory ground_traj(ground_traj_elements);

    // visualization
    cv::viz::Viz3d vis("Visual Odometry");
    cv::Point3d cam_pos( 0, -1.0, -1.0 ), cam_focal_point(0,0,0), cam_y_dir(0,1,0);
    cv::Affine3d cam_pose = cv::viz::makeCameraPose( cam_pos, cam_focal_point, cam_y_dir );
    vis.setViewerPose( cam_pose );
    
    cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5),groundtruth_coor(0.7);
    world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
    groundtruth_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
    camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
    vis.showWidget( "World", world_coor );
    vis.showWidget( "Groundtruth", groundtruth_coor );
    vis.showWidget( "Camera", camera_coor );
    vis.showWidget("GroundTraj",ground_traj);
    
    
    
    
    //run vo loop
    myslam::Camera::Ptr camera ( new myslam::Camera );
    for ( int i=0; i<rgb_files.size(); i++ )
    {
        Mat color = cv::imread ( rgb_files[i] );
        Mat depth = cv::imread ( depth_files[i], -1 );
        if ( color.data==nullptr || depth.data==nullptr )
            break;
        myslam::Frame::Ptr pFrame = myslam::Frame::createFrame();
        pFrame->camera_ = camera;
        pFrame->color_ = color;
        pFrame->depth_ = depth;
        pFrame->time_stamp_ = rgb_times[i];

        boost::timer timer;
        vo->addFrame ( pFrame );
        cout<<"VO costs time: "<<timer.elapsed()<<endl;
        
        if ( vo->state_ == myslam::VisualOdometry::LOST )
            break;
        SE3 Tcw = pFrame->T_c_w_.inverse();
	
	SE3 ground=groundtruths[i];
	
	cout<<"Tcw:"<<endl<<Tcw<<endl;
	cout<<"ground:"<<endl<<ground<<endl;
	
        // show the map and the camera pose 
        cv::Affine3d M(
            cv::Affine3d::Mat3( 
                Tcw.rotation_matrix()(0,0), Tcw.rotation_matrix()(0,1), Tcw.rotation_matrix()(0,2),
                Tcw.rotation_matrix()(1,0), Tcw.rotation_matrix()(1,1), Tcw.rotation_matrix()(1,2),
                Tcw.rotation_matrix()(2,0), Tcw.rotation_matrix()(2,1), Tcw.rotation_matrix()(2,2)
            ), 
            cv::Affine3d::Vec3(
                Tcw.translation()(0,0), Tcw.translation()(1,0), Tcw.translation()(2,0)
            )
        );
	
	//show the groundtruth
	cv::Affine3d M_ground(
            cv::Affine3d::Mat3( 
                ground.rotation_matrix()(0,0), ground.rotation_matrix()(0,1), ground.rotation_matrix()(0,2),
                ground.rotation_matrix()(1,0), ground.rotation_matrix()(1,1), ground.rotation_matrix()(1,2),
                ground.rotation_matrix()(2,0), ground.rotation_matrix()(2,1), ground.rotation_matrix()(2,2)
            ), 
            cv::Affine3d::Vec3(
                ground.translation()(0,0), ground.translation()(1,0), ground.translation()(2,0)
            )
        );
        
        cv::imshow("image", color );
        cv::waitKey(1);
        vis.setWidgetPose( "Camera", M);
	vis.setWidgetPose("Groundtruth",M_ground);
        vis.spinOnce(1, false);
    }

    return 0;
}
