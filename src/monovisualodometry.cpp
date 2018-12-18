/*
 * Copyright 2018 <copyright holder> <email>
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * this file was made by tianxiaochun.
 * if any question ,touch me by 13051526769@163.com
 * 2018,12,17
 */



#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>

#include "myslam/config.h"
#include "myslam/g2o_types.h"
#include "myslam/monovisualodometry.h"
namespace myslam {
MONOVisualOdometry::MONOVisualOdometry():
    state_ ( INITIALIZING ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 ), matcher_flann_ ( new cv::flann::LshIndexParams ( 5,10,2 ) )
{
    num_of_features_    = Config::get<int> ( "number_of_features" );
    scale_factor_       = Config::get<double> ( "scale_factor" );
    level_pyramid_      = Config::get<int> ( "level_pyramid" );
    match_ratio_        = Config::get<float> ( "match_ratio" );
    max_num_lost_       = Config::get<float> ( "max_num_lost" );
    min_inliers_        = Config::get<int> ( "min_inliers" );
    key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
    key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
    map_point_erase_ratio_ = Config::get<double> ( "map_point_erase_ratio" );
    orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );
    num_frame=0;
  
}
MONOVisualOdometry::~MONOVisualOdometry()
{

}

bool MONOVisualOdometry::addFrame(Frame::Ptr frame)
{
  num_frame++;
  switch(state_)
  {
    case INITIALIZING:
    {
      state_=OK;
      curr_=ref_=frame;
      extractKeyPoints();
      computeDescriptors();
      addKeyFrame();
      break;
    }
    case OK:
    {
      curr_=frame;
      curr_->T_c_w_=ref_->T_c_w_;
      extractKeyPoints();
      computeDescriptors();
      featureMatching();
      poseEstimationPnP();
      if(checkEstimatedPose())
      {
	curr_->T_c_w_=T_c_w_estimated_;
	optimizeMap();
	num_lost_=0;
	if(checkKeyFrame())
	{
	  addKeyFrame();
	}
      }
      else
      {
	num_lost_++;
	if(num_lost_>max_num_lost_)
	{
	  state_=LOST;
	}
	return false;
      }
      break;
    }
    case LOST:
    {
      break;
    }
    default:
        cout<<"vo has lost."<<endl;
        break;
  }
  return true;
}
void MONOVisualOdometry::extractKeyPoints()
{
    boost::timer timer;
    orb_->detect ( curr_->color_, keypoints_curr_ );
    cout<<"extract keypoints cost time: "<<timer.elapsed() <<endl;
}
void MONOVisualOdometry::computeDescriptors()
{
  orb_->compute(curr_->color_,keypoints_curr_,descriptors_curr_);
}

void MONOVisualOdometry::featureMatching()
{
  boost::timer timer;
  vector<cv::DMatch> matches;
  Mat desp_map;
  vector<MapPoint::Ptr> candidate;
  for(auto& allpoints:map_->map_points_ )
  {
    MapPoint::Ptr& p=allpoints.second;
    if(curr_->isInFrame(p->pos_))
    {
      p->visible_times_++;
      candidate.push_back(p);
      desp_map.push_back(p->descriptor_);
    }
  }
  matcher_flann_.match(desp_map,descriptors_curr_,matches);
  
  float min_dis=std::min_element(matches.begin(),matches.end(),[](const cv::DMatch& m1,const cv::DMatch& m2)
  {
    return m1.distance<m2.distance;
  }).distance;
  match_3dpts_.clear();
  match_2dkp_index_.clear();
  for(cv::DMatch& m:matches)
  {
    if(m.distance<max <float>(min_dis*match_ratio_,30.0))
    {
      match_3dpts_.push_back(candidate[m.queryIdx]);
      match_2dkp_index_.push_back(m.trainIdx);
    }
  }
    cout<<"good matches: "<<match_3dpts_.size() <<endl;
    cout<<"match cost time: "<<timer.elapsed() <<endl;  
}

void MONOVisualOdometry::poseEstimationPnP()
{
    // construct the 3d 2d observations
    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;

    for ( int index:match_2dkp_index_ )
    {
        pts2d.push_back ( keypoints_curr_[index].pt );
    }
    for ( MapPoint::Ptr pt:match_3dpts_ )
    {
        pts3d.push_back( pt->getPositionCV() );
    }

    Mat K = ( cv::Mat_<double> ( 3,3 ) <<
              ref_->camera_->fx_, 0, ref_->camera_->cx_,
              0, ref_->camera_->fy_, ref_->camera_->cy_,
              0,0,1
            );
    Mat rvec, tvec, inliers;
    cv::solvePnPRansac ( pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
    num_inliers_ = inliers.rows;
    cout<<"pnp inliers: "<<num_inliers_<<endl;
    T_c_w_estimated_ = SE3 (
                           SO3 ( rvec.at<double> ( 0,0 ), rvec.at<double> ( 1,0 ), rvec.at<double> ( 2,0 ) ),
                           Vector3d ( tvec.at<double> ( 0,0 ), tvec.at<double> ( 1,0 ), tvec.at<double> ( 2,0 ) )
                       );

    // using bundle adjustment to optimize the pose
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block* solver_ptr = new Block ( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
        T_c_w_estimated_.rotation_matrix(), T_c_w_estimated_.translation()
    ));
    optimizer.addVertex ( pose );

    // edges
    for ( int i=0; i<inliers.rows; i++ )
    {
        int index = inliers.at<int> ( i,0 );
        // 3D -> 2D projection
        EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
        edge->setId ( i );
        edge->setVertex ( 0, pose );
        edge->camera_ = curr_->camera_.get();
        edge->point_ = Vector3d ( pts3d[index].x, pts3d[index].y, pts3d[index].z );
        edge->setMeasurement ( Vector2d ( pts2d[index].x, pts2d[index].y ) );
        edge->setInformation ( Eigen::Matrix2d::Identity() );
        optimizer.addEdge ( edge );
        // set the inlier map points 
        match_3dpts_[index]->matched_times_++;
    }

    optimizer.initializeOptimization();
    optimizer.optimize ( 10 );

    T_c_w_estimated_ = SE3 (
        pose->estimate().rotation(),
        pose->estimate().translation()
    );
    
    cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;
}


bool MONOVisualOdometry::checkEstimatedPose()
{
    // check if the estimated pose is good
    if ( num_inliers_ < min_inliers_ )
    {
        cout<<"reject because inlier is too small: "<<num_inliers_<<endl;
        return false;
    }
    // if the motion is too large, it is probably wrong
    SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    if ( d.norm() > 5.0 )
    {
        cout<<"reject because motion is too large: "<<d.norm() <<endl;
        return false;
    }
    return true;  
}
bool MONOVisualOdometry::checkKeyFrame()
{
    SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}
void MONOVisualOdometry::addKeyFrame()
{
  if(map_->keyframes_.empty())
  {
    for(size_t i=0;i<keypoints_curr_.size();i++)
    {
      double d=curr_->findDepth(keypoints_curr_[i]);
      if(d<0)
	continue;
      Vector3d p_world = ref_->camera_->pixel2world(
	Vector2d(keypoints_curr_[i].pt.x,keypoints_curr_[i].pt.y),curr_->T_c_w_,d
      );
      Vector3d n=p_world-ref_->getCamCenter();
      n.normalize();
      MapPoint::Ptr map_point = MapPoint::createMapPoint(
	p_world,n,descriptors_curr_.row(i).clone(),curr_.get()
      );
      map_->insertMapPoint(map_point);
    }
    
  }
  map_->insertKeyFrame(curr_);
  ref_=curr_;
}
void MONOVisualOdometry::addMapPoints()
{
//add the new map points into map_
  vector<bool> matched(keypoints_curr_.size(),false);
  for(int index:match_2dkp_index_)
    matched[index] =true;
  for(int i=0;i<keypoints_curr_.size();i++)
  {
    if(matched[i]==true)
      continue;
    double d=ref_->findDepth(keypoints_curr_[i]);
    if(d<0)
      continue;
    Vector3d p_world=ref_->camera_->pixel2world(
      Vector2d(keypoints_curr_[i].pt.x,keypoints_curr_[i].pt.y),
      curr_->T_c_w_,d
    );
    Vector3d n=p_world-ref_->getCamCenter();
    n.normalize();
    MapPoint::Ptr map_point=MapPoint::createMapPoint(
            p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
        );
        map_->insertMapPoint( map_point );
  }
}
void MONOVisualOdometry::optimizeMap()
{
  for(auto iter=map_->map_points_.begin();iter !=map_->map_points_;)
  {
    if(curr_->isInFrame(iter->second->pos))
    {
      iter = map_->map_points_.erase(iter);
    }
    float match_ratio =float(iter->second->matched_times_)/iter->second->visible_times_;
    if(match_ratio<map_point_erase_ratio_)
    {
      iter=map_->map_points_.erase(iter);
      continue;
    }
    double angle=getViewAngle(curr_,iter->second);
    if(angle>M_PI/6.)
    {
      iter=map->map_points_.erase(iter);
      continue;
    }
    if(iter->second->good_==false)
    {
      //TODO try trianglate this map point
    }
    iter++;
  }
  if(match_2dkp_index_.size()<100)
    addMapPoints();
  if(map_->map_points_.size()>1000)
  {
    //TODO map is too large,remove someone
    map_point_erase_ratio_+=0.05;
  }
  else
    map_point_erase_ratio_=0.1;
  cout<<"map points: "<<map_->map_points_.size()<<endl;  
}



double MONOVisualOdometry::getViewAngle ( Frame::Ptr frame, MapPoint::Ptr point )
{
    Vector3d n = point->pos_ - frame->getCamCenter();
    n.normalize();
    return acos( n.transpose()*point->norm_ );
}
//增加一部分单目的
void MONOVisualOdometry::pose_estimation_2d2d (
    const std::vector<KeyPoint>& keypoints_1,
    const std::vector<KeyPoint>& keypoints_2,
    const std::vector< DMatch >& matches,
    Mat& R, Mat& t )
{
    // 相机内参,TUM Freiburg2
//     Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    Mat K = ( cv::Mat_<double> ( 3,3 ) <<
              ref_->camera_->fx_, 0, ref_->camera_->cx_,
              0, ref_->camera_->fy_, ref_->camera_->cy_,
              0,0,1
            );
    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for ( int i = 0; i < ( int ) matches.size(); i++ )
    {
        points1.push_back ( keypoints_1[matches[i].queryIdx].pt );
        points2.push_back ( keypoints_2[matches[i].trainIdx].pt );
    }

//     //-- 计算基础矩阵
//     Mat fundamental_matrix;
//     fundamental_matrix = findFundamentalMat ( points1, points2, CV_FM_8POINT );
//     cout<<"fundamental_matrix is "<<endl<< fundamental_matrix<<endl;

    //-- 计算本质矩阵
    Point2d principal_point ( ref_->camera_->cx_, ref_->camera_->cy_ );				//相机主点, TUM dataset标定值
    int focal_length = 521;						//相机焦距, TUM dataset标定值
    Mat essential_matrix;
    essential_matrix = cv::findEssentialMat ( points1, points2, focal_length, principal_point );
    cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

    //-- 计算单应矩阵
//     Mat homography_matrix;
//     homography_matrix = findHomography ( points1, points2, RANSAC, 3 );
//     cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    cv::recoverPose ( essential_matrix, points1, points2, R, t, focal_length, principal_point );
//     cout<<"R is "<<endl<<R<<endl;
//     cout<<"t is "<<endl<<t<<endl;
    T_c_w_estimated_ = SE3 (
                           SO3 ( R ),
                           Vector3d ( t.at<double> ( 0,0 ), t.at<double> ( 1,0 ), t.at<double> ( 2,0 ) )
                       );    
    
}



void MONOVisualOdometry::triangulation(const std::vector< std::allocator >& keypoint_1, const std::vector< std::allocator >& keypoint_2, const std::vector< std::allocator >& matches, const Mat& R, const Mat& t, std::vector< std::allocator >& points)
{
    Mat T1 = (cv::Mat_<float> (3,4) <<
        1,0,0,0,
        0,1,0,0,
        0,0,1,0);
    Mat T2 = (cv::Mat_<float> (3,4) <<
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0)
    );
    
//     Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
        Mat K = ( cv::Mat_<double> ( 3,3 ) <<
              ref_->camera_->fx_, 0, ref_->camera_->cx_,
              0, ref_->camera_->fy_, ref_->camera_->cy_,
              0,0,1
            );
    vector<Point2f> pts_1, pts_2;
    for ( DMatch m:matches )
    {
        // 将像素坐标转换至相机坐标
        pts_1.push_back (  ref_->camera_->pixel2camera( keypoint_1[m.queryIdx].pt, K) );
        pts_2.push_back ( ref_->camera_->pixel2camera( keypoint_2[m.trainIdx].pt, K) );
    }
    
    Mat pts_4d;
    cv::triangulatePoints( T1, T2, pts_1, pts_2, pts_4d );
    
    // 转换成非齐次坐标
    for ( int i=0; i<pts_4d.cols; i++ )
    {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3,0); // 归一化
        cv::Point3d p (
            x.at<float>(0,0), 
            x.at<float>(1,0), 
            x.at<float>(2,0) 
        );
        points.push_back( p );
    }
}



}
