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

#ifndef MONOVISUALODOMETRY_H
#define MONOVISUALODOMETRY_H
#include "myslam/common_include.h"
#include "myslam/map.h"

#include <opencv2/features2d/features2d.hpp>

namespace myslam 
{
class MONOVisualOdometry
{
    typedef shared_ptr<VisualOdometry> Ptr;
    enum VOState {
        INITIALIZING=-1,
        OK=0,
        LOST
    };
    
    VOState     state_;     // current VO status
    Map::Ptr    map_;       // map with all frames and map points
    vector<cv::Point3d> points;
    Frame::Ptr  ref_;       // reference key-frame 
    Frame::Ptr  curr_;      // current frame 
    
    cv::Ptr<cv::ORB> orb_;  // orb detector and computer 
    vector<cv::KeyPoint>    keypoints_curr_;    // keypoints in current frame
    Mat                     descriptors_curr_;  // descriptor in current frame 
    vector<cv::KeyPoint>    keypoints_ref_;    // keypoints in current frame
    Mat                     descriptors_ref_;  // descriptor in current frame 
    
    cv::FlannBasedMatcher   matcher_flann_;     // flann matcher
    vector<MapPoint::Ptr>   match_3dpts_;       // matched 3d points 
    vector<int>             match_2dkp_index_;  // matched 2d pixels (index of kp_curr)
   
    SE3 T_c_w_estimated_;    // the estimated pose of current frame 
    int num_inliers_;        // number of inlier features in icp
    int num_lost_;           // number of lost times
    
    int num_frame;
    // parameters 
    int num_of_features_;   // number of features
    double scale_factor_;   // scale in image pyramid
    int level_pyramid_;     // number of pyramid levels
    float match_ratio_;     // ratio for selecting  good matches
    int max_num_lost_;      // max number of continuous lost times
    int min_inliers_;       // minimum inliers
    double key_frame_min_rot;   // minimal rotation of two key-frames
    double key_frame_min_trans; // minimal translation of two key-frames
    double  map_point_erase_ratio_; // remove map point ratio
public: // functions 
MONOVisualOdometry();
~MONOVisualOdometry();
bool addFrame( Frame::Ptr frame );      // add a new frame 


protected:  
    // inner operation 
    void extractKeyPoints();
    void computeDescriptors(); 
    void featureMatching();
    void poseEstimationPnP(); 
    void optimizeMap();
    
    void addKeyFrame();
    void addMapPoints();
    bool checkEstimatedPose(); 
    bool checkKeyFrame();
    void addFirstFrame()  ;  
    void updateFirstFrame();
    void featureMatchingInit();
    void pose_estimation_2d2d (
    const std::vector<KeyPoint>& keypoints_1,
    const std::vector<KeyPoint>& keypoints_2,
    const std::vector< DMatch >& matches,
    Mat& R, Mat& t );
    void triangulation (
	const vector<KeyPoint>& keypoint_1,
	const vector<KeyPoint>& keypoint_2,
	const std::vector< DMatch >& matches,
	const Mat& R, const Mat& t,
	vector<Point3d>& points
    );
    
    double getViewAngle( Frame::Ptr frame, MapPoint::Ptr point );  
};
cv::Point2f pixel2cam ( const Point2d& p, const Mat& K )
{
    return cv::Point2f
    (
        ( p.x - K.at<double>(0,2) ) / K.at<double>(0,0), 
        ( p.y - K.at<double>(1,2) ) / K.at<double>(1,1) 
    );
}
}
#endif // MONOVISUALODOMETRY_H
