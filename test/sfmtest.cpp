// -------------- test the visual odometry -------------
#include <fstream>
#include <boost/timer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"

using namespace std;
using namespace cv;
using namespace Eigen;
struct SFMFeature
{
    bool state;
    int id;
    vector<pair<int, Vector2d>> observation;
    double position[3];
    double depth;
};

void find_feature_matches(
    const Mat &img_1, const Mat &img_2,
    std::vector<KeyPoint> &keypoints_1,
    std::vector<KeyPoint> &keypoints_2,
    std::vector<DMatch> &matches);

void pose_estimation_2d2d(
    const std::vector<KeyPoint> &keypoints_1,
    const std::vector<KeyPoint> &keypoints_2,
    const std::vector<DMatch> &matches,
    Mat &R, Mat &t);

void triangulation(
    const vector<KeyPoint> &keypoint_1,
    const vector<KeyPoint> &keypoint_2,
    const std::vector<DMatch> &matches,
    const Mat &R, const Mat &t,
    vector<Point3d> &points);
// 像素坐标转相机归一化坐标
Point2f pixel2cam(const Point2d &p, const Mat &K);

int main ( int argc, char** argv )
{
    // if ( argc != 2 )
    // {
    //     cout<<"usage: run_vo parameter_file"<<endl;
    //     return 1;
    // }

    myslam::Config::setParameterFile ( "../config/default.yaml" );
    myslam::VisualOdometry::Ptr vo ( new myslam::VisualOdometry );

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



    // // visualization
    // cv::viz::Viz3d vis ( "Visual Odometry" );
    // cv::viz::WCoordinateSystem world_coor ( 1.0 ), camera_coor ( 0.5 );
    // cv::Point3d cam_pos ( 0, -1.0, -1.0 ), cam_focal_point ( 0,0,0 ), cam_y_dir ( 0,1,0 );
    // cv::Affine3d cam_pose = cv::viz::makeCameraPose ( cam_pos, cam_focal_point, cam_y_dir );
    // vis.setViewerPose ( cam_pose );

    // world_coor.setRenderingProperty ( cv::viz::LINE_WIDTH, 2.0 );
    // camera_coor.setRenderingProperty ( cv::viz::LINE_WIDTH, 1.0 );
    // vis.showWidget ( "World", world_coor );
    // vis.showWidget ( "Camera", camera_coor );

    // cout<<"read total "<<rgb_files.size() <<" entries"<<endl;

    int WINDOWSIZE=10;
    Quaterniond Q[WINDOWSIZE + 1];
    Vector3d T[WINDOWSIZE + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f; //共视点的集合
    


    Mat img_1 = imread(rgb_files[0], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(rgb_files[WINDOWSIZE-1], CV_LOAD_IMAGE_COLOR);
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;

    vector<Point3d> pointsfirst_last;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;
    //-- 估计两张图像间运动
    Mat relative_R, relative_t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, relative_R, relative_t);
    triangulation(keypoints_1,keypoints_2,matches,relative_R,relative_t,pointsfirst_last);


    // int feature_num = WINDOWSIZE;
    // //cout << "set 0 and " << l << " as known " << endl;
    // // have relative_r relative_t
    // // intial two view
    // //确定最后一帧的相对值
    // //设置初始值，ql是确定的帧的，t【-1】当然是给出计算的那个部分，以l帧作为起点0
    // int l=0;
    // Eigen::Quaterniond q[feature_num];
    // q[l].w() = 1;
    // q[l].x() = 0;
    // q[l].y() = 0;
    // q[l].z() = 0;
    // T[l].setZero();
    // q[frame_num - 1] = q[l] * Quaterniond(relative_R);
    // T[frame_num - 1] = relative_T;
    // //cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
    // //cout << "init t_l " << T[l].transpose() << endl;

    // //rotate to cam frame
    // //为啥要定义两个部分的RT
    // Matrix3d c_Rotation[frame_num];
    // Vector3d c_Translation[frame_num];
    // Quaterniond c_Quat[frame_num];
    // double c_rotation[frame_num][4];
    // double c_translation[frame_num][3];
    // Eigen::Matrix<double, 3, 4> Pose[frame_num]; //位姿用于三角化

    // c_Quat[l] = q[l].inverse();
    // c_Rotation[l] = c_Quat[l].toRotationMatrix(); //最后一帧当然是算出来的那个
    // c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
    // Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
    // Pose[l].block<3, 1>(0, 3) = c_Translation[l];

    // c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
    // c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
    // c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
    // Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
    // Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];

    // for (int i = 0; i < WINDOWSIZE-1; i++)
    // {
    //     // cout<<"****** loop "<<i<<" ******"<<rgb_files[i] <<endl;
    //     // Mat color = cv::imread ( rgb_files[i] );
    //     // solve pnp跳过前面那些不用的帧
    //     if (i > l)
    //     {
    //         //就是用上一帧的作为初值去求解pnp
    //         Matrix3d R_initial = c_Rotation[i - 1];
    //         Vector3d P_initial = c_Translation[i - 1];
    //         if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
    //             return false;
    //         c_Rotation[i] = R_initial;
    //         c_Translation[i] = P_initial;
    //         c_Quat[i] = c_Rotation[i];
    //         Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
    //         Pose[i].block<3, 1>(0, 3) = c_Translation[i];
    //     }

    //     // triangulate point based on the solve pnp result
    //     //前面条件不满足，就是首先进来当i=l时第i帧和最后一帧之间的三角化，得到了这两帧之间的共视点
    //     //然后重复进行从i+1帧到最后一帧之间的三角化，sfm里对应的部分的位姿
    //     //经过这一步，可以在一开始定义的那些在l和最后一帧之间的共视点上加上深度
    //     triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
    // }
    // for (int i = l + 1; i < frame_num - 1; i++)
    //     triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
    return 0;
}


void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches)
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = match[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if (match[i].distance <= max(2 * min_dist, 30.0))
        {
            matches.push_back(match[i]);
        }
    }
}

void pose_estimation_2d2d(
    const std::vector<KeyPoint> &keypoints_1,
    const std::vector<KeyPoint> &keypoints_2,
    const std::vector<DMatch> &matches,
    Mat &R, Mat &t)
{
    // 相机内参,TUM Freiburg2
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for (int i = 0; i < (int)matches.size(); i++)
    {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    //-- 计算基础矩阵
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
    cout << "fundamental_matrix is " << endl
         << fundamental_matrix << endl;

    //-- 计算本质矩阵
    Point2d principal_point(325.1, 249.7); //相机主点, TUM dataset标定值
    int focal_length = 521;                //相机焦距, TUM dataset标定值
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    cout << "essential_matrix is " << endl
         << essential_matrix << endl;

    //-- 计算单应矩阵
    Mat homography_matrix;
    homography_matrix = findHomography(points1, points2, RANSAC, 3);
    cout << "homography_matrix is " << endl
         << homography_matrix << endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    cout << "R is " << endl
         << R << endl;
    cout << "t is " << endl
         << t << endl;
}

void triangulation(
    const vector<KeyPoint> &keypoint_1,
    const vector<KeyPoint> &keypoint_2,
    const std::vector<DMatch> &matches,
    const Mat &R, const Mat &t,
    vector<Point3d> &points)
{
    Mat T1 = (Mat_<float>(3, 4) << 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0);
    Mat T2 = (Mat_<float>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
              R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
              R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point2f> pts_1, pts_2;
    for (DMatch m : matches)
    {
        // 将像素坐标转换至相机坐标
        pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
        pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
    }

    Mat pts_4d;
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    // 转换成非齐次坐标
    for (int i = 0; i < pts_4d.cols; i++)
    {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0); // 归一化
        Point3d p(
            x.at<float>(0, 0),
            x.at<float>(1, 0),
            x.at<float>(2, 0));
        points.push_back(p);
    }
}

Point2f pixel2cam(const Point2d &p, const Mat &K)
{
    return Point2f(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}