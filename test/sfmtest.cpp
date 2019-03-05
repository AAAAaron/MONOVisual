// -------------- test the visual odometry -------------
#ifndef SFMTEST_INCLUDE_H
#define SFMTEST_INCLUDE_H
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <fstream>
#include <Eigen/Dense>
// #include "myslam/common_include.h"
#include <boost/timer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <map>
#include <opencv2/video/tracking.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"

using namespace std;
using namespace cv;
using namespace Eigen;
struct SFMFeature
{
    bool state;
    int id;
    vector<pair<int, Vector2d>> observation;//这个就可以表示在每帧听到的位置
    double position[3];
    double depth;
};
struct FrameInfo
{
  int id;
  Mat img;
  vector<KeyPoint> frameKeypoints;
  Mat FrameDescriptors;
  vector<Point3d> depth;
};
// 像素坐标转相机归一化坐标
Point2f pixel2cam(const Point2d &p, const Mat &K);

struct ReprojectionError3D
{
    ReprojectionError3D(double observed_u, double observed_v)
        : observed_u(observed_u), observed_v(observed_v)
    {
    }

    template <typename T>
    bool operator()(const T *const camera_R, const T *const camera_T, const T *point, T *residuals) const
    {
        T p[3];
        ceres::QuaternionRotatePoint(camera_R, point, p);
        p[0] += camera_T[0];
        p[1] += camera_T[1];
        p[2] += camera_T[2];
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        residuals[0] = xp - T(observed_u);
        residuals[1] = yp - T(observed_v);
        return true;
    }

    static ceres::CostFunction *Create(const double observed_x,
                                       const double observed_y)
    {
        return (new ceres::AutoDiffCostFunction<
                ReprojectionError3D, 2, 4, 3, 3>(
            new ReprojectionError3D(observed_x, observed_y)));
    }

    double observed_u;
    double observed_v;
};
template <class T> void reduceVector(vector<T> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}
void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d);
//找到两帧之间的共视点
void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f);
bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l,int WINDOW_SIZE,vector<SFMFeature> &sfm_f);

void rejectWithF(vector<cv::Point2f> &forw_pts, vector<cv::Point2f> &cur_pts)
{

        vector<uchar> status;
        //Calculates a fundamental matrix from the corresponding points in two images.
        //根据两队点算F,以及status再次筛除一些点
        cv::findFundamentalMat(forw_pts, cur_pts, cv::FM_RANSAC, 3.0, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        // ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        // ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    
}

vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r,vector<SFMFeature> &sfm_f);
//5帧法恢复rt，是从前面找到的特征点
bool solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &Rotation, Vector3d &Translation)
{
    if (corres.size() >= 15)
    {
      int ptCount = (int)corres.size();
      Mat p1(ptCount, 2, CV_32F);
      Mat p2(ptCount, 2, CV_32F);

      

      // 把Keypoint转换为Mat
      for (int i=0; i<ptCount; i++)
      {

	  p1.at<float>(i, 0) = corres[i].first(0);
	  p1.at<float>(i, 1) = corres[i].first(1);

	  p2.at<float>(i, 0) = corres[i].second(0);
	  p2.at<float>(i, 1) = corres[i].second(1);
// 	  cout<<p1.at<float>(i, 0)<<"----"<<p1.at<float>(i, 1)<<endl;
// 	  cout<<p2.at<float>(i, 0)<<"----"<<p2.at<float>(i, 1)<<endl;	  
      }      

        cv::Mat mask;

        //找基础质矩阵
        cv::Mat E = cv::findFundamentalMat(p1, p2, cv::FM_RANSAC, 3., 0.99, mask);

        cv::Mat cameraMatrix =  ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );

        cv::Mat rot, trans;
        //判断相位符合的点

        int inlier_cnt = cv::recoverPose(E, p1, p2, cameraMatrix, rot, trans, mask);
        cout << "inlier_cnt " << inlier_cnt << endl;

        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        for (int i = 0; i < 3; i++)
        {   
            T(i) = trans.at<double>(i, 0);
            for (int j = 0; j < 3; j++)
                R(i, j) = rot.at<double>(i, j);
        }

        Rotation = R.transpose();
        Translation = -R.transpose() * T;
        if(inlier_cnt > 12)//如果点合符要求，就返回
            return true;
        else
            return false;
    }
    return false;
}



bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,vector<SFMFeature> &sfm_f);


int WINDOWSIZE=15;
int main(int argc, char **argv)
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
    ifstream fin(dataset_dir + "/associate.txt");

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



    // visualization
    cv::viz::Viz3d vis ( "Visual Odometry" );
    cv::viz::WCoordinateSystem world_coor ( 1.0 ), camera_coor ( 0.5 );
    cv::Point3d cam_pos ( 0, -1.0, -1.0 ), cam_focal_point ( 0,0,0 ), cam_y_dir ( 0,1,0 );
    cv::Affine3d cam_pose = cv::viz::makeCameraPose ( cam_pos, cam_focal_point, cam_y_dir );
    vis.setViewerPose ( cam_pose );

    world_coor.setRenderingProperty ( cv::viz::LINE_WIDTH, 2.0 );
    camera_coor.setRenderingProperty ( cv::viz::LINE_WIDTH, 1.0 );
    vis.showWidget ( "World", world_coor );
    vis.showWidget ( "Camera", camera_coor );

    cout<<"read total "<<rgb_files.size() <<" entries"<<endl;


    Vector3d T[WINDOWSIZE + 1];
    map<int, Vector3d> sfm_tracked_points;

    
    vector< cv::Point2f > keypoints;      // 因为要删除跟踪失败的点，使用list
    vector< SFMFeature > sfm_f;
    cv::Mat color, depth, last_color;
   
    for(int index=0;index<WINDOWSIZE;index++)
    {
        color = imread(rgb_files[index], CV_LOAD_IMAGE_COLOR);
	cout<<"read..."<<endl;
        if (index == 0 )
        {
            // 对第一帧提取FAST特征点
            vector<cv::KeyPoint> kps;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect( color, kps );
	    int first_count=0;
            for ( auto kp:kps )
	    {
                keypoints.push_back( kp.pt );
		SFMFeature tmpsfm;
		tmpsfm.state=false;
		tmpsfm.id=first_count;
		tmpsfm.observation.push_back(make_pair(index, Eigen::Vector2d{kp.pt.x, kp.pt.y}));
		first_count++;
		sfm_f.push_back(tmpsfm);
	    }
            last_color = color;
            continue;
        }
        if ( color.data==nullptr)
            continue;
        // 对其他帧用LK跟踪特征点
        vector<cv::Point2f> next_keypoints; 
        vector<cv::Point2f> prev_keypoints;
        for ( auto kp:keypoints )
            prev_keypoints.push_back(kp);

        vector<unsigned char> status;
        vector<float> error; 
//         chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        cv::calcOpticalFlowPyrLK( last_color, color, prev_keypoints, next_keypoints, status, error );

        //         chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        //         chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
        //         cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
        // 把跟丢的点删掉

        int keypoint_index = 0;
        for ( auto kp:next_keypoints )
	{
	  sfm_f[keypoint_index].observation.push_back(make_pair(index, Eigen::Vector2d{kp.x, kp.y}));
      ++keypoint_index;
    }
        reduceVector(keypoints,status);
	reduceVector(sfm_f,status);


    //Calculates a fundamental matrix from the corresponding points in two images.
    //根据两队点算F,以及status再次筛除一些点
    cv::findFundamentalMat(prev_keypoints, next_keypoints, cv::FM_RANSAC, 1.0, 0.99, status);
    reduceVector(keypoints, status);
    reduceVector(sfm_f, status);

    cout << "tracked keypoints: " << keypoints.size() << endl;
    if (keypoints.size() == 0)
    {
        cout << "all keypoints are lost." << endl;
        break; 
        }
        // 画出 keypoints
//         cv::Mat img_show = color.clone();
//         for ( auto kp:keypoints )
//             cv::circle(img_show, kp, 10, cv::Scalar(0, 240, 0), 1);
//         cv::imshow("corners", img_show);
//         cv::waitKey(0);
        last_color = color;
	
    }        



    int frame_num = WINDOWSIZE;
    // //cout << "set 0 and " << l << " as known " << endl;
    // // have relative_r relative_t
    // // intial two view
    // //确定最后一帧的相对值
    // //设置初始值，ql是确定的帧的，t【-1】当然是给出计算的那个部分，以l帧作为起点0
    int l=0;
    Eigen::Quaterniond q[frame_num];
    q[l].w() = 1;
    q[l].x() = 0;
    q[l].y() = 0;
    q[l].z() = 0;
    T[l].setZero();
    Matrix3d relative_R;
    Vector3d relative_T;

    if (!relativePose(relative_R, relative_T, l,WINDOWSIZE,sfm_f))
    {
        printf("Not enough features or parallax; Move device around");
        return false;
    }    

    q[frame_num - 1] = q[l] * Quaterniond(relative_R);
    T[frame_num - 1] = relative_T;
    // //cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
    // //cout << "init t_l " << T[l].transpose() << endl;

    // //rotate to cam frame
    // //为啥要定义两个部分的RT
    Matrix3d c_Rotation[frame_num];
    Vector3d c_Translation[frame_num];
    Quaterniond c_Quat[frame_num];
    double c_rotation[frame_num][4];
    double c_translation[frame_num][3];
    Eigen::Matrix<double, 3, 4> Pose[frame_num]; //位姿用于三角化
//第一帧的值
    c_Quat[l] = q[l].inverse();//旋转是什么意思？？？
    c_Rotation[l] = c_Quat[l].toRotationMatrix(); //最后一帧当然是算出来的那个
    c_Translation[l] = -1 * (c_Rotation[l] * T[l]);//意思是以最后一帧载体系处理吗
    Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
    Pose[l].block<3, 1>(0, 3) = c_Translation[l];
//最后一帧的值
    c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
    c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
    c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
    Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
    Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];

    for (int i = l; i < WINDOWSIZE-1; i++)
    {
        // cout<<"****** loop "<<i<<" ******"<<rgb_files[i] <<endl;
        // Mat color = cv::imread ( rgb_files[i] );
        // solve pnp跳过前面那些不用的帧
	if (i > l)
        {
            //就是用上一帧的作为初值去求解pnp
            Matrix3d R_initial = c_Rotation[i - 1];
            Vector3d P_initial = c_Translation[i - 1];
            if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
                return false;
            c_Rotation[i] = R_initial;
            c_Translation[i] = P_initial;
            c_Quat[i] = c_Rotation[i];
            Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
            Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        }

    //     // triangulate point based on the solve pnp result
    //     //前面条件不满足，就是首先进来当i=l时第i帧和最后一帧之间的三角化，得到了这两帧之间的共视点
    //     //然后重复进行从i+1帧到最后一帧之间的三角化，sfm里对应的部分的位姿
    //     //经过这一步，可以在一开始定义的那些在l和最后一帧之间的共视点上加上深度
        triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
      
    }	
    for (int i = l + 1; i < frame_num - 1; i++)
    {  triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
    }
    //4: solve pnp l-1; triangulate l-1 ----- l
    //             l-2              l-2 ----- l
    //三角化所有窗口后部分的帧和地l帧
    //l帧已经被认定为0了，那么其他帧自然就是和他确定
    for (int i = l - 1; i >= 0; i--)
    {
        //solve pnp
        Matrix3d R_initial = c_Rotation[i + 1];
        Vector3d P_initial = c_Translation[i + 1];
        if (!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
            return false;
        c_Rotation[i] = R_initial;
        c_Translation[i] = P_initial;
        c_Quat[i] = c_Rotation[i];
        Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
        Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        //triangulate
        triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
    }
    //5: triangulate all other points
    for (int j = 0; j < sfm_f.size(); j++)
    {
        if (sfm_f[j].state == true)
            continue;
        if ((int)sfm_f[j].observation.size() >= 2)
        {
            Vector2d point0, point1;
            int frame_0 = sfm_f[j].observation[0].first;
            point0 = sfm_f[j].observation[0].second;
            int frame_1 = sfm_f[j].observation.back().first;
            point1 = sfm_f[j].observation.back().second;
            Vector3d point_3d;
            triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
            sfm_f[j].state = true;
            sfm_f[j].position[0] = point_3d(0);
            sfm_f[j].position[1] = point_3d(1);
            sfm_f[j].position[2] = point_3d(2);
            //cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
        }


    }
    //full BA
    //full BA
    //全优化方式
    ceres::Problem problem;
    ceres::LocalParameterization *local_parameterization = new ceres::QuaternionParameterization();
    //cout << " begin full BA " << endl;
    for (int i = 0; i < frame_num; i++)
    {
        //double array for ceres
        c_translation[i][0] = c_Translation[i].x();
        c_translation[i][1] = c_Translation[i].y();
        c_translation[i][2] = c_Translation[i].z();
        c_rotation[i][0] = c_Quat[i].w();
        c_rotation[i][1] = c_Quat[i].x();
        c_rotation[i][2] = c_Quat[i].y();
        c_rotation[i][3] = c_Quat[i].z();
        problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
        problem.AddParameterBlock(c_translation[i], 3);
        if (i == l)
        {
            problem.SetParameterBlockConstant(c_rotation[i]);
        }
        if (i == l || i == frame_num - 1)
        {
            problem.SetParameterBlockConstant(c_translation[i]);
        }
    }

    for (int i = 0; i < sfm_f.size(); i++)
    {
        if (sfm_f[i].state != true)
            continue;
        for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
        {
            int l = sfm_f[i].observation[j].first;
            ceres::CostFunction *cost_function = ReprojectionError3D::Create(
                sfm_f[i].observation[j].second.x(),
                sfm_f[i].observation[j].second.y());

            problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l],
                                     sfm_f[i].position);
        }
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.minimizer_progress_to_stdout = true;
    options.max_solver_time_in_seconds = 0.2;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //std::cout << summary.BriefReport() << "\n";
    if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
    {
        cout << "vision only BA converge" << endl;
    }
    else
    {
        cout << "vision only BA not converge " << endl;
        // return false;
    }
    for (int i = 0; i < frame_num; i++)
    {
        q[i].w() = c_rotation[i][0];
        q[i].x() = c_rotation[i][1];
        q[i].y() = c_rotation[i][2];
        q[i].z() = c_rotation[i][3];
        q[i] = q[i].inverse();
        // cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
        cout<<"final q  "<<endl;
        cout<<q[i].toRotationMatrix()<<endl;

        T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
        cout << "final t  "<<endl;
        cout << T[i](0) << "  " << T[i](1) << "  " << T[i](2) << endl;
    }

    for (int i = 0; i < (int)sfm_f.size(); i++)
    {
        if (sfm_f[i].state)
            sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
    }
    // return true;


    for(int i=0;i<frame_num;i++)
    {
      
    //visual
      Matrix3d newtmp=q[i].toRotationMatrix();
      cv::Affine3d M(
          cv::Affine3d::Mat3(
              newtmp(0, 0), newtmp(0, 1), newtmp(0, 2),
              newtmp(1, 0), newtmp(1, 1), newtmp(1, 2),
              newtmp(2, 0), newtmp(2, 1), newtmp(2, 2)),
          cv::Affine3d::Vec3(
              T[i](0), T[i](1), T[i](2)));

      cv::Mat img_show = imread(rgb_files[i], CV_LOAD_IMAGE_COLOR);
      vector<KeyPoint> kp;
      for (auto sfm : sfm_f)
      {
          for (unsigned sfmindex = 0; sfmindex < sfm.observation.size(); sfmindex++)
          {
              if (sfm.observation[sfmindex].first == i)
              {
                  cv::circle(img_show, Point(sfm.observation[sfmindex].second(0), sfm.observation[sfmindex].second(1)), 10, cv::Scalar(0, 240, 0), 1);
              }
          }
	    }
        // cv::imshow("corners", img_show);     
        cv::imshow ( "image", img_show );
     
        cv::waitKey ( 0);
        vis.setWidgetPose ( "Camera", M );
        vis.spinOnce ( 10, false );
        cout<<endl;
    }

    return 0;
}
void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}
//找到两帧之间的共视点
void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1);
	for (int j = 0; j < sfm_f.size(); j++)
	{
		if (sfm_f[j].state == true)
			continue;
		bool has_0 = false, has_1 = false;
		Vector2d point0;
		Vector2d point1;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == frame0)
			{
				point0 = sfm_f[j].observation[k].second;
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)
			{
				point1 = sfm_f[j].observation[k].second;
				has_1 = true;
			}
		}
		//在所有的点中，这两帧之间存在共视点
		if (has_0 && has_1)
		{
			Vector3d point_3d;
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}

bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (int j = 0; j < sfm_f.size(); j++)
	{
		if (sfm_f[j].state != true)
			continue;
		Vector2d point2d;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == i)
			{
				Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);
				break;
			}
		}
	}
	if (int(pts_2_vector.size()) < 15)
	{
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10)
			return false;
	}
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);


// int  cvRodrigues2cvRodr ( const CvMat* src, CvMat* dst, CvMat* jacobian=0 );
//      src为输入的旋转向量（3x1或者1x3）或者旋转矩阵（3x3）。
//      dst为输出的旋转矩阵（3x3）或者旋转向量（3x1或者1x3）。
//      jacobian为可选的输出雅可比矩阵（3x9或者9x3），是输入与输出数组的偏导数。



	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ;
	//K在这里内参默认是I都可以吗？？
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
	if(!pnp_succ)
	{
		return false;
	}
	cv::Rodrigues(rvec, r);
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;
	P_initial = T_pnp;
	return true;

}


Point2f pixel2cam(const Point2d &p, const Mat &K)
{
    return Point2f(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

//获得corres焦点
vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r,vector<SFMFeature> &sfm_f)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : sfm_f)
    {

            Vector3d a , b ;
	    for(auto &frameId:it.observation)	    
	    {
	      if(frameId.first==frame_count_l)
	      {
		a=Vector3d( frameId.second(0),frameId.second(1),0);
	      }
	      if(frameId.first==frame_count_r)
	      {
		b=Vector3d( frameId.second(0),frameId.second(1),0);
	      }
	    }
            
            corres.push_back(make_pair(a, b));
        
    }
    return corres;
}

bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l,int WINDOW_SIZE,vector<SFMFeature> &sfm_f)
{
    //corres考察视差
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = getCorresponding(i, WINDOW_SIZE-1,sfm_f);
        //判断窗口内所有的帧和最后的帧之间的交点个数
        //判断交点个数，要求大于20个
	
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            //满足特征点之间的平均距离要符合一个要求，然后使用5点法求解RT
            //这里求出的是基于第i帧的位姿
// 	    cout<<average_parallax<<endl;
            if(average_parallax * 460 > 30 && solveRelativeRT(corres, relative_R, relative_T))
            {
	      cout<<corres.size()<<endl;
                l = i;
//                 ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
        	
    }
    return false;
}
#endif