#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>  
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include<pcl/io/pcd_io.h>
#include <iostream>  
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iterator>
#include <fstream>
#include <Windows.h>
#include<cuda.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include<pcl/visualization/cloud_viewer.h>

#define NDEBUG
#include <GL/glut.h>
#pragma comment(lib,"freeglut.lib")
#include <atlstr.h> // use STL string instead, although not as convenient...
#include <atltrace.h>
#define TRACE ATLTRACE

using namespace std;
using namespace cv;
using namespace cv::cuda;
class CalibrationBox
{
public:
	/*****************************************************************
	* 构造函数
	* 输入：
	*		_board_w：标定板横向角点数
	*		_board_h：标定板纵向角点数
	*		_imageWidth：图像宽
	*		_imageHeight：图像高
	* 返回：
	*		裁剪后图像，存在dstimg文件夹中
	*****************************************************************/
	CalibrationBox(int _board_w, int _board_h, int _imageWidth, int _imageHeight);
	/*****************************************************************
	* 双目图像裁剪函数
	* 输入：
	*		待裁剪图像，存在srcimg文件夹中
	* 返回：
	*		裁剪后图像，存在dstimg文件夹中
	*****************************************************************/
	enum RET imagecropping();
	/*****************************************************************
	* 内参外参计算函数
	* 输入：
	*		裁剪后图像，存在dstimg文件夹中
	* 返回：
	*		若已经检测到名为parameter.txt的参数文件，则读取内参、
	*		外参，否则直接计算内参外参
	*****************************************************************/
	enum RET cal_parameters();
	enum RET cal_rectifymap();
	enum RET show_parameter();
	//以下为用于输出各参数的函数
	Mat get_cameraMatrixL() { return cameraMatrixL; }						    //获得左相机内参矩阵
	Mat get_distCoeffL() { return distCoeffL; }
	Mat get_cameraMatrixR() { return cameraMatrixR; }					    //获得右相机内参矩阵
	Mat get_distCoeffR() { return distCoeffR; }
	Mat get_T() { return T; }																		//获得T平移向量
	Mat get_rec() { return rec; }															    //获得rec旋转矩阵																															
	Mat get_R() { return R; }																        //获得R矩阵，用于中间计算  
	Mat get_mapLx() { return mapLx; }
	Mat get_mapLy() { return mapLy; }
	Mat get_mapRx() { return mapRx; }
	Mat get_mapRy() { return mapRy; }
	Mat get_Q() { return Q; }

private:
	Rect validROIL, validROIR;                 //双目图像校正后的ROI区域
	int board_w, board_h;							//标定板的尺寸
	const char *imageListin = 
	"../srcimage/imageListin.txt";			    //输入待处理标定图片的地址
	const char *imageListout =
	"../srcimage/imageListout.txt";			//输出待处理标定图片的地址
	const char *parameterFile=
	"parameter.txt";                                  //参数文件的名字
	int imageWidth;								    //摄像头单目的分辨率
	int imageHeight;
	Size imageSize;
	Mat cameraMatrixL;						    //左相机内参矩阵
	Mat distCoeffL;
	Mat cameraMatrixR;						    //右相机内参矩阵
	Mat distCoeffR;
	Mat T;													//T平移向量
	Mat rec;							                    //rec旋转矩阵																															
	Mat R;										            //R矩阵，用于中间计算  
	Mat Q;                                                 //视差映射矩阵
	Mat mapLx, mapLy, mapRx, mapRy; //立体校正重投影矩阵
	/*****************************************************************
	* 标定函数
	* 输入：
	*		裁剪后图像，存在dstimg文件夹中
	* 返回：
	*		标定计算内参外参
	*****************************************************************/
	RET StereoCalib();
};



class RestructionBox
{
public:
	/*****************************************************************
	* 读入多个方向的双目图片
	* 输入：
	*		imagedir：待读取图片路径
	* 返回：
	*		为imageWidth, imageHeight赋值，获得裁剪后的左右目
	*		灰度图像
	*****************************************************************/
	enum RET loadimg();
	/*****************************************************************
	* 构造函数
	* 输入：
	*		Lx：左目的列map
	*		Ly：左目的行map
	*		Rx：右目的列map
	*		Ry：右目的行map
	*		Qin：视差映射矩阵
	* 返回：
	*		初始化立体校正重映射矩阵
	*****************************************************************/
	RestructionBox(Mat Lx, Mat Ly, Mat Rx, Mat Ry, Mat Qin);
	/*****************************************************************
	* 立体校正函数
	* 输入：
	*		garyimageL/garyimageR：左右目灰度图片
	*		mapRx等：校正重映射矩阵
	* 返回：
	*		RectifyImageL/RectifyImageR：左右目立体校正后图片
	*****************************************************************/
	enum RET rectify_img();
	/*****************************************************************
	* 特征点提取函数
	* 输入：
	*		RectifyImageL/RectifyImageR：左右目立体校正后图片
	* 返回：
	*		KeypointsL/KeypointsR：左右目全部特征点
	*		best_mask：左右目的匹配度最佳的特征点掩膜
	*		KeypointsL, KeypointsR：特征点（KeyPoint格式）
	*		best_matches_pointsL, best_matches_pointsR：
	*		最好的匹配点（Point2f格式）
	*		matches_pointsL, matches_pointsR：特征点（Point2f格式）
	*****************************************************************/
	enum RET feature_extract();
	/*****************************************************************
	* 匹配结果展示函数
	* 输入：
	*		KeypointsL/KeypointsR：左右目全部特征点
	*		best_mask：左右目的匹配度最佳的特征点掩膜
	* 返回：
	*		显示每个方向的匹配结果
	*****************************************************************/
	void show_matches();
	/*****************************************************************
	* 三维坐标计算函数
	* 输入：
	*		best_matches_pointsL：左目全部特征点
	*		best_matches_pointsR：右目全部特征点
	*		Q：视差映射矩阵
	* 返回：
	*		xyz：每个视角的三维点云
	*****************************************************************/
	void cal_3dpoints();
	/*****************************************************************
	* 三角细分函数
	* 输入：
	*		best_matches_pointsL：左目全部特征点
	*		RectifyImageL：校正后左目视图
	* 返回：
	*		triangles_idx：三角面顶点的索引值
	*		glcenter：OpenGL显示中心
	*		glsize：OpenGL显示尺寸
	*****************************************************************/
	void triangle_subdiv();
	/*****************************************************************
	* 刚体变换矩阵计算函数（计算原理：本征矩阵E）
	* 输入：
	*		cameraMatrix：左相机内参矩阵
	*		distCoeff：左相机畸变矩阵
	* 返回：
	*		R：各视角间旋转矩阵
	*		T：各视角间平移矩阵
	*****************************************************************/
	//void cal_rigidtrans_emat(Mat cameraMatrix, Mat distCoeff);

	/*****************************************************************
	* 刚体变换矩阵计算函数（计算原理：SVD）
	* 输入：
	*		cameraMatrix：左相机内参矩阵
	*		distCoeff：左相机畸变矩阵
	* 返回：
	*		R：各视角间旋转矩阵
	*		T：各视角间平移矩阵
	*****************************************************************/
	void cal_rigidtrans_SVD();
	/*****************************************************************
	* 三维点刚体变换函数
	* 输入：
	*		xyz：Mat格式的每个视角的点云
	*		RT：每个视角间的刚体变换阵
	* 输出：
	*		pcl_cloud：PCL格式下旋转到主视角下的三维点云
	*****************************************************************/
	void trans_3dpoints();

	void point_cloud_process();

	void get_GLparam(Mat &img, vector<Vec3i> &tri,
		vector<Point2f> &pts2DTex, vector<Point3f> &pts3D,
		Point3f &center3D, Vec3f &size3D);

	//double RestructionBox::getTlength(int n, Mat R, Mat mask);
private:
	const string imagedir = "../testimg/*.jpg";													     //三维成像的双目图片地址
	int imageWidth, imageHeight;																	         //图片宽
	vector<Mat> rgbImageL, grayImageL;                                                              //左目原图像
	vector<Mat> rgbImageR, grayImageR;															  	 //右目原图像
	vector<Mat> RectifyImageL, RectifyImageR;
	vector<Mat> rgbRectifyImageL, rgbRectifyImageR;
	Mat mapLx, mapLy, mapRx, mapRy;															          //立体校正重投影矩阵
	int orientation_num;
	vector<vector<KeyPoint>> KeypointsL, KeypointsR;									      //左右目全部特征点（KeyPoint格式）
	vector<Mat> Disparity;
	Mat Q;																												      //视差映射矩阵
	vector<Mat> xyz;																						    	  //三维点云(Mat格式）
	vector<vector<int>>points_cloud_index;			 												  //三维点云源特征点的索引
	vector<vector<Point3f>> points_cloud;								   							  //三维点云(Point3f格式）
	vector<vector<Point2f>> matches_pointsL, matches_pointsR;                        //左右目最终特征点（Point2f格式）

	pcl::PointCloud<pcl::PointXYZ> pcl_cloud;														  //PCL格式的全体点云
	vector<Point3f> all_3dpoints;                                                                             //刚体变换后的全体三维点
	vector<vector<uchar>> fusion_ptsIdx;																  //在points_cloud中重复点的索引值，融合时应剔除
	vector<Vec3i> triangles_idx;																				  //三角剖分顶点的索引
	Point3f glcenter;                                                                                                  //OpenGL界面的显示范围
	Vec3f glsize; 
	vector<Mat> RT;																								   	 //每个视角间的旋转矩阵和平移矩阵
	vector<vector<DMatch>> matches;																	  //左右目SURF匹配结果（初步剔除误匹配）
	vector<vector<DMatch>> orients_matches;													  //多视角左目SURF匹配结果（初步剔除误匹配）
	const int hess_thresh = 50;															                          //海塞矩阵阈值
};


/*****************************************************************
* 特征匹配及误匹配剔除函数
* 输入：
*		gpu_keypoints1：左目全部特征点
*		gpu_keypoints2：右目全部特征点
* 返回：
*		goodmatches：经过对称性检验的较好的匹配点对
*****************************************************************/
void eliminate_mismatch(const GpuMat gpu_descriptors1, const GpuMat gpu_descriptors2, vector<DMatch>& goodmatches);
/*****************************************************************
* 提取图片中主要轮廓掩膜
* 输入：
*		img：输入的灰度图片
* 返回：
*		图片中主要轮廓的掩膜
*****************************************************************/
Mat roi_filter(Mat img);

Mat GetRigidTrans3D(vector<Point3f> srcPoints, vector<Point3f> dstPoints);

std::string getTimeStamp();

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewportsVis(
	pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud1, pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud2, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer);
/* OpenGL functions*/
void InitGl();

void Init_lightGl();

void displayGl();

void resizeGl(int w, int h);

void mouseGl(int button, int state, int x, int y);

void mouse_move_Gl(int x, int y);

void keyboard_control_Gl(unsigned char key, int a, int b);

void special_control_Gl(int key, int x, int y);

GLuint Create3DTexture(Mat &img, vector<Vec3i> &tri,
	vector<Point2f> pts2DTex, vector<Point3f> &pts3D,
	Point3f center3D, Vec3f size3D);

void Show(GLuint tex, Point3f center3D, Vec3i size3D);

