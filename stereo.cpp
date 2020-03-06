#include"stereo.h"
enum RET
{
	Succeed,
	CalibFolderName_Error,
	CalibImage_Error,
	NoTestImage_Error,
	TestImageSize_Error,
	NoRemapMat_Error,
	BinocularLowMatch_Error,
	OrientsLowMatch_Error,
};
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CalibrationBox::CalibrationBox(int _board_w, int _board_h, int _imageWidth, int _imageHeight)
{
	board_w = _board_w; board_h = _board_h;
	imageWidth = _imageWidth; imageHeight = _imageHeight;
	imageSize = Size(imageWidth, imageHeight);
	cameraMatrixL = Mat(3, 3, CV_64FC1);               //左相机内参矩阵
	distCoeffL = Mat(1, 5, CV_64FC1);
	cameraMatrixR = Mat(3, 3, CV_64FC1);               //右相机内参矩阵
	distCoeffR = Mat(1, 5, CV_64FC1);
	T = Mat(3, 1, CV_64FC1);                           //T平移向量
	rec = Mat(3, 3, CV_64FC1);                         //rec旋转矩阵			
}

enum RET CalibrationBox::imagecropping()
{
	ifstream fin(imageListin);
	ofstream fout(imageListout);
	cv::Rect rightrect(0, 0, imageWidth, imageHeight);
	cv::Rect leftrect(imageWidth, 0, imageWidth, imageHeight);
	char ichar[] = "0";
	char leftname[100] = "";
	char rightname[100] = "";
	if (!fin.is_open())
	{
		std::cout << getTimeStamp() << "无法打开文件 " << imageListin << endl;
		return CalibFolderName_Error;
	}
	for (int i = 0; !fin.eof(); i++)
	{
		char filename[1024];
		fin.getline(filename, 1024 - 1);
		size_t len = strlen(filename);
		while (len > 0 && isspace(filename[len - 1]))                                    //尾部的所有空字符填入\0
			filename[--len] = '\0';
		cv::Mat img = cv::imread(filename, 0);
		if (img.empty())                                                                                  //判定读取第i张图片是否成功
		{
			std::cout << getTimeStamp() << "未找到图片" << filename << endl;
			continue;
		}
		cv::Mat rightimg = img(rightrect);
		cv::Mat leftimg = img(leftrect);
		CreateDirectory("../dstimage", NULL);
		sprintf(leftname, "../dstimage/left%i.jpg", i);
		sprintf(rightname, "../dstimage/right%i.jpg", i);
		cv::imwrite(leftname, leftimg);
		cv::imwrite(rightname, rightimg);
		fout << leftname << "\n";
		fout << rightname << "\n";
	}
	return Succeed;
}

enum RET CalibrationBox::cal_parameters()
{
	FILE* fparameter;
	fparameter = fopen("parameter.txt", "r+");
	//------若无参数文件，调用立体标定函数求出参数------------------------------------------
	if (NULL == (fparameter))
	{

		std::cout << getTimeStamp() << "无内参外参文件，正在进行标定..." << endl;
		StereoCalib();
		FILE* fparameter = fopen("parameter.txt", "w");
		setbuf(fparameter, NULL);
		for (int i = 0; i <= 2; i++)                                       //将6个参数矩阵写入parameter.txt文件中
		{
			for (int j = 0; j <= 2; j++)
				fprintf(fparameter, "%lf ", *cameraMatrixL.ptr<double>(i, j));
			fprintf(fparameter, "\n");
		}
		for (int i = 0; i <= 2; i++)
		{
			for (int j = 0; j <= 2; j++)
				fprintf(fparameter, "%lf ", *cameraMatrixR.ptr<double>(i, j));
			fprintf(fparameter, "\n");
		}

		for (int i = 0; i <= 4; i++)
		{
			fprintf(fparameter, "%lf ", *distCoeffL.ptr<double>(0, i));
			fprintf(fparameter, "\n");
		}

		for (int i = 0; i <= 4; i++)
		{
			fprintf(fparameter, "%lf ", *distCoeffR.ptr<double>(0, i));
			fprintf(fparameter, "\n");
		}

		for (int i = 0; i <= 2; i++)
		{
			for (int j = 0; j <= 2; j++)
				fprintf(fparameter, "%lf ", *rec.ptr<double>(i, j));
			fprintf(fparameter, "\n");
		}

		for (int i = 0; i <= 2; i++)
		{
			fprintf(fparameter, "%lf ", *T.ptr<double>(i, 0));
			fprintf(fparameter, "\n");
		}
		fclose(fparameter);
	}
	//------若有参数文件，读取参数文件------------------------------------------------------
	else
	{
		std::cout << getTimeStamp() << "正在读取内参外参矩阵..." << endl;
		for (int i = 0; i <= 2; i++)                                       //将6个参数矩阵从parameter.txt文件中读出
			for (int j = 0; j <= 2; j++)
				fscanf(fparameter, "%lf ", cameraMatrixL.ptr<double>(i, j));

		for (int i = 0; i <= 2; i++)
			for (int j = 0; j <= 2; j++)
				fscanf(fparameter, "%lf ", cameraMatrixR.ptr<double>(i, j));

		for (int i = 0; i <= 4; i++)
			fscanf(fparameter, "%lf ", distCoeffL.ptr<double>(0, i));

		for (int i = 0; i <= 4; i++)
			fscanf(fparameter, "%lf ", distCoeffR.ptr<double>(0, i));

		for (int i = 0; i <= 2; i++)
			for (int j = 0; j <= 2; j++)
				fscanf(fparameter, "%lf ", rec.ptr<double>(i, j));

		for (int i = 0; i <= 2; i++)
			fscanf(fparameter, "%lf ", T.ptr<double>(i, 0));

		fclose(fparameter);
	}
	return Succeed;
}

enum RET CalibrationBox::StereoCalib()
{
	bool displayCorners = true;
	bool showUndistorted = true;
	bool isVerticalStereo = false; // horiz or vert cams
	const int maxScale = 1;
	const float squareSize = 30;

	// actual square size
	FILE *f = fopen(imageListout, "rt");
	int i, j, lr;                                                                                            //lr=0,左目;lr=1，右目
	int N = board_w * board_h;
	cv::Size board_sz = cv::Size(board_w, board_h);
	vector<string> imageNames[2];                                                     //imageName[0]存放左目，imageName[1]存放右目
	vector<cv::Point3f> boardModel;
	vector<vector<cv::Point3f> > objectPoints;
	vector<vector<cv::Point2f> > points[2];                                        //point[0]中存储的是左目的图片中角点的像素坐标，point[1]为右目
	vector<cv::Point2f> corners[2];
	bool found[2] = { false, false };
	cv::Size imageSize;

	// READ IN THE LIST OF CIRCLE GRIDS:
	//
	for (i = 0; i < board_h; i++)
		for (j = 0; j < board_w; j++)
			boardModel.push_back(
				cv::Point3f((float)(i * squareSize), (float)(j * squareSize), 0.f));  //i*j即为角点个数，boardModel动态数组中存储了角点在第几行第几列的信息
	i = 0;
	for (;;) {
		char buf[1024];
		lr = i % 2;
		if (lr == 0)
			found[0] = found[1] = false;
		if (!fgets(buf, sizeof(buf) - 3, f))                                                 //把f指向的文件中的字符的第一行存入buf字符数组
			break;
		size_t len = strlen(buf);
		while (len > 0 && isspace(buf[len - 1]))                                     //尾部的空字符填入\0
			buf[--len] = '\0';
		if (buf[0] == '#')
			continue;
		cv::Mat img = cv::imread(buf, 0);                                              //读取当前行的图片进入img
		if (img.empty())
			break;
		imageSize = img.size();
		imageNames[lr].push_back(buf);                                              //把图片名称存入左右目对应的动态字符数组
		i++;

		// If we did not find board on the left image,
		// it does not make sense to find it on the right.
		//
		if (lr == 1 && !found[0])
			continue;

		// Find circle grids and centers therein:
		for (int s = 1; s <= maxScale; s++) {
			cv::Mat timg = img;
			if (s > 1)
				cv::resize(img, timg, cv::Size(), s, s, cv::INTER_CUBIC);
			// Just as example, this would be the call if you had circle calibration
			// boards ...
			//      found[lr] = cv::findCirclesGrid(timg, cv::Size(nx, ny),
			//      corners[lr],
			//                                      cv::CALIB_CB_ASYMMETRIC_GRID |
			//                                          cv::CALIB_CB_CLUSTERING);
			//...but we have chessboards in our images
			found[lr] = cv::findChessboardCorners(timg, board_sz, corners[lr]);

			if (found[lr] || s == maxScale) {
				cv::Mat mcorners(corners[lr]);
				mcorners *= (1. / s);
			}
			if (found[lr])
				break;
		}
		if (displayCorners) {
			std::cout << buf << endl;
			cv::Mat cimg;
			cv::cvtColor(img, cimg, cv::COLOR_GRAY2BGR);

			// draw chessboard corners works for circle grids too
			cv::drawChessboardCorners(cimg, cv::Size(board_w, board_h), corners[lr], found[lr]);
			cv::imshow("Corners", cimg);
			if ((cv::waitKey(0) & 255) == 27) // Allow ESC to quit
				exit(-1);
		}
		else
			std::cout << '.';
		if (lr == 1 && found[0] && found[1]) {
			objectPoints.push_back(boardModel);
			points[0].push_back(corners[0]);
			points[1].push_back(corners[1]);
		}
	}
	fclose(f);

	// CALIBRATE THE STEREO CAMERAS
	cameraMatrixL = cv::Mat::eye(3, 3, CV_64FC1);
	cameraMatrixR = cv::Mat::eye(3, 3, CV_64FC1);
	cv::Mat E, F;
	std::cout << "\nRunning stereo calibration ...\n";
	cv::stereoCalibrate(
		objectPoints, points[0], points[1], cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, rec, T, E, F,
		cv::CALIB_FIX_ASPECT_RATIO | cv::CALIB_ZERO_TANGENT_DIST |
		cv::CALIB_SAME_FOCAL_LENGTH,
		cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 100,
			1e-5));

	std::cout << "Done! Press any key to step through images, ESC to exit\n\n" << endl;

	// CALIBRATION QUALITY CHECK
	// because the output fundamental matrix implicitly
	// includes all the output information,
	// we can check the quality of calibration using the
	// epipolar geometry constraint: m2^t*F*m1=0
	vector<cv::Point3f> lines[2];
	double avgErr = 0;
	int nframes = (int)objectPoints.size();
	for (i = 0; i < nframes; i++) {
		vector<cv::Point2f> &pt0 = points[0][i];
		vector<cv::Point2f> &pt1 = points[1][i];
		cv::undistortPoints(pt0, pt0, cameraMatrixL, distCoeffL, cv::Mat(), cameraMatrixL);
		cv::undistortPoints(pt1, pt1, cameraMatrixR, distCoeffR, cv::Mat(), cameraMatrixR);
		cv::computeCorrespondEpilines(pt0, 1, F, lines[0]);
		cv::computeCorrespondEpilines(pt1, 2, F, lines[1]);

		for (j = 0; j < N; j++) {
			double err = fabs(pt0[j].x * lines[1][j].x + pt0[j].y * lines[1][j].y +
				lines[1][j].z) +
				fabs(pt1[j].x * lines[0][j].x + pt1[j].y * lines[0][j].y +
					lines[0][j].z);
			avgErr += err;
		}
	}

	std::cout << getTimeStamp ()<< "Calibrate done!\navg err = " << avgErr / (nframes * N) << endl;
	return Succeed;
}

enum RET CalibrationBox::show_parameter()
{
	std::cout << getTimeStamp() << "内参外参如下:" << endl;
	std::cout << "cameraMatrixL=\n" << cameraMatrixL << endl;
	std::cout << "cameraMatrixR=\n" << cameraMatrixR << endl;
	std::cout << "distCoeffL=\n" << distCoeffL << endl;
	std::cout << "distCoeffR=\n" << distCoeffR << endl;
	std::cout << "rec=\n" << rec << endl;
	std::cout << "T=\n" << T << endl;
	return Succeed;
}

enum RET CalibrationBox::cal_rectifymap()
{
	Mat Rl, Rr, Pl, Pr;
	R = rec.clone();
	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
		0, imageSize, &validROIL, &validROIR);
	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy);
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);
	return Succeed;
}


//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
RestructionBox::RestructionBox(Mat Lx, Mat Ly, Mat Rx, Mat Ry,Mat Qin)
{
	mapLx = Lx.clone(); mapLy = Ly.clone(), mapRx = Rx.clone(), mapRy = Ry.clone(); Q = Qin.clone();
}

enum RET RestructionBox::loadimg()
{
	vector<String> imglist;
	glob(imagedir, imglist);
	if (imglist.size() == 0) {
		std::cout << getTimeStamp() << "无法读取测试双目图片！\n";
		return NoTestImage_Error;
	}
	orientation_num = imglist.size();
	imageWidth = imread(imglist[0]).cols / 2;
	imageHeight = imread(imglist[0]).rows;
	for (int n = 0, width = imread(imglist[0]).cols; n < orientation_num; n++)
	{
		Mat temp = imread(imglist[n]);
		if (temp.cols != width){
			std::cout << getTimeStamp() << "测试文件夹中的双目图片尺寸不一致！\n";
			return TestImageSize_Error;
		}
		width = temp.cols;
		rgbImageL.push_back(temp(Rect(imageWidth, 0, imageWidth, imageHeight)));
		rgbImageR.push_back(temp(Rect(0, 0, imageWidth, imageHeight)));

		cv::cvtColor(rgbImageL[n], temp, CV_BGR2GRAY);
		grayImageL.push_back(temp);
		Mat L = grayImageL[n];

		Mat temp2;
		cv::cvtColor(rgbImageR[n], temp2, CV_BGR2GRAY);
		grayImageR.push_back(temp2);
		Mat R = grayImageR[n];

	}
	return Succeed;
}

enum RET RestructionBox::rectify_img()
{
	if (mapLx.empty() == 1 || mapLy.empty() == 1 || mapRx.empty() == 1 || mapRy.empty() == 1)
	{
		cout << getTimeStamp() << "没有输入四个重映射矩阵！\n";
		return NoRemapMat_Error;
	}
	for (int n = 0; n < orientation_num; n++)
	{
		Mat temp, rgbtemp;
		cv::remap(rgbImageL[n], rgbtemp, mapLx, mapLy, INTER_LINEAR);
		cv::cvtColor(rgbtemp, temp, CV_BGR2GRAY);
		RectifyImageL.push_back(temp);
		rgbRectifyImageL.push_back(rgbtemp);

		Mat temp2, rgbtemp2;
		cv::remap(rgbImageR[n], rgbtemp2, mapRx, mapRy, INTER_LINEAR);
		cv::cvtColor(rgbtemp2, temp2, CV_BGR2GRAY);
		RectifyImageR.push_back(temp2);
		rgbRectifyImageR.push_back(rgbtemp2);
	}
	return Succeed;
}

enum RET RestructionBox::feature_extract()
{
	//【1】对图一的左右相机图片进行特征提取------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	double starttime = getTickCount();
	std::cout << getTimeStamp() << "正在进行特征提取...\n";
	int N = orientation_num;
	cv::cuda::SURF_CUDA SURF(hess_thresh);
	vector<cv::cuda::GpuMat> gpu_keypointsL(N), gpu_keypointsR(N);
	vector<cv::cuda::GpuMat> gpu_descriptorL(N), gpu_descriptorR(N);
	vector<cv::cuda::GpuMat> gpu_grayimgL(N), gpu_grayimgR(N);
	vector<cv::cuda::GpuMat> gpu_roimaskL(N), gpu_roimaskR(N);
	for (int n = 0; n < N; n++){
		gpu_grayimgL[n].upload(RectifyImageL[n]);
		gpu_grayimgR[n].upload(RectifyImageR[n]);
		Mat roi_tempL, roi_tempR;
		roi_tempL = roi_filter(RectifyImageL[n]).clone();
		roi_tempR = roi_filter(RectifyImageR[n]).clone();
		gpu_roimaskL[n].upload(roi_tempL);
		gpu_roimaskR[n].upload(roi_tempR);
	}
	for (int n = 0; n < N; n++)
	{
		SURF(gpu_grayimgL[n], gpu_roimaskL[n], gpu_keypointsL[n], gpu_descriptorL[n]);
		SURF(gpu_grayimgR[n], gpu_roimaskR[n], gpu_keypointsR[n], gpu_descriptorR[n]);
		vector<KeyPoint> temp;
		SURF.downloadKeypoints(gpu_keypointsL[n], temp); 
		KeypointsL.push_back(temp);

		vector<KeyPoint> temp2;
		SURF.downloadKeypoints(gpu_keypointsR[n], temp2); 
		KeypointsR.push_back(temp2);
	}
	double endtime = getTickCount();
	std::cout <<getTimeStamp()<< "特征检测与描述子生成用时：" << (endtime - starttime) / (getTickFrequency()) << "s" << endl;
	//【2】特征匹配-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	starttime = getTickCount();
	for (int n = 0; n < N; n++)
	{
		vector<DMatch> temp,orient_temp;
		eliminate_mismatch(gpu_descriptorL[n], gpu_descriptorR[n], temp);		
		matches.push_back(temp);
		if (n < N - 1)
		{
			eliminate_mismatch(gpu_descriptorL[n], gpu_descriptorL[n + 1], orient_temp);
			orients_matches.push_back(orient_temp);
		}
	
	}
		
	vector<vector<Point2f>> good_matches_pointsL(N), good_matches_pointsR(N);
	//vector<vector<Point2f>> best_matches_pointsL(N - 1), best_matches_pointsR2(N - 1);
	vector<vector<char>> bestmask(N);
	//【3】误匹配点剔除以及匹配度过低报警------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//先筛选出相对较好的匹配

	for (int n = 0; n < N; n++)
	{
		vector<Point2f> tempL,tempR;
		vector<int> pc_index;
		for (int i = 0; i < matches[n].size(); i++)
		{
			tempL.push_back(KeypointsL[n][matches[n][i].queryIdx].pt);
			tempR.push_back(KeypointsR[n][matches[n][i].trainIdx].pt);
			pc_index.push_back(matches[n][i].queryIdx);
		}
		points_cloud_index.push_back(pc_index);
		matches_pointsL.push_back(tempL);
		matches_pointsR.push_back(tempR);
	}

	for (int n = 0; n < N; n++)
	{
		vector<DMatch>::iterator iteD= matches[n].begin();
		vector<Point2f>::iterator itePtL= matches_pointsL[n].begin();
		vector<Point2f>::iterator itePtR = matches_pointsR[n].begin();
		vector<int>::iterator iteidx = points_cloud_index[n].begin();
		for (; iteD != matches[n].end();)
		{
			if (abs(itePtR->y - itePtL->y) > 5)
			{
				iteD=matches[n].erase(iteD);
				itePtL=matches_pointsL[n].erase(itePtL);
				itePtR=matches_pointsR[n].erase(itePtR);
				iteidx = points_cloud_index[n].erase(iteidx);
			}
			else
			{
				iteD++; itePtL++; itePtR++; iteidx++;
			}
		}
		int best_num = matches[n].size();
		float ratio = float(best_num) / float(KeypointsL[n].size());
		std::cout << "第" << n<< "个视角的双目图片之间" << endl;
		std::cout << "好的匹配点有" << best_num << "个" << endl;
		std::cout << "总的匹配点有" << KeypointsL[n].size() << "个" << endl;
		std::cout << "匹配率为" << ratio * 100 << "%" << endl;
		if (ratio < 0.10)
		{
			std::cout << getTimeStamp() << "第" << n << "个视角左右目之间匹配度太低" << endl;
			return BinocularLowMatch_Error;
		}
		if (n < orientation_num - 1)
		{
			int orient_num = orients_matches[n].size();
			float ratio2 = float(orient_num) / float(KeypointsL[n].size());
			if (ratio < 0.10)
			{
				std::cout << getTimeStamp() << "第" << n << "和" << n + 1 << "个视角左目图片之间匹配度太低" << endl;
				return OrientsLowMatch_Error;
			}
		}
	}

	endtime = getTickCount();
	std::cout << getTimeStamp() << "特征匹配用时：" << (endtime - starttime) / (getTickFrequency()) << "s" << endl;
	return Succeed;


}

void RestructionBox::show_matches()
{
	vector<Mat> img_matches(orientation_num);
	for (int n = 0; n < orientation_num; n++)
	{
		char a[2] = { ' ','\0' };
		a[0] = n + 48;
		string Num = a;
		drawMatches(RectifyImageL[n], KeypointsL[n], RectifyImageR[n], KeypointsR[n],
			matches[n], img_matches[n], Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::DEFAULT);
		cv::imshow("图" + Num + "【匹配结果】", img_matches[n]);
	}
	cv::waitKey(0);
	for (int n = 0; n < orientation_num-1; n++)
	{
		char a[2] = { ' ','\0' };
		a[0] = n + 48;
		string Num = a;
		drawMatches(RectifyImageL[n], KeypointsL[n], RectifyImageL[n+1], KeypointsL[n+1],
			orients_matches[n], img_matches[n], Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::DEFAULT);
		cv::imshow("图" + Num + "【多视角匹配结果】", img_matches[n]);
	}
	cv::waitKey(0);
}

void RestructionBox::cal_3dpoints()
{
	//计算视差图
	for (int n = 0; n < orientation_num; n++)
	{
		Mat temp = Mat::zeros(4, matches_pointsL[n].size(), CV_64F);
		for (int i = 0; i < matches_pointsL[n].size(); i++)
		{
				temp.at<double>(0, i) = matches_pointsL[n][i].x;
				temp.at<double>(1, i) = matches_pointsL[n][i].y;
				temp.at<double>(2, i) = abs(matches_pointsL[n][i].x - matches_pointsR[n][i].x);
				temp.at<double>(3, i) = 1;
		}
		Disparity.push_back(temp);
	}
	//根据视差，计算深度Z
	for (int n = 0; n < orientation_num; n++)
	{
		Mat temp = Mat::zeros(4, matches_pointsL[n].size(), CV_64F);
		for (int i = 0; i < matches_pointsL[n].size(); i++)
		{
			temp.colRange(i, i + 1) = Q*Disparity[n].colRange(i, i + 1);
			temp.colRange(i, i + 1) = temp.colRange(i, i + 1) / temp.at<double>(3, i);
		}	
		xyz.push_back(temp);
	}

	for (int n = 0; n < orientation_num; n++)
	{
		Mat xyzt = xyz[n].t();
		vector<Point3f> temp_point;
		for (int i = 0; i < xyzt.rows; i++)
		{	
			double *data = xyzt.ptr<double>(i);
			temp_point.push_back(Point3f(data[0], data[1], data[2]));
		}
		points_cloud.push_back(temp_point);
	}
}



void RestructionBox::cal_rigidtrans_SVD()
{
	for (int n = 0; n < orientation_num - 1; n++)
	{
		vector<Point3f> pt3d1, pt3d2;
		vector<uchar> fusion;
		for (int j = 0; j < orients_matches[n].size(); j++)
		{
			int k = 0, l = 0;
			for (; k < points_cloud[n].size(); k++) {//找出第n个视角下的特征点对应的三维点的下标l，找到就跳出循环
				if (points_cloud_index[n][k] == orients_matches[n][j].queryIdx) {
					break;
				}
			}
			for (; l < points_cloud[n + 1].size(); l++) {//找出第n+1个视角下的特征点对应的三维点的下标k，找到就跳出循环
				if (points_cloud_index[n + 1][l] == orients_matches[n][j].trainIdx) {
					break;
				}
			}
			if (k != points_cloud[n].size() && l != points_cloud[n + 1].size())//该特征点在第n和n+1个视角下必须是一个三维点，若不是则跳过该点
			{
				pt3d1.push_back(points_cloud[n][k]);
				pt3d2.push_back(points_cloud[n + 1][l]);
				fusion.push_back(k);
			}
		}
		fusion_ptsIdx.push_back(fusion);
		Mat temp=GetRigidTrans3D(pt3d2, pt3d1);
		RT.push_back(temp);
	}
}




void RestructionBox::trans_3dpoints()
{
	vector <Mat> pts3d;//旋转到主视角的三维点云
	vector<vector<uchar>> fusion_mask;
	pts3d.push_back(xyz[0].rowRange(0, 3));
	for (int n = 1; n < orientation_num; n++)
	{
		Mat temp = xyz[n].clone();
		for (int view = n; view > 0; view--)
			temp = RT[view - 1] * temp;
		pts3d.push_back(temp.rowRange(0, 3));
	}

	for (int n = 0; n < orientation_num - 1; n++)
	{
		vector<uchar> temp;
		for (int i = 0; i < points_cloud[n].size(); i++)
			temp.push_back(1);
		fusion_mask.push_back(temp);
	}
	//计算融合时剔除点的索引值
	for (int n = 0; n < fusion_ptsIdx.size(); n++)
		for (int i = 0; i < fusion_ptsIdx[n].size(); i++)
			fusion_mask[n][fusion_ptsIdx[n][i]] = 0;

	for (int n = 0; n < orientation_num; n++)
	{
		//将三维点云输出为点云格式
		for (int i = 0; i < pts3d[n].cols; i++)
		{
			if (n < orientation_num - 1)
			{
				if (fusion_mask[n][i])
				{
					float x = pts3d[n].at<double>(0, i);
					float y = pts3d[n].at<double>(1, i);
					float z = pts3d[n].at<double>(2, i);
					pcl_cloud.push_back(pcl::PointXYZ(x, y, z));
				}
			}
			else//最后一个视角保留其全部点云不做剔除
			{
				float x = pts3d[n].at<double>(0, i);
				float y = pts3d[n].at<double>(1, i);
				float z = pts3d[n].at<double>(2, i);
				pcl_cloud.push_back(pcl::PointXYZ(x, y, z));
			}
		}
	}

	//FILE *foutall = fopen("all3Dpoints.txt", "w");
	//setbuf(foutall, NULL);
	//for (int n = 0; n < orientation_num; n++)
	//{
	//	//将三维点云输出为txt

	//	for (int i = 0; i < pts3d[n].cols; i++)
	//	{
	//		if (n < orientation_num - 1)
	//		{
	//			if (fusion_mask[n][i])
	//			{
	//				float x = pts3d[n].at<double>(0, i);
	//				float y = pts3d[n].at<double>(1, i);
	//				float z = pts3d[n].at<double>(2, i);
	//				fprintf(foutall, "%lf ", x);
	//				fprintf(foutall, "%lf ", y);
	//				fprintf(foutall, "%lf\n", z);
	//			}
	//		}
	//		else
	//		{
	//			float x = pts3d[n].at<double>(0, i);
	//			float y = pts3d[n].at<double>(1, i);
	//			float z = pts3d[n].at<double>(2, i);
	//			fprintf(foutall, "%lf ", x);
	//			fprintf(foutall, "%lf ", y);
	//			fprintf(foutall, "%lf\n", z);
	//		}
	//	}
	//}
	//std::fclose(foutall);

	//for (int n = 0; n < orientation_num; n++)
	//{
	//	//将三维点云输出为txt
	//	char index[10] = { '3','D','p','t',' ','.','t','x','t','\0' };
	//	index[4] = 48 + n;
	//	FILE *fout = fopen(index, "w");
	//	setbuf(fout, NULL);
	//	for (int i = 0; i < pts3d[n].cols; i++)
	//	{
	//		if (n < orientation_num - 1)
	//		{
	//			if (fusion_mask[n][i])
	//			{
	//				float x = pts3d[n].at<double>(0, i);
	//				float y = pts3d[n].at<double>(1, i);
	//				float z = pts3d[n].at<double>(2, i);
	//				fprintf(fout, "%lf ", x);
	//				fprintf(fout, "%lf ", y);
	//				fprintf(fout, "%lf\n", z);
	//			}
	//		}
	//		else
	//		{
	//			float x = pts3d[n].at<double>(0, i);
	//			float y = pts3d[n].at<double>(1, i);
	//			float z = pts3d[n].at<double>(2, i);
	//			fprintf(fout, "%lf ", x);
	//			fprintf(fout, "%lf ", y);
	//			fprintf(fout, "%lf\n", z);
	//		}
	//	}
	//	std::fclose(fout);
	//}
}

void RestructionBox::point_cloud_process()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr allpts_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
	sor.setInputCloud(pcl_cloud.makeShared());
	sor.setMeanK(10);
	sor.setStddevMulThresh(1.5);
	sor.filter(*allpts_filtered);
	//sor.setNegative(true);
	//sor.filter(*cloud_filtered);

	pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;
	mls.setComputeNormals(true);
	//点云平滑
	pcl::PointCloud<pcl::PointXYZ>::Ptr allpts_mls(new pcl::PointCloud<pcl::PointXYZ>);
	//pcl::copyPointCloud(allpts, *allpts_mls);
	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_outlier(new pcl::PointCloud<pcl::PointXYZ>);
	// 创建一个KD树
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	// 输出文件中有PointNormal类型，用来存储移动最小二乘法算出的法线
	pcl::PointCloud<pcl::PointNormal> mls_points;
	// 定义对象 (第二种定义类型是为了存储法线, 即使用不到也需要定义出来)
	mls.setComputeNormals(true);
	//设置参数
	mls.setInputCloud(allpts_filtered);
	mls.setPolynomialFit(true);
	mls.setSearchMethod(tree);
	mls.setSearchRadius(10);
	mls.setPolynomialOrder(10);
	// 曲面重建
	mls.process(*allpts_mls);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer1(new pcl::visualization::PCLVisualizer("SOR前后对比"));
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer2(new pcl::visualization::PCLVisualizer("MLS前后对比"));
	viewportsVis(pcl_cloud.makeShared(), allpts_filtered, viewer1);
	viewportsVis(allpts_filtered, allpts_mls, viewer2);
	while (!viewer1->wasStopped() && !viewer2->wasStopped())
	{
		viewer1->spinOnce(100);
		viewer2->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
	pcl::copyPointCloud(*allpts_mls, pcl_cloud);
}

void RestructionBox::triangle_subdiv()
{
	cv::Rect rect(0, 0, imageWidth, imageHeight); // Our outer bounding box
	cv::Subdiv2D subdiv(rect); // Create the initial subdivision
	cv::Point2f pt; //This is our point holder
	int k = 0;
	for (int i = 0; i < matches_pointsL[k].size(); i++) {
		pt = matches_pointsL[k][i];
		subdiv.insert(pt);
	}
	vector<cv::Vec6f> triangles;
	subdiv.getTriangleList(triangles);
	vector<Point> p(3);
	Mat img = RectifyImageL[k].clone();

	for (int i = 0; i < triangles.size(); i++)
	{
		Vec6f t = triangles[i];
		p[0] = Point(cvRound(t[0]), cvRound(t[1]));
		p[1] = Point(cvRound(t[2]), cvRound(t[3]));
		p[2] = Point(cvRound(t[4]), cvRound(t[5]));
		// Draw rectangles completely inside the image.
		if (rect.contains(p[0]) && rect.contains(p[1]) && rect.contains(p[2]))
		{
			line(img, p[0], p[1], Scalar(rand() % 255, rand() % 255, rand() % 255), 1, CV_AA, 0);
			line(img, p[1], p[2], Scalar(rand() % 255, rand() % 255, rand() % 255), 1, CV_AA, 0);
			line(img, p[2], p[0], Scalar(rand() % 255, rand() % 255, rand() % 255), 1, CV_AA, 0);
		}
	}
	cv::imshow("三角剖分", img);
	cv::waitKey(0);

	for (vector<Vec6f>::iterator tri = triangles.begin(); tri != triangles.end(); tri++)
	{
		Vec3i tri_idx;
		for (int n = 0; n < 3; n++)
		{
			int edge_idx = -1, vertex_idx = -1;
			//若顶点在图片范围内
			if (tri[0][2 * n] < imageWidth&&tri[0][2 * n]>0 && tri[0][2 * n + 1] < imageHeight&&tri[0][2 * n + 1]>0)
			{
				Point2f temp_pt(tri[0][2 * n], tri[0][2 * n + 1]);
				subdiv.locate(temp_pt, edge_idx, vertex_idx);
			}
			//三角细分类中的vertex的前四个是虚顶点，故与特征点的下标有4的偏移量
			tri_idx[n] = vertex_idx - 4;
		}
		//若索引值为-1，则表示该点是虚顶点或该点并未落在顶点上，排除之
		if (tri_idx[0] > 0 && tri_idx[1] > 0 && tri_idx[2] > 0)
			triangles_idx.push_back(tri_idx);
	}
}


void RestructionBox::get_GLparam(Mat &img, vector<Vec3i> &tri,
	vector<Point2f> &pts2DTex, vector<Point3f> &pts3D,
	Point3f &center3D, Vec3f &size3D)
{
	//计算opengl的立体模型空间位置
	float minX = 1e9, maxX = -1e9;
	float minY = 1e9, maxY = -1e9;
	float minZ = 1e9, maxZ = -1e9;
	for (vector<Point3f>::iterator pt3D = points_cloud[0].begin(); pt3D != points_cloud[0].end(); pt3D++)
	{
		minX = min(minX, pt3D->x); maxX = max(maxX, pt3D->x);
		minY = min(minY, pt3D->y); maxY = max(maxY, pt3D->y);
		minZ = min(minZ, pt3D->z); maxZ = max(maxZ, pt3D->z);
	}

	glcenter.x = (minX + maxX) / 2;
	glcenter.y = (minY + maxY) / 2;
	glcenter.z = (minZ + maxZ) / 2;
	glsize[0] = maxX - minX;
	glsize[1] = maxY - minY;
	glsize[2] = maxZ - minZ;
	img = rgbRectifyImageL[0].clone();
	tri = triangles_idx;
	pts2DTex = matches_pointsL[0];
	pts3D = points_cloud[0];
	center3D = glcenter;
	size3D = glsize;
}

Mat roi_filter(Mat img)
{
	Mat edgex,edgey,edge;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	GaussianBlur(img, img, Size(7, 7), 0);
	Sobel(img, edgex, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(edgex, edgex);
	Sobel(img, edgey, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(edgey, edgey);
	cv::cuda::addWeighted(edgex, 0.5, edgey, 0.5, 0, edge);
	double thresh=cv::threshold(edge, edge, 0, 255, THRESH_OTSU);
	findContours(edge, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	vector<float> area;
	for (int i = 0; i < contours.size(); i++)
		area.push_back(contourArea(contours[i]));
	double avg_area = mean(area)[0];
	for (vector<vector<Point>>::iterator ite = contours.begin(); ite != contours.end(); )
	{
		if (contourArea(*ite) < avg_area/2)
			ite = contours.erase(ite);
		else
			ite++;
	}
	//仅用于观察当前轮廓
	Mat imgcontours = Mat::zeros(img.rows, img.cols, CV_8UC3);
	vector<Point> all_contours;
	for (int i = 0; i < contours.size(); i++)
	{
		all_contours.insert(all_contours.end(), contours[i].begin(), contours[i].end());
		Scalar color = Scalar(rand() % 255, rand() % 255, rand() % 255);
		drawContours(imgcontours, contours, i, color);
	}
	//Mat kernel = getStructuringElement(MORPH_RECT, Size(img.cols / 30, img.cols / 30));
	//cv::morphologyEx(edge, edge, MORPH_CLOSE, kernel);
	//cv::morphologyEx(edge, edge, MORPH_OPEN, kernel);
	vector<int> hull;
	convexHull(Mat(all_contours), hull);
	vector<Point> object_hull;
	for (int i = 0; i < hull.size(); i++)
		object_hull.push_back(all_contours[hull[i]]);
	
	Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
	vector<vector<Point>> a;
	a.push_back(object_hull);
	drawContours(mask, a, 0, Scalar(255),-1);
	return mask;
}

void eliminate_mismatch(const GpuMat gpu_descriptors1,const GpuMat gpu_descriptors2, vector<DMatch>& goodmatches)
{
	auto matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_L2);
	vector<DMatch>matches1, matches2;
	matcher->match(gpu_descriptors1, gpu_descriptors2, matches1);
	matcher->match(gpu_descriptors2, gpu_descriptors1, matches2);
	for (vector<DMatch>::const_iterator matchIterator1 = matches1.begin(); matchIterator1 != matches1.end(); ++matchIterator1)
		for (vector<DMatch>::const_iterator matchIterator2 = matches2.begin(); matchIterator2 != matches2.end(); ++matchIterator2)
			if ((*matchIterator1).queryIdx == (*matchIterator2).trainIdx && (*matchIterator1).trainIdx == (*matchIterator2).queryIdx&& (*matchIterator1).distance<0.2&& (*matchIterator2).distance<0.2)
				goodmatches.push_back(*matchIterator1);
}

Mat GetRigidTrans3D(vector<Point3f> srcPoints, vector<Point3f> dstPoints)
{
	Point3f p_center = (0, 0, 0);
	Point3f q_center = (0, 0, 0);
	int Points_Num = srcPoints.size();
	for (int i = 0; i < Points_Num; i++)
	{
		p_center.x += srcPoints[i].x;
		p_center.y += srcPoints[i].y;
		p_center.z += srcPoints[i].z;
		q_center.x += dstPoints[i].x;
		q_center.y += dstPoints[i].y;
		q_center.z += dstPoints[i].z;
	}
	p_center = p_center / Points_Num;
	q_center = q_center / Points_Num;
	Mat matX(3, Points_Num, CV_64FC1);
	Mat matY(3, Points_Num, CV_64FC1);
	double* data_x = matX.ptr<double>(0);
	double* data_y = matX.ptr<double>(1);
	double* data_z = matX.ptr<double>(2);
	for (int i = 0; i < Points_Num; i++, data_x++, data_y++, data_z++)
	{
		*data_x = srcPoints[i].x - p_center.x;
		*data_y = srcPoints[i].y - p_center.y;
		*data_z = srcPoints[i].z - p_center.z;
	}

	data_x = matY.ptr<double>(0);
	data_y = matY.ptr<double>(1);
	data_z = matY.ptr<double>(2);
	for (int i = 0; i < Points_Num; i++, data_x++, data_y++, data_z++)
	{
		*data_x = dstPoints[i].x - q_center.x;
		*data_y = dstPoints[i].y - q_center.y;
		*data_z = dstPoints[i].z - q_center.z;
	}
	Mat S = matX*matY.t();
	Mat W, U, Vt;
	SVDecomp(S, W, U, Vt);
	cv::Mat UVt = U * Vt;
	double det = cv::determinant(UVt);
	double datM[] = { 1, 0, 0, 0, 1, 0, 0, 0, det };
	Mat M(3, 3, CV_64FC1, datM);
	Mat R = Vt.t() * M * U.t();
	Mat T(3, 1, CV_64FC1);
	double* dataR = R.ptr<double>(0);
	T.at<double>(0, 0) = q_center.x - (dataR[0] * p_center.x + dataR[1] * p_center.y + dataR[2] * p_center.z);
	T.at<double>(1, 0) = q_center.y - (dataR[3] * p_center.x + dataR[4] * p_center.y + dataR[5] * p_center.z);
	T.at<double>(2, 0) = q_center.z - (dataR[6] * p_center.x + dataR[7] * p_center.y + dataR[8] * p_center.z);
	Mat RT;
	hconcat(R, T, RT);
	Mat_<double> Z = (Mat_<double>(1, 4) << 0, 0, 0, 1);
	vconcat(RT, Z, RT);
	return RT;
}

std::string getTimeStamp()
{
	char buf[1024];
	SYSTEMTIME systime;
	GetLocalTime(&systime);
	sprintf_s(buf, "%04d/%02d/%02d-%02d:%02d:%02d.%03d ", systime.wYear, systime.wMonth, systime.wDay,
		systime.wHour, systime.wMinute, systime.wSecond, systime.wMilliseconds);
	return std::string(buf);
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewportsVis(
	pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud1, pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud2, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer)
{

	viewer->initCameraParameters();
	int v1(0);
	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer->setBackgroundColor(0, 0, 0, v1);
	viewer->addText("Cloud1", 10, 10, "v1 text", v1);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color1(cloud1, 255, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud1, single_color1, "Cloud1", v1);
	int v2(0);
	viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	viewer->setBackgroundColor(0.3, 0.3, 0.3, v2);
	viewer->addText("Cloud2", 10, 10, "v2 text", v2);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color2(cloud2, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud2, single_color2, "Cloud2", v2);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Cloud1");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "Cloud2");
	viewer->addCoordinateSystem(1.0);


	return (viewer);
}

//void RestructionBox::cal_rigidtrans_emat(Mat cameraMatrix, Mat distCoeff)
//{
//	
//	for (int n = 0; n < orientation_num - 1; n++)
//	{
//		vector<Point2f> Pts1, Pts2;
//		Mat mask, r, t;
//		for (int i = 0; i < orients_matches[n].size(); i++)
//		{
//			Pts1.push_back(KeypointsL[n][orients_matches[n][i].queryIdx].pt);
//			Pts2.push_back(KeypointsL[n+1][orients_matches[n][i].trainIdx].pt);
//		}
//		Mat E = findEssentialMat(Pts2, Pts1, cameraMatrix, RANSAC, 0.999, 1.0, mask);
//		recoverPose(E, Pts2, Pts1, cameraMatrix, r, t, mask);
//		double Tlength = getTlength(n, r, mask);
//		R.push_back(r);
//		T.push_back(Tlength*t);
//	}
//}
//double RestructionBox::getTlength(int n, Mat R, Mat mask)
//{
//	double Tlength = 0;
//	Mat pts1, pts2;
//	vector<Point3f> pt3d1, pt3d2;
//	for (int j = 0; j < orients_matches[n].size(); j++)
//	{
//		if (mask.at<uchar>(j, 0))
//		{
//			int k = 0, l = 0;
//			for (; k < points_cloud[n].size(); k++)//找出第n个视角下的特征点对应的三维点的下标l，找到就跳出循环
//			{
//				if (points_cloud_index[n][k] == orients_matches[n][j].queryIdx)
//				{
//					break;
//				}
//			}
//
//			for (; l < points_cloud[n + 1].size(); l++)//找出第n+1个视角下的特征点对应的三维点的下标k，找到就跳出循环
//			{
//				if (points_cloud_index[n + 1][l] == orients_matches[n][j].trainIdx)
//				{
//					break;
//				}
//			}
//			if (k != points_cloud[n].size() && l != points_cloud[n + 1].size())//该特征点在第n和n+1个视角下必须是一个三维点，若不是则跳过该点
//			{
//				pt3d1.push_back(points_cloud[n][k]);
//				pt3d2.push_back(points_cloud[n + 1][l]);
//			}
//		}
//	}
//	pts1 = Mat(3, pt3d1.size(), CV_64FC1);
//	for (int m = 0; m < pt3d1.size(); m++)
//	{
//		pts1.at<double>(0, m) = pt3d1[m].x; pts1.at<double>(1, m) = pt3d1[m].y; pts1.at<double>(2, m) = pt3d1[m].z;
//	}
//	pts2 = Mat(3, pt3d2.size(), CV_64FC1);
//	for (int m = 0; m < pt3d2.size(); m++)
//	{
//		pts2.at<double>(0, m) = pt3d2[m].x; pts2.at<double>(1, m) = pt3d2[m].y; pts2.at<double>(2, m) = pt3d2[m].z;
//	}
//	
//	Mat a = R*pts2;
//	pts2=a.clone();
//	for (int j = 0; j < pts1.cols; j++)
//			Tlength += cv::norm(pts1.colRange(j, j + 1), pts2.colRange(j, j + 1));
//	return Tlength / pts1.cols;
//}