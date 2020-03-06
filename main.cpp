#include"stereo.h"
int main(int argc, char* argv[])
{
	Mat img; vector<Vec3i> tri;
	vector<Point2f> pts2DTex;
	vector<Point3f> pts3D;
	Point3f center3D;Vec3f size3D;

	CalibrationBox calib(11,8,640,480);
	calib.imagecropping();
	calib.cal_parameters();
	calib.show_parameter();
	calib.cal_rectifymap();
	Mat Lx = calib.get_mapLx(); Mat Ly = calib.get_mapLy(); Mat Rx = calib.get_mapRx(); Mat Ry = calib.get_mapRy();
	RestructionBox rest(Lx, Ly, Rx, Ry, calib.get_Q());
	rest.loadimg();
	rest.rectify_img();

	//rest.sgbm_match();
	rest.feature_extract();
	rest.show_matches();
	rest.cal_3dpoints();
	rest.cal_rigidtrans_SVD();
	rest.trans_3dpoints();
	rest.point_cloud_process();
	//rest.triangle_subdiv();
	rest.get_GLparam(img, tri, pts2DTex, pts3D, center3D, size3D);


	/************************************************************************/
	/* Draw 3D scene using OpenGL                                           */
	/************************************************************************/
	glutInit(&argc, argv); // must be called first in a glut program

	InitGl(); // must be called first in a glut program

	cout << getTimeStamp() << "creating 3D texture..." << endl;
	GLuint tex = Create3DTexture(img, tri, pts2DTex, pts3D, center3D, size3D);
	Show(tex, center3D, size3D);
	system("pause");
	return 0;
}