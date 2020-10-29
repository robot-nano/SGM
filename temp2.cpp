
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

const string window_name = "用户界面";

#define USE_CAMERA
//#define USE_VIDEO

int main()
{
    Mat frame;

    double brightness = 0;		//亮度
    double contrast = 0;		//对比度
    double saturation = 0;		//饱和度
    double hue = 0;				//色调
    double gain = 0;			//增益
    double exposure = 0;		//曝光
    double white_balance = 0;	//白平衡

    double pos_msec = 0;		//当前视频位置(ms)
    double pos_frame = 0;		//从0开始下一帧的索引
    double pos_avi_ratio = 0;	//视频中的相对位置(范围为0.0到1.0)
    double frame_width = 0;		//视频帧的像素宽度
    double frame_height = 0;	//视频帧的像素高度
    double fps = 0;				//帧速率
    double frame_count = 0;		//视频总帧数
    double video_duration = 0.00;	//视频时长
    double format = 0;			//格式

#ifdef USE_VIDEO
    const string file_name = "201910915314.avi";
	VideoCapture capture(file_name);
	
	frame_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
	frame_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
	fps = capture.get(cv::CAP_PROP_FPS);
	frame_count = capture.get(cv::CAP_PROP_FRAME_COUNT);
	format = capture.get(cv::CAP_PROP_FORMAT);
	pos_avi_ratio = capture.get(cv::CAP_PROP_POS_AVI_RATIO);
	video_duration = frame_count / fps;
 
	cout << "---------------------------------------------" << endl;
	cout << "视频中的相对位置(范围为0.0到1.0):" << pos_avi_ratio << endl;
	cout << "视频帧的像素宽度:" << frame_width << endl;
	cout << "视频帧的像素高度:" << frame_height << endl;
	cout << "录制视频的帧速率(帧/秒):" << fps << endl;
	cout << "视频文件总帧数:" << frame_count << endl;
	cout << "图像的格式:" << format << endl;
	cout << "视频时长:" << video_duration << endl;
	cout << "---------------------------------------------" << endl;
#endif // USE_VIDEO

#ifdef USE_CAMERA
    VideoCapture capture(0);
    brightness = capture.get(cv::CAP_PROP_BRIGHTNESS);
    contrast= capture.get(cv::CAP_PROP_CONTRAST);
    saturation = capture.get(cv::CAP_PROP_SATURATION);
    hue = capture.get(cv::CAP_PROP_HUE);
    gain = capture.get(cv::CAP_PROP_GAIN);
    exposure = capture.get(cv::CAP_PROP_EXPOSURE);
    white_balance = capture.get(cv::CAP_PROP_WHITE_BALANCE_BLUE_U);

    std::cout << "---------------------------------------------" << endl;
    std::cout << "摄像头亮度：" << brightness << endl;
    std::cout << "摄像头对比度：" << contrast << endl;
    std::cout << "摄像头饱和度：" << saturation << endl;
    std::cout << "摄像头色调：" << hue << endl;
    std::cout << "摄像头增益：" << gain << endl;
    std::cout << "摄像头曝光度：" << exposure << endl;
    std::cout << "摄像头白平衡：" << white_balance << endl;
    std::cout << "---------------------------------------------" << endl;
#endif // USE_CAMERA

    namedWindow(window_name,WINDOW_AUTOSIZE);
    while (capture.isOpened())
    {
        capture >> frame;

#ifdef USE_VIDEO
        pos_msec = capture.get(cv::CAP_PROP_POS_MSEC);
		pos_frame = capture.get(cv::CAP_PROP_POS_FRAMES);
		pos_avi_ratio = capture.get(cv::CAP_PROP_POS_AVI_RATIO);
		cout << "---------------------------------------------" << endl;
		cout << "视频文件中当前位置(ms):" << pos_msec << endl;
		cout << "从0开始下一帧的索引:" << pos_frame << endl;
		cout << "视频中的相对位置(范围为0.0到1.0):" << pos_avi_ratio << endl;
		cout << "---------------------------------------------" << endl;
#endif // USE_VIDEO

        imshow(window_name, frame);
        if (waitKey(60)==27)
        {
            break;
        }
    }
    capture.release();
    destroyAllWindows();
    return 0;
}