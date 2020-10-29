////
//// Created by wserver on 2020/10/25.
////
//
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <iostream>
//
//int main(int argc, char **argv) {
//    cv::Mat frame;
//    cv::VideoCapture cap;
//    int deviceID = 0;
//    int apiID = cv::CAP_ANY;
//    cap.open(deviceID, apiID);
//
//    if (!cap.isOpened()) {
//        std::cerr << "Error! " << std::endl;
//        return -1;
//    }
//
//    while (true) {
//        cap.read(frame);
//        if (frame.empty()) {
//            std::cerr << "Error! empty frame; " << std::endl;
//            break;
//        }
//
//        cv::imshow("image", frame);
//        cv::waitKey(1);
////        if (cv::waitKey(5) >= 0)
////            break;
//    }
//
//    return 0;
//}

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "disparity.h"
#include <ctime>

int main(int argc, char **argv) {
    uint8_t p1, p2;
    p1 = atoi(argv[1]);
    p2 = atoi(argv[2]);

    init_disparity_method(p1, p2);

    cv::Mat frame;
    cv::VideoCapture cap;
    int deviceID = 0;
    int apiID = cv::CAP_ANY;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 2560);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cv::waitKey(2);
    cap.open(deviceID);

    if (!cap.isOpened()) {
        std::cerr << "Error! " << std::endl;
        return -1;
    }

    while (true) {
        cap.read(frame);
        if (frame.empty()) {
            std::cerr << "Error! empty frame; " << std::endl;
        }
        cv::Mat im, im2;
        im = frame(cv::Rect(0, 0, 320, 240));
        im2 = frame(cv::Rect(320, 0, 320, 240));
        cv::imshow("left", im);
        cv::imshow("right", im2);

        if (cv::waitKey(2) == 27)
            break;


    if (im.channels() > 1) {
        cv::cvtColor(im, im, CV_RGB2GRAY);
    }
    if (im2.channels() > 1) {
        cv::cvtColor(im2, im2, CV_RGB2GRAY);
    }

    if (im.rows % 4 || im.cols % 4) {
        cv::resize(im, im, cv::Size(im.cols / 4 * 4, im.rows / 4 * 4));
        cv::resize(im2, im2, cv::Size(im.cols / 4 * 4, im.rows / 4 * 4));
    }

    compute_disparity_method(im, im2);
    }
}
