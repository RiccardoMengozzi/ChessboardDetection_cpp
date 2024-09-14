#ifndef DISTANCE_HPP
#define DISTANCE_HPP

#include <opencv2/opencv.hpp>

#define _3WAY 1

struct SGBMParam {
    int block_size; 
    int num_disparities;
    int min_disparity;
    int offset;
    int p1;
    int p2;
    int mode;
    int speckle_range;
    int speckle_window_size;
    int pre_filter_cap;
    int uniqueness_ratio;
    int disp_12_max_diff;
};

class Disparity {
    public:
        Disparity();
        ~Disparity();
        void setImages(cv::Mat imgL, cv::Mat imgR);
        void setSGBMParameters(SGBMParam parameters);
        void setRoi(cv::Rect roi);
        void setRoi(int roi_x, int roi_y, int roi_width, int roi_height);
        void setDmainRoi(cv::Point2i center, int window_radius) ;
        void setFocalAndBaseline(float focal, float baseline);
        cv::Mat getLeftImage();
        cv::Mat getRightImage();
        SGBMParam getSGBMParameters();
        cv::Rect getRoi();
        cv::Rect getDmainRoi();
        float getFocal();
        float getBaseline();

        cv::Mat computeDisparityMap(bool vis = false,
                                    const cv::Point2i image_position = cv::Point2i(0, 0), 
                                    const std::string image_name = "disparityMap");
        float computeMainDisparity(bool vis = false,
                                   const cv::Point2i image_position = cv::Point2i(0, 0),
                                   const std::string image_name = "mainDisparityROI");
        float computeDistance();


    private:
        cv::Mat imgL, imgR, disparityMap;
        cv::Ptr<cv::StereoSGBM> SGBM = cv::StereoSGBM::create();
        SGBMParam parameters;
        float focal, baseline;
        float dmain;
        cv::Rect dmain_roi;
        cv::Rect roi;
        enum class CheckType {
            EmptyImages,
            SGBMParameters,
            Roi,
            EmptyDisparityMap,
            DmainRoi,
            FocalAndBaseline,
        };
        bool param_set = false;
        bool focal_and_baseline_set = false;
        void check(CheckType type, const char* file, int line) const ;
        void assignSGBMParameters();

};


#endif