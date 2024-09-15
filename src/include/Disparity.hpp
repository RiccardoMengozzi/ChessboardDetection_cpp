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
        /**
         * @brief Default constructor for the Disparity class.
         */
        Disparity();
        /**
         * @brief Destructor for the Disparity class.
         */
        ~Disparity();
        /**
         * @brief Sets the left and right stereo images for disparity computation.
         * @param imgL The left image.
         * @param imgR The right image.
         */
        void setImages(cv::Mat imgL, cv::Mat imgR);
        /**
         * @brief Sets the parameters for the SGBM (Semi-Global Block Matching) algorithm.
         * @param parameters Struct containing the SGBM parameters.
         */
        void setSGBMParameters(SGBMParam parameters);
        /**
         * @brief Sets the Region of Interest (ROI) for the disparity computation using a cv::Rect.
         * @param roi The region of interest.
         */
        void setRoi(cv::Rect roi);
        /**
         * @brief Sets the Region of Interest (ROI) for disparity computation using individual coordinates.
         * @param roi_x X-coordinate of the ROI.
         * @param roi_y Y-coordinate of the ROI.
         * @param roi_width Width of the ROI.
         * @param roi_height Height of the ROI.
         */
        void setRoi(int roi_x, int roi_y, int roi_width, int roi_height);
        /**
         * @brief Sets the main region of interest (ROI) for disparity computation centered around a point.
         * @param center The center point of the ROI.
         * @param window_radius The radius of the window around the center.
         */
        void setDmainRoi(cv::Point2i center, int window_radius);
        /**
         * @brief Sets the focal length and baseline for computing distances from the disparity map.
         * @param focal The focal length of the camera.
         * @param baseline The baseline distance between the stereo cameras.
         */
        void setFocalAndBaseline(float focal, float baseline);
        /**
         * @brief Returns the left stereo image.
         * @return The left image as a cv::Mat.
         */
        cv::Mat getLeftImage();
        /**
         * @brief Returns the right stereo image.
         * @return The right image as a cv::Mat.
         */
        cv::Mat getRightImage();
        /**
         * @brief Returns the parameters for the SGBM algorithm.
         * @return The SGBM parameters.
         */
        SGBMParam getSGBMParameters();
        /**
         * @brief Returns the Region of Interest (ROI).
         * @return The ROI as a cv::Rect.
         */
        cv::Rect getRoi();
        /**
         * @brief Returns the main region of interest (ROI) for disparity computation.
         * @return The main ROI as a cv::Rect.
         */
        cv::Rect getDmainRoi();
        /**
         * @brief Returns the focal length.
         * @return The focal length.
         */
        float getFocal();
        /**
         * @brief Returns the baseline.
         * @return The baseline.
         */
        float getBaseline();
        /**
         * @brief Computes the disparity map for the stereo images.
         * Optionally visualizes the result.
         * @param vis Flag to visualize the disparity map.
         * @param image_position Position of the image window.
         * @param image_name Name of the image window.
         * @return The computed disparity map as a cv::Mat.
         */
        cv::Mat computeDisparityMap(bool vis = false,
                                    const cv::Point2i image_position = cv::Point2i(0, 0), 
                                    const std::string image_name = "disparityMap");
        /**
         * @brief Computes the average disparity value in the main ROI.
         * Optionally visualizes the main ROI.
         * @param vis Flag to visualize the ROI on the disparity map.
         * @param image_position Position of the image window.
         * @param image_name Name of the image window.
         * @return The average disparity value within the main ROI.
         */
        float computeMainDisparity(bool vis = false,
                                   const cv::Point2i image_position = cv::Point2i(0, 0),
                                   const std::string image_name = "mainDisparityROI");
        /**
         * @brief Computes the distance from the stereo setup using the disparity map, focal length, and baseline.
         * @return The computed distance in meters.
         */
        float computeDistance();
        /**
         * @brief Computes the real-world size of a chessboard based on its detected vertices and the stereo setup parameters.
         * @param vertices The vertices of the chessboard.
         * @param pattern_size The size of the chessboard pattern.
         * @return The real-world size of the chessboard as a cv::Size2f.
         */
        cv::Size2f computeChessboardSize(std::vector<cv::Point2i> vertices, cv::Size parttern_size);




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
        float distance;
        cv::Size2f chessSize;
        void check(CheckType type, const char* file, int line) const ;
        void assignSGBMParameters();

};


#endif