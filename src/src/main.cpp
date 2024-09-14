#include <iostream>
#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <filesystem>

#include <yaml-cpp/yaml.h>
#include <Chessboard.hpp>
#include <Disparity.hpp>
#include <Image.hpp>



class VideoProcessing {
    private:
        YAML::Node config;
        std::filesystem::path left_video_directory, right_video_directory, config_directory;
        cv::Point2i vis_left_window_position, vis_right_window_position;
        cv::Point2i chessboard_corners_window_position, center_window_position;
        cv::Point2i  disparity_window_position, main_disparity_roi_window_position;

        bool chessboard_corners_vis, center_vis, disparity_vis, main_disparity_roi_vis;
        int min_percentile, max_percentile;

        cv::VideoCapture videoL;  // Capture for left video
        cv::VideoCapture videoR;  // Capture for right video
        bool checkL, checkR;
        cv::Mat frameL, frameR;
        cv::Mat grayL, grayR;
        cv::Mat stretched_img;
        cv::Mat cropped_img;
        std::vector<int> histogramL, histogramR;
        Image image;
        Chessboard chess;
        Disparity disparity;
        SGBMParam SGBM_parameters;
        cv::Size pattern_size;
        cv::Size2f squares_size;
        int chess_roi_x, chess_roi_y, chess_roi_width, chess_roi_height;
        int disp_roi_x, disp_roi_y, disp_roi_width, disp_roi_height;
        float dmain, distance;
        float focal;
        float baseline;
        int dmain_window_radius;

    public:
        VideoProcessing(std::filesystem::path left_video_directory,
                        std::filesystem::path right_video_directory,
                        std::filesystem::path config_directory) :   image(),
                                                                    chess(),
                                                                    disparity(),
                                                                    left_video_directory(left_video_directory),
                                                                    right_video_directory(right_video_directory),
                                                                    config_directory(config_directory) {}

        void run() {
            config = YAML::LoadFile(config_directory);

            vis_left_window_position.x = config["vis"]["vis_left_window_position"][0].as<int>();
            vis_left_window_position.y = config["vis"]["vis_left_window_position"][1].as<int>();
            vis_right_window_position.x = config["vis"]["vis_right_window_position"][0].as<int>();
            vis_right_window_position.y = config["vis"]["vis_right_window_position"][1].as<int>();
            chessboard_corners_window_position.x = config["vis"]["chessboard_corners_window_position"][0].as<int>();
            chessboard_corners_window_position.y = config["vis"]["chessboard_corners_window_position"][1].as<int>();
            center_window_position.x = config["vis"]["center_window_position"][0].as<int>();
            center_window_position.y = config["vis"]["center_window_position"][1].as<int>();
            disparity_window_position.x = config["vis"]["disparity_window_position"][0].as<int>();
            disparity_window_position.y = config["vis"]["disparity_window_position"][1].as<int>();
            main_disparity_roi_window_position.x = config["vis"]["main_disparity_roi_window_position"][0].as<int>();
            main_disparity_roi_window_position.y = config["vis"]["main_disparity_roi_window_position"][1].as<int>();

            chessboard_corners_vis = config["vis"]["chessboard_corners_vis"].as<bool>();
            center_vis = config["vis"]["center_corners_vis"].as<bool>();
            disparity_vis = config["vis"]["disparity_vis"].as<bool>();
            main_disparity_roi_vis = config["vis"]["main_disparity_roi_vis"].as<bool>();

            min_percentile = config["processing"]["min_percentile"].as<int>();
            max_percentile = config["processing"]["max_percentile"].as<int>(); 
            pattern_size.height = config["chessboard"]["pattern_size"][0].as<int>();
            pattern_size.width = config["chessboard"]["pattern_size"][1].as<int>();
            squares_size.height = config["chessboard"]["squares_size"][0].as<float>();
            squares_size.width = config["chessboard"]["squares_size"][1].as<float>();

            chess_roi_x = config["chessboard"]["roi"][0].as<int>();
            chess_roi_y = config["chessboard"]["roi"][1].as<int>();
            chess_roi_width = config["chessboard"]["roi"][2].as<int>();
            chess_roi_height = config["chessboard"]["roi"][3].as<int>();

            disp_roi_x = config["disparity"]["roi"][0].as<int>();
            disp_roi_y = config["disparity"]["roi"][1].as<int>();
            disp_roi_width = config["disparity"]["roi"][2].as<int>();
            disp_roi_height = config["disparity"]["roi"][3].as<int>();

            SGBM_parameters.block_size = config["disparity"]["SGBM"]["block_size"].as<int>();
            SGBM_parameters.num_disparities = config["disparity"]["SGBM"]["num_disparities"].as<int>();
            SGBM_parameters.min_disparity = config["disparity"]["SGBM"]["min_disparity"].as<int>();
            SGBM_parameters.offset = config["disparity"]["SGBM"]["offset"].as<int>();
            SGBM_parameters.p1 = config["disparity"]["SGBM"]["p1"].as<int>();
            SGBM_parameters.p2 = config["disparity"]["SGBM"]["p2"].as<int>();
            SGBM_parameters.mode = config["disparity"]["SGBM"]["mode"].as<int>();
            SGBM_parameters.speckle_range = config["disparity"]["SGBM"]["speckle_range"].as<int>();
            SGBM_parameters.speckle_window_size = config["disparity"]["SGBM"]["speckle_window_size"].as<int>();
            SGBM_parameters.pre_filter_cap = config["disparity"]["SGBM"]["pre_filter_cap"].as<int>();
            SGBM_parameters.uniqueness_ratio = config["disparity"]["SGBM"]["uniqueness_ratio"].as<int>();
            SGBM_parameters.disp_12_max_diff = config["disparity"]["SGBM"]["disp_12_max_diff"].as<int>();

            focal = config["disparity"]["focal"].as<float>();
            baseline = config["disparity"]["baseline"].as<float>();
            dmain_window_radius = config["disparity"]["window_radius"].as<int>();



            // Initialize interface windows
            cv::namedWindow("ImageL", cv::WINDOW_AUTOSIZE);
            cv::moveWindow("ImageL", vis_right_window_position.x, vis_right_window_position.y);
            cv::namedWindow("ImageR", cv::WINDOW_AUTOSIZE);
            cv::moveWindow("ImageR", vis_right_window_position.x, vis_right_window_position.y);

            // Capture videos
            videoL = cv::VideoCapture(left_video_directory);
            videoR = cv::VideoCapture(right_video_directory);

            while (videoL.isOpened() && videoR.isOpened()) {
            checkL = videoL.read(frameL);
            checkR = videoR.read(frameR);
                if (!checkL) {
                    videoL.release();
                    std::cout << "VideoL has been released" << std::endl;
                    break;
                }
                else if (!checkR) {
                    videoR.release();
                    std::cout << "VideoR has been released" << std::endl;
                    break;
                }

                // Convert to grayscale
                cv::cvtColor(frameL, grayL, cv::COLOR_BGR2GRAY);
                cv::cvtColor(frameR, grayR, cv::COLOR_BGR2GRAY);

                image.setImage(grayL);
                image.setRoi(chess_roi_x, chess_roi_y, chess_roi_width, chess_roi_height);
                // cropped_img = image.cropImage();

                // image.setImage(cropped_img);
                image.setStretchingBounds(min_percentile, max_percentile);
                
                stretched_img = image.stretchImage();

                chess.setPatternSize(pattern_size);
                chess.setSquaresSize(squares_size);
                chess.setImage(stretched_img);

                chess.detect(chessboard_corners_vis, chessboard_corners_window_position);

                chess.computeCenter(center_vis, center_window_position);

                disparity.setImages(grayL, grayR);
                disparity.setRoi(disp_roi_x, disp_roi_y, disp_roi_width, disp_roi_height);
                disparity.setFocalAndBaseline(focal, baseline);
                disparity.setDmainRoi(chess.getCenter(), dmain_window_radius);
                disparity.setSGBMParameters(SGBM_parameters);

                if (chess.isFound())
                    disparity.computeDisparityMap(disparity_vis, disparity_window_position);
                    dmain = disparity.computeMainDisparity(main_disparity_roi_vis, main_disparity_roi_window_position);
                    distance = disparity.computeDistance();

                    std::cout << "distance = " << distance << ", dmain = " << dmain << '\n';
            }       
        }
};





int main() {
    std::filesystem::path config_directory = std::filesystem::absolute("src/config/parameters.yaml");
    std::filesystem::path left_video_directory = std::filesystem::absolute("robotL.avi");
    std::filesystem::path right_video_directory = std::filesystem::absolute("robotR.avi");

    VideoProcessing videoProcessor(left_video_directory, right_video_directory, config_directory);
    videoProcessor.run();

    return 0;
}