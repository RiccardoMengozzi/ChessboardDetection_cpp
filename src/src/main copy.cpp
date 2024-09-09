#include <iostream>
#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <filesystem>


struct ChessboardType {
    std::vector<cv::Point2f> corners;
    bool found;
    ChessboardType(size_t length) : corners(length), found(false) {
        std::fill(corners.begin(), corners.end(), cv::Point2f(0,0));
    }
};


int sumVectorValues(std::vector<int> vec) {
    int sum = 0;
    for (int i = 0; i < vec.size(); ++i) {
        sum += vec[i];
    }
    return sum;
}


std::vector<int> computeHistogram(cv::Mat img) {
    std::vector<int> histogram(256);
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            int pixel = img.at<uchar>(i, j);
            histogram[pixel] += 1;
        }
    }
    return histogram;
}

int findPercentileValue(std::vector<int> histogram, int percentile) {
    int s = 0;
    int idx = 0;
    int total_pixels = sumVectorValues(histogram);
    while (s < total_pixels * percentile / 100) {
        s += histogram[idx];
        idx += 1;
    }
    return idx;
}

cv::Mat linearStretching(cv::Mat img, int max_value, int min_value) {
    cv::Mat linear_stretched_img = img.clone();

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (static_cast<int>(img.at<uchar>(i,j)) < min_value) {
                linear_stretched_img.at<uchar>(i,j) = min_value; 
            }
            else if (static_cast<int>(img.at<uchar>(i,j)) > max_value) {
                linear_stretched_img.at<uchar>(i,j) = max_value; 
            }
            else {


                linear_stretched_img.at<uchar>(i,j) = 255 / (max_value - min_value) * (static_cast<int>(img.at<uchar>(i,j)) - min_value); 
            }
        }
    }
    return linear_stretched_img;
}

cv::Mat stretchImg(cv::Mat img, std::vector<int> histogram, int min_percentile, int max_percentile) {
    int min_value = findPercentileValue(histogram, min_percentile);
    int max_value = findPercentileValue(histogram, max_percentile);
    cv::Mat stretched_img = linearStretching(img, max_value, min_value);
    return stretched_img;
}

std::vector<std::vector<std::pair<int,int>>> createGridIndices(std::pair<int,int> pattern_size) {
    std::vector<std::vector<std::pair<int,int>>> indices(pattern_size.first, std::vector<std::pair<int, int>>(pattern_size.second, std::make_pair(0, 0)));
    for (int i = 0; i < pattern_size.first; i++) {
        for (int j = 0; j < pattern_size.second; j++) {
            indices[i][j] = std::make_pair(i,j);
        }
    }
    return indices;
}

std::vector<std::vector<std::pair<float,float>>> createGridCoordinates(std::vector<std::vector<std::pair<int,int>>> grid_indices, std::pair<float,float> square_size) {
    std::vector<std::vector<std::pair<float,float>>> coordinates(grid_indices.size(), std::vector<std::pair<float, float>>(grid_indices[0].size(), std::make_pair(0, 0)));
    for (int i = 0; i < coordinates.size(); i++) {
        for (int j = 0; j < coordinates[0].size(); j++) {
            coordinates[i][j].first = static_cast<float>(grid_indices[i][j].first) * square_size.first;
            coordinates[i][j].second = static_cast<float>(grid_indices[i][j].second) * square_size.second;
        }
    }
    return coordinates;
}

std::vector<cv::Point2f> flattenGridCoordinates(std::vector<std::vector<std::pair<float,float>>> grid_coordinates) {
    std::vector<cv::Point2f> flatten_coordinates_cv(grid_coordinates.size() * grid_coordinates[0].size(), cv::Point2f(0,0));
    for (int i = 0; i < grid_coordinates.size(); i++) {
        for (int j = 0; j < grid_coordinates[0].size(); j++) {
            flatten_coordinates_cv[(i * grid_coordinates[0].size() + 1) + j].x = grid_coordinates[i][j].first;
            flatten_coordinates_cv[(i * grid_coordinates[0].size() + 1) + j].y = grid_coordinates[i][j].second;
        }
    }
    return flatten_coordinates_cv;
}


ChessboardType findChessboard(cv::Mat img, cv::Size pattern_size) {
    ChessboardType chess(pattern_size.area());
    cv::Mat vis = img.clone();
    chess.found = cv::findChessboardCorners(img, pattern_size, chess.corners);
    if (chess.found) {
        // std::cout << "CHESSBOARD FOUND!!!" << std::endl;
        // Optional: Refine corner positions
        cv::cornerSubPix(img, chess.corners, cv::Size(5, 5), cv::Size(-1, -1),
                        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
    } else {
        // std::cout << "Chessboard pattern not found!" << std::endl;
    }
    return chess;
}

cv::Point2f computeChessboardCenter(ChessboardType chess) {
    cv::Point2i center(0,0);
    float sum_x = 0;
    float sum_y = 0;
    for(int i = 0; i < chess.corners.size(); i++) {
        sum_x += chess.corners[i].x;
        sum_y += chess.corners[i].y;
    }
    center.x = sum_x / chess.corners.size();
    center.y = sum_y / chess.corners.size();
    return center;
}


ChessboardType correctCroppedCoordinates(ChessboardType chess, cv::Rect crop_roi) {
    ChessboardType new_chess(chess.corners.size());
    for (int i = 0; i < chess.corners.size(); i++) {
        new_chess.corners[i].y = chess.corners[i].y - crop_roi.y;
        new_chess.corners[i].x = chess.corners[i].x;
    }
    return new_chess;
} 

cv::Mat computeDisparityMap(cv::Mat imgL, cv::Mat imgR, int offset, int num_disparities, int block_size) {


        // Create StereoSGBM object
    cv::Ptr<cv::StereoSGBM> SGBM = cv::StereoSGBM::create();
    SGBM->setBlockSize(block_size);
    SGBM->setNumDisparities(num_disparities);
    SGBM->setPreFilterCap(63);
    SGBM->setMinDisparity(offset);
    SGBM->setUniquenessRatio(15);
    SGBM->setSpeckleWindowSize(100);
    SGBM->setSpeckleRange(32);
    SGBM->setDisp12MaxDiff(1);
    SGBM->setP1(8*block_size*block_size);
    SGBM->setP2(32*block_size*block_size);
    SGBM->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
    // Compute the disparity map
    cv::Mat disparity;
    SGBM->compute(imgL, imgR, disparity);
    // Convert disparity to float for further processing
    disparity.convertTo(disparity, CV_32F);
    disparity = disparity / 16
;

    return disparity;
}


float computeMainDisparity(cv::Mat disparityMap, cv::Point2i center, int wind_radius) {
    cv::Rect roi(center.x - wind_radius, center.y - wind_radius, wind_radius * 2, wind_radius * 2);
    std::cout << "roi_x" << roi.x << std::endl;
    std::cout << "roi_y" << roi.y << std::endl;
    std::cout << "roi_width" << roi.width << std::endl;
    std::cout << "roi_height" << roi.height << std::endl;
    std::cout << "rows" << disparityMap.rows << std::endl;
    std::cout << "cols" << disparityMap.cols << std::endl;
    float sum = 0;
    float dmain = 0;
    for (int i = roi.x; i < roi.x + roi.width; i++) {
        for (int j = roi.y; j < roi.y + roi.height; j++) {
            sum += static_cast<float>(disparityMap.at<uchar>(j,i));
            std::cout << "value at [" << j << "," << i << "] = " << static_cast<float>(disparityMap.at<uchar>(j,i)) << std::endl;
        }
    }
    dmain = sum / roi.area();
    std::cout << "sum = " << sum << std::endl;
    // std::cout << "dmain = " << dmain << std::endl;
    return dmain;
}

float computeDistance(float dmain, float focal, float baseline) {
    return focal * baseline / dmain / 100.0;
}

int main() {   
    //////////////// PARAMETERS //////////////////
    // Get current folder directory
    std::filesystem::path dir_path = std::filesystem::current_path();
    // Size of plot window
    const int wind_width = 320;
    const int wind_height = 240;

    std::pair<const int,const int> pattern_size = std::make_pair(8,6);
    cv::Size pattern_size_cv(pattern_size.first, pattern_size.second);
    std::pair<const float,const float> squares_size = std::make_pair(178.0 / pattern_size.first, 125.0 / pattern_size.second);

    const std::string side_L = "L";
    const std::string side_R = "R";

    const int min_percentile = 25;
    const int max_percentile = 88;

    const int min_h = 200;
    const int max_h = 400;
    const int min_w = 0;
    const int max_w = 640;

    const cv::Point2f roi_origin(0,200);
    const int roi_width = 640;
    const int roi_height = 240;

    // Parameters for StereoSGBM

    const int num_disparities = 16 * 4;
    const int blockSize = 11;
    int offset = 0;
    const int wind_radius = 20;

    const float focal = 567.2;
    const float baseline = 92.226;

    bool first_chess_found = false;


    std::cout << "square_size = [" << squares_size.first << "," << squares_size.second << "]"  << std::endl;


    // variable to keep track of current frame
    int i = 0;

    bool checkL, checkR;
    cv::Mat frameL, frameR;
    cv::Mat grayL, grayR;
    ChessboardType previous_chess(pattern_size_cv.area());
    bool is_copied = false;

    // variables to save previous value of chessboard corners
    //previous_chessL = None
    //previous_chessR = None
    
    // Initialize interface windows
    cv::namedWindow("ImageL", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("ImageL", 225, 100);
    cv::namedWindow("ImageR", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("ImageR", 875, 100);

    // Capture left video
    cv::VideoCapture videoL = cv::VideoCapture(dir_path / "robotL.avi");
    videoL.set(cv::CAP_PROP_FRAME_WIDTH, wind_width);
    videoL.set(cv::CAP_PROP_FRAME_HEIGHT, wind_height);

    // Capture right video
    cv::VideoCapture videoR = cv::VideoCapture(dir_path / "robotR.avi");
    videoR.set(cv::CAP_PROP_FRAME_WIDTH, wind_width);
    videoR.set(cv::CAP_PROP_FRAME_HEIGHT, wind_height);

    


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

        std::vector<int> histogramL = computeHistogram(grayL);
        std::vector<int> histogramR = computeHistogram(grayR);

        // Stretch the left and right grayscale images
        cv::Mat stretched_grayL = stretchImg(grayL, histogramL, min_percentile, max_percentile);
        cv::Mat stretched_grayR = stretchImg(grayR, histogramR, min_percentile, max_percentile);

        std::vector<std::vector<std::pair<int,int>>> grid_indices = createGridIndices(pattern_size);
        std::vector<std::vector<std::pair<float,float>>> grid_coordinates = createGridCoordinates(grid_indices, squares_size);
        // for (int i = 0; i < grid_coordinates.size(); i++) {
        //     for (int j = 0; j < grid_coordinates[0].size(); j++) {
        //         std::cout << "pixel[" << i << "," << j << "] index = [" << grid_coordinates[i][j].first << "," << grid_coordinates[i][j].second << "]" << std::endl;
        //     }
        // }
        std::vector<cv::Point2f> flatten_coordinates_cv = flattenGridCoordinates(grid_coordinates);
        // for (int i = 0; i < flatten_coordinates_cv.size(); i++) {
        //     std::cout << "pixel[" << i << "] index = [" << flatten_coordinates_cv[i].x << "," << flatten_coordinates_cv[i].y << "]" << std::endl;
        // }
        ChessboardType chess(pattern_size_cv.area());
        chess = findChessboard(stretched_grayL, pattern_size_cv); 

        if (chess.found) {
            previous_chess.corners = chess.corners;
            first_chess_found = true;
            is_copied = false;
        }
        else {
            chess.corners = previous_chess.corners;
            is_copied = true;
        }


        cv::Rect crop_roi(roi_origin.x, roi_origin.y, roi_width, roi_height);

        cv::Mat cropL = grayL.clone();
        cv::Mat cropR = grayR.clone();

        cropL = cropL(crop_roi);
        cropR = cropR(crop_roi);

        if (!is_copied) {
            chess = correctCroppedCoordinates(chess, crop_roi);
        }


        cv::Point2i center = computeChessboardCenter(chess);


        cv::Mat disparityMap = computeDisparityMap(cropL, cropR, offset, num_disparities, blockSize);

        cv::Mat disparityMapNormalized;
        cv::normalize(disparityMap, disparityMapNormalized, 0, 255, cv::NORM_MINMAX);
        disparityMapNormalized.convertTo(disparityMapNormalized, CV_8U);
        if (first_chess_found) {
            float dmain = computeMainDisparity(disparityMap, center, wind_radius);
        
            float distance = computeDistance(dmain, focal, baseline);
            std::cout << "dmain = " << dmain << std::endl;
            std::cout << "distance = " << distance << std::endl;
        }
        cv::Mat visL = disparityMapNormalized.clone();
        cv::Mat visR = cropR.clone();       
        // Draw the corners on the image
        // cv::drawChessboardCorners(vis, pattern_size_cv, chess.corners, chess.found);
    // Draw the point as a circle
        cv::circle(visL, center, 5, (255,255,255), -1); // -1 means filled circle

        cv::imshow("ImageL", visL);
        cv::waitKey(1);
        cv::imshow("ImageR", visR);
        cv::waitKey(1);

    }

    return 0;
}
