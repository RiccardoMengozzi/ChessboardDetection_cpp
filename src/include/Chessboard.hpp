#ifndef CHESSBOARD_HPP
#define CHESSBOARD_HPP

#include <opencv2/opencv.hpp>

/**
 * @class Chessboard
 * @brief Class for handling chessboard detection in images.
 * 
 * The class uses OpenCV functions to perform chessboard detection and corner refinement.
 */
class Chessboard {
    public:
        /**
         * @brief Default constructor for Chessboard class.
         * Initializes pattern_size and squares_size to default values.
         */
        Chessboard();
        ~Chessboard();
        /**
         * @brief Sets the pattern size of the chessboard.
         * 
         * @param pattern_size The pattern size (number of inner corners per chessboard row and column).
         */
        void setPatternSize(cv::Size pattern_size);
        /**
         * @brief Sets the size of each square on the chessboard.
         * 
         * @param squares_size The size of each square in floating-point precision.
         */
        void setSquaresSize(cv::Size2f squares_size);
        /**
         * @brief Sets the image in which the chessboard will be detected.
         * 
         * @param img The image as a cv::Mat object.
         */
        void setImage(cv::Mat img);
        /**
         * @brief Retrieves the pattern size of the chessboard.
         * 
         * @return The pattern size as a cv::Size object.
         * @throws runtime_error if pattern_size has not been set.
         */
        cv::Size getPatternSize();
        /**
         * @brief Retrieves the size of each square on the chessboard.
         * 
         * @return The size of each square as a cv::Size2f object.
         * @throws runtime_error if squares_size has not been set.
         */
        cv::Size2f getSquaresSize();
        /**
         * @brief Retrieves the image in which the chessboard was detected.
         * 
         * @return The image as a cv::Mat object.
         * @throws runtime_error if the image has not been set.
         */
        cv::Mat getImage();
        /**
         * @brief Checks if the chessboard was found in the image.
         * 
         * @return True if the chessboard was detected; false otherwise.
         */
        bool isFound();
        /**
         * @brief Detects the chessboard corners in the image.
         * 
         * @param vis If true, displays the image with detected corners.
         * @throws runtime_error if the image, pattern size, or squares size is not set.
         */

        cv::Point2i getCenter();
        std::vector<cv::Point2f> getCorners();
        void detect(bool vis = false,
                               const cv::Point2i image_position = cv::Point2i(0, 0),
                               const std::string image_name = "ChessboardVis");   
        // std::vector<cv::Point2f> createChessboardCoordinates();

        void computeCenter(bool vis = false,
                           const cv::Point2i image_position = cv::Point2i(0, 0), 
                           const std::string image_name = "CenterVis");
    private:
        cv::Size pattern_size;
        cv::Size2f squares_size;
        std::vector<cv::Point2f> corners;
        bool found;
        cv::Point2i center;
        bool center_computed = false;
        cv::Mat img;
        enum class CheckType {
            EmptyImage,
            PatternSize,
            SquaresSize,
            Center,
            Corners,
        };
        void check(CheckType type, const char* file, int line) const ;

};

#endif 