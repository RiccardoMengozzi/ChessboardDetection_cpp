#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <opencv2/opencv.hpp>


/**
 * @class Image
 * @brief A class for processing images.
 *
 * This class provides methods to set and process images using OpenCV.
 */
class Image {
    public:
        /**
         * @brief Default constructor.
         *
         * Initializes the image member with an empty cv::Mat.
         */
        Image();
        /**
         * @brief Constructor with an image parameter.
         *
         * Initializes the image member with the provided cv::Mat object.
         *
         * @param img The cv::Mat object representing the image to initialize.
         */
        Image(const cv::Mat& img);
        /**
         * @brief Sets the image for processing.
         *
         * @param img The cv::Mat object representing the image to set.        __cpp_enumerator_attributes

         */
        void setImage (cv::Mat img);
        /**
         * @brief Retrieves the current image.
         *
         * @return A cv::Mat object containing the current image.
         */
        cv::Mat getImage();
        /**
         * @brief Sets the stretching bounds.

         * @param min The minimum percentile value for stretching. Should be between 0 and 100.
         * @param max The maximum percentile value for stretching. Should be between 0 and 100.
         */
        void setStretchingBounds(int min, int max);
        /**
         * @brief Sets the stretching bounds.

         * @param bounds A vector of two integers representing the minimum and maximum percentile
         *               values for stretching. Both values should be between 0 and 100.
         */
        void setStretchingBounds(std::vector<int> bounds);
        /**
         * @brief Retrieves the current stretching bounds.
         * 
         * @return A vector of two integers representing the current minimum and maximum percentile
         *         values for stretching.
         */
        void setRoi(cv::Rect roi);
        void setRoi(int roi_x, int roi_y, int roi_width, int roi_height);
        cv::Rect getRoi();
        std::vector<int> getStretchingBounds();
        /**
         * @brief Computes the histogram of a grayscale image.
         *
         * This function calculates the histogram of a grayscale image, which is a distribution of pixel intensity values.
         * The histogram represents the frequency of each intensity level (0-255) in the image.
         *
         * @param img The input grayscale image for which to compute the histogram. The image must be of type `CV_8UC1` (8-bit single-channel).
         * @return A vector of 256 integers, where each element represents the count of pixels with a specific intensity value.
         *         The index of the vector corresponds to the intensity value (0-255), and the value at that index represents
         *         the frequency of that intensity in the image.
         *
         * @note The image should be in grayscale format. For color images, convert to grayscale first using `cv::cvtColor`.
         */
        std::vector<int> computeHistogram();
        /**
         * @brief Enhances the contrast of the grayscale image by applying linear contrast stretching.
         * @param minmax_values A vector containing two integers that represent the minimum and maximum intensity values.
         * @param histogram A vector of integers where each entry represents the count of pixels with a specific intensity.
         * @return cv::Mat The contrast-stretched image.
         */
        cv::Mat stretchImage();
        cv::Mat cropImage();


        ~Image();

    private:
        cv::Mat image;
        int min_percentile;
        int max_percentile;
        bool bounds_set = false;
        cv::Rect roi;

        /**
         * @brief Checks the validity of the current image.
         * 
         * @throws std::runtime_error If the image is empty or invalid.
         */
        enum class CheckType {
            EmptyImage,
            StretchingBounds,
        };
        void check(CheckType type) const;
        std::vector<int> findPercentileValues(std::vector<int> histogram);
        cv::Mat linearStretching(std::vector<int> minmax_values);
};

#endif