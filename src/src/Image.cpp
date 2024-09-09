#include <Image.hpp>
#include <numeric>

#define THROW_RUNTIME_ERROR(msg) throw std::runtime_error(std::string("Error: ") + msg + " at " + __FILE__ + ":" + std::to_string(__LINE__))


Image::Image() :    image(cv::Mat()),
                    min_percentile(0),
                    max_percentile(100) {}

Image::Image(const cv::Mat& img) : image(img.clone()) {}

void Image::check(CheckType type) const {
    switch(type)
    {
        case CheckType::EmptyImage:
            if (this->image.empty()) {
                THROW_RUNTIME_ERROR("Error: The image is empty! Please set an image with setImage().");
            }
        case CheckType::StretchingBounds:
            if (!this->bounds_set) {
                THROW_RUNTIME_ERROR("Error: Stretching percentile bounds have not been set yet. Please use setStretchingBounds().");
            }
    }
}

void Image::setImage(cv::Mat img) {
    this->image = img.clone();
}

cv::Mat Image::getImage() {
    check(CheckType::EmptyImage);
    return this->image;
}

void Image::setStretchingBounds(int min, int max) {
    this->bounds_set = true;
    this->min_percentile = min;
    this->max_percentile = max;
}

void Image::setStretchingBounds(std::vector<int> bounds) {
    this->bounds_set = true;
    this->min_percentile = bounds.at(0);
    this->min_percentile = bounds.at(1);
}

std::vector<int> Image::getStretchingBounds() {
    check(CheckType::StretchingBounds);
    return {this->min_percentile, this->max_percentile};
}

void Image::setRoi(cv::Rect roi) {
    this->roi = roi;
}

void Image::setRoi(int roi_x, int roi_y, int roi_width, int roi_height) {
    this->roi.x = roi_x;
    this->roi.y = roi_y;
    this->roi.width = roi_width;
    this->roi.height = roi_height;
}

cv::Rect Image::getRoi() {
    return this->roi;
}

std::vector<int> Image::findPercentileValues(std::vector<int> histogram) {
    check(CheckType::StretchingBounds);
    std::vector<int> idxs(2);
    int s = 0;
    int idx_min = 0;
    int idx_max = 0;
    int total_pixels = std::accumulate(histogram.begin(), histogram.end(), 0);

    while (s < total_pixels * min_percentile / 100 && idx_min < histogram.size()) {
        s += histogram[idx_min];
        ++idx_min;
    }
    idxs.at(0) = idx_min - 1;  // Adjust because idx_min increments after satisfying the condition

    s = 0;
    while (s < total_pixels * max_percentile / 100 && idx_max < histogram.size()) {
        s += histogram[idx_max];
        ++idx_max;
    }
    idxs.at(1) = idx_max - 1;
    return idxs;
}

cv::Mat Image::linearStretching(std::vector<int> minmax_values) {
    cv::Mat stretched_img;
    image.convertTo(stretched_img, CV_64F);  // Convert to double precision to perform stretching
    
    // Apply min and max value constraints
    cv::threshold(stretched_img, stretched_img, minmax_values.at(0), minmax_values.at(0), cv::THRESH_TOZERO);
    cv::threshold(stretched_img, stretched_img, minmax_values.at(1), minmax_values.at(1), cv::THRESH_TRUNC);
    
    // Perform linear stretching
    cv::Mat linear_stretched_img = 255.0 / (minmax_values.at(1) - minmax_values.at(0)) * (stretched_img - minmax_values.at(0));

    linear_stretched_img.convertTo(linear_stretched_img, CV_8U);
    
    return linear_stretched_img;
}

std::vector<int> Image::computeHistogram() {
    check(CheckType::EmptyImage);
    std::vector<int> histogram(256);
        for (int i = 0; i < this->image.rows; ++i) {
            for (int j = 0; j < this->image.cols; ++j) {
                int pixel = this->image.at<uchar>(i,j);
                histogram.at(pixel) += 1;
            }
        }
    return histogram; 
}

cv::Mat Image::stretchImage() {
    std::vector<int> histogram = this->computeHistogram();
    std::vector<int> minmax_values = this->findPercentileValues(histogram);
    cv::Mat stretched_img = linearStretching(minmax_values);
    return stretched_img;
}

cv::Mat Image::cropImage() {
    cv::Mat cropped_img = this->image(this->roi);
    return cropped_img;
}

Image::~Image() {}