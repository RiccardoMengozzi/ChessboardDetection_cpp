#include <Disparity.hpp>


#define THROW_RUNTIME_ERROR_AT(msg, file, line) throw std::runtime_error(std::string("Error: ") + msg + " at " + file + ":" + std::to_string(line))
#define THROW_RUNTIME_ERROR(msg) THROW_RUNTIME_ERROR_AT(msg, __FILE__, __LINE__)
#define CHECK(type) check(type, __FILE__, __LINE__)

Disparity::Disparity() {}

Disparity::~Disparity() {}


void Disparity::check(CheckType type, const char* file, int line) const {
    switch(type)
    {
        case CheckType::EmptyImages:
            if (this->imgL.empty()) {
                THROW_RUNTIME_ERROR_AT("The left image is empty! Please set images with setImages().", file, line);
            }
            if (this->imgR.empty()) {
                THROW_RUNTIME_ERROR_AT("The right image is empty! Please set images with setImages().", file, line);
            }
            break;
        case CheckType::SGBMParameters:
            if (!this->param_set) {
                THROW_RUNTIME_ERROR_AT("SGBM parameters have not been set yet. Please use setSGBMParameters().", file, line);
            }
            break;
        case CheckType::Roi:
            if (this->roi.empty()) {
                THROW_RUNTIME_ERROR_AT("ROI has not been set yet. Please use setRoi().", file, line);
            }
            break;
        case CheckType::EmptyDisparityMap:
            if (this->disparityMap.empty()) {
                THROW_RUNTIME_ERROR_AT("The disparity map has not been set yet. Please use computeDisparityMap().", file, line);
            }
            break;
        case CheckType::DmainRoi:
            if (this->dmain_roi.empty()) {
                THROW_RUNTIME_ERROR_AT("The ROI for main disparity computation has not been set yet. Please use setDmainRoi().", file, line);
            }
            break;
        case CheckType::FocalAndBaseline:
            if (!this->focal_and_baseline_set) {
                THROW_RUNTIME_ERROR_AT("Focal and/or Baseline parameters for distance computation have not been set yet. Please use setFocal() and/or setBaseline().", file, line);
            }
            break;
    }
}

void Disparity::setImages(cv::Mat imgL, cv::Mat imgR) {
    this->imgL = imgL.clone();
    this->imgR = imgR.clone();
}

void Disparity::setSGBMParameters(SGBMParam parameters) {
    this->parameters = parameters;
    this->param_set = true;
}

void Disparity::setRoi(cv::Rect roi) {
    this->roi = roi;
}

void Disparity::setRoi(int roi_x, int roi_y, int roi_width, int roi_height) {
    this->roi.x = roi_x;
    this->roi.y = roi_y;
    this->roi.width = roi_width;
    this->roi.height = roi_height;
}

void Disparity::setDmainRoi(cv::Point2i center, int window_radius) {
    this->dmain_roi.x = center.x - window_radius;
    this->dmain_roi.y = center.y - window_radius;
    this->dmain_roi.width = window_radius * 2;
    this->dmain_roi.height = window_radius * 2;
}

void Disparity::setFocalAndBaseline(float focal, float baseline) {
    this->focal = focal;
    this-> baseline = baseline;
    this->focal_and_baseline_set = true;
}

cv::Mat Disparity::getLeftImage() {
    return this->imgL;
}

cv::Mat Disparity::getRightImage() {
    return this->imgR;
}

SGBMParam Disparity::getSGBMParameters() {
    CHECK(CheckType::SGBMParameters);
    return this->parameters;
}

cv::Rect Disparity::getRoi() {
    CHECK(CheckType::Roi);
    return this->roi;
}

cv::Rect Disparity::getDmainRoi() {
    CHECK(CheckType::DmainRoi);
    return this->roi;
}

float Disparity::getFocal() {
    CHECK(CheckType::FocalAndBaseline);
    return this->focal;
}

float Disparity::getBaseline() {
    CHECK(CheckType::FocalAndBaseline);
    return this->baseline;
}

void Disparity::assignSGBMParameters() {
    SGBM->setBlockSize(this->parameters.block_size);
    SGBM->setNumDisparities(this->parameters.num_disparities);
    SGBM->setPreFilterCap(this->parameters.pre_filter_cap);
    SGBM->setMinDisparity(this->parameters.offset);
    SGBM->setUniquenessRatio(this->parameters.uniqueness_ratio);
    SGBM->setSpeckleWindowSize(this->parameters.speckle_window_size);
    SGBM->setSpeckleRange(this->parameters.speckle_range);
    SGBM->setDisp12MaxDiff(this->parameters.disp_12_max_diff);
    SGBM->setP1(this->parameters.p1);
    SGBM->setP2(this->parameters.p2);
    switch (this->parameters.mode)
    {
    case _3WAY:
        SGBM->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);        
        break;
    }
}


cv::Mat Disparity::computeDisparityMap(bool vis, const cv::Point2i image_position, const std::string image_name) {
    Disparity::assignSGBMParameters();
    this->disparityMap.convertTo(this->disparityMap, CV_32F);
    this->SGBM->compute(this->imgL, this->imgR, this->disparityMap);
    if (vis) {
        cv::namedWindow(image_name, cv::WINDOW_AUTOSIZE);
        cv::moveWindow(image_name, image_position.x, image_position.y);
        cv::Mat vis_img;
        cv::normalize(this->disparityMap, vis_img, 0, 255, cv::NORM_MINMAX);
        vis_img.convertTo(vis_img, CV_8U);
        cv::imshow(image_name, vis_img);
        cv::waitKey(1);
    }
    this->disparityMap = this->disparityMap / 16;
    return disparityMap;
}

float Disparity::computeMainDisparity(bool vis, const cv::Point2i image_position, const std::string image_name) {
    CHECK(CheckType::EmptyDisparityMap);
    CHECK(CheckType::DmainRoi);
    std::cout << "roi" << dmain_roi << '\n';
    float sum = 0;
    int count = 0;
    float dmain = 0;
    float value = 0;
    for (int i = this->dmain_roi.x; i < this->dmain_roi.x + this->dmain_roi.width; i++) {
        for (int j = this->dmain_roi.y; j < this->dmain_roi.y + this->dmain_roi.height; j++) {
            value = static_cast<float>(disparityMap.at<uchar>(j,i));
            if (value != 255 && value > 0) {
                std::cout << "VALUE = " << value << '\n';
                sum += value;
                count++;
            }
        }
    }
    this->dmain = sum / count;
    if (vis) {

        cv::namedWindow(image_name, cv::WINDOW_AUTOSIZE);
        cv::moveWindow(image_name, image_position.x, image_position.y);
        cv::Mat vis_img;
        cv::normalize(this->disparityMap, vis_img, 0, 255, cv::NORM_MINMAX);
        vis_img.convertTo(vis_img, CV_8U);
        cv::rectangle(vis_img, this->dmain_roi, (255,255,255), 5);
        cv::imshow(image_name, vis_img);
        cv::waitKey(1);
    }

    return this->dmain;
}

float Disparity::computeDistance() {
    CHECK(CheckType::FocalAndBaseline);
    return (this->focal * this->baseline / this->dmain / 1000);
}




