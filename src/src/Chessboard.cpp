#include <Chessboard.hpp>

#define THROW_RUNTIME_ERROR_AT(msg, file, line) throw std::runtime_error(std::string("Error: ") + msg + " at " + file + ":" + std::to_string(line))
#define THROW_RUNTIME_ERROR(msg) THROW_RUNTIME_ERROR_AT(msg, __FILE__, __LINE__)
#define CHECK(type) check(type, __FILE__, __LINE__)

Chessboard::Chessboard() :  pattern_size(pattern_size),
                            squares_size(squares_size) {}

Chessboard::~Chessboard() {}

void Chessboard::check(CheckType type, const char* file, int line) const {

    switch (type)
    {
    case CheckType::EmptyImage :
            if (this->img.empty()) {
                THROW_RUNTIME_ERROR_AT("Error: The image is empty! Please set an image with setImage()", file, line);
            }
        break;
    
    case CheckType::PatternSize:
            if (this->pattern_size.empty()) {
                THROW_RUNTIME_ERROR_AT("Error: pattern size not defined! Please set a pattern size with setPatternSize()", file, line);
            }
        break;

    case CheckType::SquaresSize:
            if (this->squares_size.empty()) {
                THROW_RUNTIME_ERROR_AT("Error: squares size is empty! Please set the squares size with setSquaresSize()", file, line);
            }
        break;
    case CheckType::Center:
            if (!this->center_computed) {
                THROW_RUNTIME_ERROR_AT("Error: chessboard center has not been computed yet! Please compute the chessboard center with computeChessboardCenter()", file, line);
            }
        break;
    case CheckType::Corners:
            if (this->corners.empty()) {
                THROW_RUNTIME_ERROR_AT("Error: corners is empty! Please compute corners with detect()", file, line);
            }
        break;
    }
}

void Chessboard::setPatternSize(cv::Size pattern_size) {
    this->pattern_size = pattern_size;
}

void Chessboard::setSquaresSize(cv::Size2f squares_size) {
    this->squares_size = squares_size;
}

void Chessboard::setImage(cv::Mat img) {
    this->img = img.clone();
}

cv::Size Chessboard::getPatternSize() {
    CHECK(CheckType::PatternSize);
    return this->pattern_size;
}

cv::Size2f Chessboard::getSquaresSize() {
    CHECK(CheckType::SquaresSize);
    return this->squares_size;
}

cv::Mat Chessboard::getImage() {
    CHECK(CheckType::EmptyImage);
    return this->img;
}

bool Chessboard::isFound() {
    return this->found;
}

cv::Point2i Chessboard::getCenter() {
    CHECK(CheckType::Center);
    return this->center;
}

std::vector<cv::Point2f> Chessboard::getCorners() {
    CHECK(CheckType::Corners);
    return this->corners;
}


void Chessboard::detect(bool vis, const cv::Point2i image_position, const std::string image_name) {
    CHECK(CheckType::EmptyImage);
    CHECK(CheckType::PatternSize);
    CHECK(CheckType::SquaresSize);
    this->found = cv::findChessboardCorners(this->img, this->pattern_size, this->corners);
    if (this->found) {
        cv::cornerSubPix(this->img, this->corners, cv::Size(5, 5), cv::Size(-1, -1),
                    cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));     
    }
    if (vis) {
        cv::namedWindow(image_name, cv::WINDOW_AUTOSIZE);
        cv::moveWindow(image_name, image_position.x, image_position.y);
        cv::Mat vis_img = this->img.clone();
        cv::cvtColor(this->img, vis_img, cv::COLOR_GRAY2BGR);
        cv::drawChessboardCorners(vis_img, this->pattern_size, this->corners, this->found);
        cv::imshow(image_name, vis_img);
        cv::waitKey(1);
    }

}

void Chessboard::computeCenter(bool vis, const cv::Point2i image_position, const std::string image_name) {
    if(this->found) {
        CHECK(CheckType::Corners);
        float sum_x = 0;
        float sum_y = 0;
        
        for (const auto& corner : this->corners) {
            sum_x += corner.x;
            sum_y += corner.y;
        }
        this->center_computed = true;
        this->center.x = static_cast<int>(sum_x / this->corners.size());
        this->center.y = static_cast<int>(sum_y / this->corners.size());

        if (vis) {
            cv::namedWindow(image_name, cv::WINDOW_AUTOSIZE);
            cv::moveWindow(image_name, image_position.x, image_position.y);
            cv::Mat vis_img = this->img.clone();
            cv::cvtColor(this->img, vis_img, cv::COLOR_GRAY2BGR);
            cv::circle(vis_img, this->center, 5, (0,0,255), 3);
            cv::imshow(image_name, vis_img);
            cv::waitKey(1); 
        }
    }
}


std::vector<cv::Point2i> Chessboard::computeVertices(bool vis, const cv::Point2i image_position, const std::string image_name) {
    this->vertices.clear();
    if(corners.empty()) {
        this->vertices.push_back(cv::Point2f(0,0));
        this->vertices.push_back(cv::Point2f(0,0));
        this->vertices.push_back(cv::Point2f(0,0));
        this->vertices.push_back(cv::Point2f(0,0));
        return this->vertices;
    }
    this->vertices.push_back(this->corners.at(0));
    this->vertices.push_back(this->corners.at(this->pattern_size.width - 1));
    this->vertices.push_back(this->corners.at(this->pattern_size.width * (this->pattern_size.height - 1)));
    this->vertices.push_back(this->corners.at(this->pattern_size.area() - 1));
    if (vis) {
        cv::namedWindow(image_name, cv::WINDOW_AUTOSIZE);
        cv::moveWindow(image_name, image_position.x, image_position.y);
        cv::Mat vis_img = this->img.clone();
        cv::cvtColor(this->img, vis_img, cv::COLOR_GRAY2BGR);    
        cv::circle(vis_img, this->vertices.at(0), 3, (0,0,255), 3);
        cv::circle(vis_img, this->vertices.at(1), 3, (0,0,255), 3);
        cv::circle(vis_img, this->vertices.at(2), 3, (0,0,255), 3);
        cv::circle(vis_img, this->vertices.at(3), 3, (0,0,255), 3);
        cv::imshow(image_name, vis_img);
        cv::waitKey(1); 
    }
    return this->vertices;

}



