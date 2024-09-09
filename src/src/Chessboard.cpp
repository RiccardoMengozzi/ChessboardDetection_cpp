#include <Chessboard.hpp>

#define THROW_RUNTIME_ERROR(msg) throw std::runtime_error(std::string("Error: ") + msg + " at " + __FILE__ + ":" + std::to_string(__LINE__))

Chessboard::Chessboard() :  pattern_size(pattern_size),
                            squares_size(squares_size) {}


void Chessboard::check(CheckType type) const {

    switch (type)
    {
    case CheckType::EmptyImage :
            if (this->img.empty()) {
                THROW_RUNTIME_ERROR("Error: The image is empty! Please set an image with setImage()");
            }
        break;
    
    case CheckType::PatternSize:
            if (this->pattern_size.empty()) {
                THROW_RUNTIME_ERROR("Error: pattern size not defined! Please set a pattern size with setPatternSize()");
            }
        break;

    case CheckType::SquaresSize:
            if (this->squares_size.empty()) {
                THROW_RUNTIME_ERROR("Error: squares size is empty! Please set the squares size with setSquaresSize()");
            }
        break;
    case CheckType::Center:
            if (!this->center_computed) {
                THROW_RUNTIME_ERROR("Error: chessboard center has not been computed yet! Please compute the chessboard center with computeChessboardCenter()");
            }
        break;
    case CheckType::Corners:
            if (this->corners.empty()) {
                THROW_RUNTIME_ERROR("Error: corners is empty! Please compute corners with detect()");
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
    this->img = img;
}

cv::Size Chessboard::getPatternSize() {
    check(CheckType::PatternSize);
    return this->pattern_size;
}

cv::Size2f Chessboard::getSquaresSize() {
    check(CheckType::SquaresSize);
    return this->squares_size;
}

cv::Mat Chessboard::getImage() {
    check(CheckType::EmptyImage);
    return this->img;
}

bool Chessboard::isFound() {
    return this->found;
}

cv::Point2i Chessboard::getCenter() {
    check(CheckType::Center);
    return this->center;
}

std::vector<cv::Point2f> Chessboard::getCorners() {
    check(CheckType::Corners);
    return this->corners;
}


void Chessboard::detect(bool vis, const cv::Point2i image_position, const std::string image_name) {
    check(CheckType::EmptyImage);
    check(CheckType::PatternSize);
    check(CheckType::SquaresSize);
    this->found = cv::findChessboardCorners(this->img, this->pattern_size, this->corners);
    if (this->found) {
        cv::cornerSubPix(this->img, this->corners, cv::Size(5, 5), cv::Size(-1, -1),
                    cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));     
    }
    if (vis) {
        cv::namedWindow(image_name, cv::WINDOW_AUTOSIZE);
        cv::moveWindow(image_name, image_position.x, image_position.y);
        cv::Mat vis_img = this->img.clone();
        cv::drawChessboardCorners(vis_img, this->pattern_size, this->corners, this->found);
        cv::imshow(image_name, vis_img);
        cv::waitKey(1);
    }

}

void Chessboard::computeCenter(bool vis, const cv::Point2i image_position, const std::string image_name) {
    if(this->found) {
        check(CheckType::Corners);
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
            std::cout << center << '\n';
            cv::namedWindow(image_name, cv::WINDOW_AUTOSIZE);
            cv::moveWindow(image_name, image_position.x, image_position.y);
            cv::Mat vis_img = this->img.clone();
            cv::circle(vis_img, this->center, 5, (255,255,255), 3);
            cv::imshow(image_name, vis_img);
            cv::waitKey(1); 
        }
    }
}


// std::vector<cv::Point2f> Chessboard::createChessboardCoordinates() {
//     // Initialize a vector to hold all points in a flattened format
//     std::vector<cv::Point2f> chessboard_coordinates;
//     chessboard_coordinates.reserve(this->pattern_size.area());

//     // Iterate through the grid and calculate the coordinates, directly flattening them into the vector
//     for (int i = 0; i < this->pattern_size.width; ++i) {
//         for (int j = 0; j < this->pattern_size.height; ++j) {
//             float x = i * this->squares_size.width;
//             float y = j * this->squares_size.height;

//             chessboard_coordinates.emplace_back(x, y);  // Add the point directly to the flattened vector
//         }
//     }

//     return chessboard_coordinates;

// }




Chessboard::~Chessboard() {}