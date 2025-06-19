# include "mlzero/core/vector.hpp"
# include <cmath>
# include <algorithm>
# include <iostream>
# include <iomanip>
# include <random>

namespace mlzero {
namespace core {


// Private helper mathods 
void Vector::reallocate(size_t new_capacity) {
    double* new_data = new double[new_capacity];

    // copy existing data 
    size_t copy_size = std::min(size_, new_capacity);
    for (size_t i = 0; i < copy_size; ++i){
        new_data[i] = data_[i];
    }

    delete[] data_;
    data_ = new_data;
    capacity_ = new_capacity;
    if (size_ > new_capacity) {
        size_ = new_capacity; // Adjust size if it exceeds new capacity
    }
}
void Vector::copy_from(const Vector& other) {
    size_ = other.size_;
    capacity_ = other.capacity_;
    data_ = new double[capacity_];
    for (size_t i = 0; i < size_; ++i) {
        data_[i] = other.data_[i];
    }
}

void Vector::cleanup() {
    delete[] data_;
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
}
// Constructors and destructor
Vector::Vector() : data_(nullptr), size_(0), capacity_(0) {}

Vector::Vector(size_t size) : size_(size), capacity_(size) {
    if (size > 0) {
        data_ = new double[size];
        std::fill(data_, data_ + size, 0.0); // Initialize with zeros
    } else {
        data_ = nullptr;
    }
}

Vector::Vector(size_t size, double value) : size_(size), capacity_(size) {
    if (size > 0) {
        data_ = new double[size];
        std::fill(data_, data_ + size, value); // Initialize with the given value
    } else {
        data_ = nullptr;
    }
}

Vector::Vector(std::initializer_list<double> values) 
    : size_(values.size()), capacity_(values.size()) {
    if (size_ > 0) {
        data_ = new double[size_];
        std::copy(values.begin(), values.end(), data_);
    } else {
        data_ = nullptr;
    }
}

Vector::Vector(const Vector& other) : data_(nullptr), size_(0), capacity_(0) {
    copy_from(other);
}

Vector::Vector(Vector&& other) noexcept 
    : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
}

Vector::~Vector() {
    cleanup();
}

// assignement operators 

Vector& Vector::operator=(const Vector& other) {
    if (this != &other) {
        cleanup(); // Clean up current data
        copy_from(other); // Copy from the other vector
    }
    return *this;
}

Vector& Vector::operator=(Vector&& other) noexcept {
    if (this != &other) {
        cleanup(); 
        data_ = other.data_;
        size_ = other.size_;
        capacity_ = other.capacity_;
        
        // Reset the moved-from object
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }
    return *this;
}

// Element access
double& Vector::operator[](size_t index) {
    return data_[index];
}

const double& Vector::operator[](size_t index) const {
    return data_[index];
}

double& Vector::at(size_t index) {
    if (index >= size_) {
        throw std::out_of_range("Index out of bounds");
    }
    return data_[index];
}

const double& Vector::at(size_t index) const {
    if (index >= size_) {
        throw std::out_of_range("Index out of bounds");
    }
    return data_[index];
}

// Size and capacity
void Vector::reserve(size_t new_capacity) {
    if (new_capacity > capacity_) {
        reallocate(new_capacity);
    }
}
void Vector::resize(size_t new_size, double value) {
    if (new_size > capacity_) {
        reallocate(new_size);
    }
    if (new_size > size_) {
        std::fill(data_ + size_, data_ + new_size, value); // Initialize new elements with value
    }
    size_ = new_size;
}

// Modifiers
void Vector::push_back(double value) {
    if (size_ >= capacity_) {
        reserve(capacity_ == 0 ? 1 : capacity_ * 2); // Double the capacity if full
    }
    data_[size_++] = value;
}

void Vector::pop_back() {
    if (size_ > 0) {
        --size_;
    } else {
        throw std::out_of_range("Cannot pop from an empty vector");
    }
}
void Vector::clear() {
    size_ = 0; // Just reset size, data remains allocated
}

// Operations
Vector Vector::operator+(const Vector& other) const {
    if (size_ != other.size_) {
        throw std::invalid_argument("Vectors must be of the same size for addition");
    }
    Vector result(size_);
    for (size_t i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Vector Vector::operator-(const Vector& other) const {
    if (size_ != other.size_) {
        throw std::invalid_argument("Vectors must be of the same size for subtraction");
    }
    Vector result(size_);
    for (size_t i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Vector Vector::operator*(double scalar) const {
    Vector result(size_);
    for (size_t i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

Vector Vector::operator/(double scalar) const {
    if (std::abs(scalar) < 1e-15) {
        throw std::invalid_argument("Division by zero");
    }
    Vector result(size_);
    for (size_t i = 0; i < size_; ++i) {
        result.data_[i] = data_[i] / scalar;
    }
    return result;
}

Vector& Vector::operator+=(const Vector& other) {
    if (size_ != other.size_) {
        throw std::invalid_argument("Vectors must be of the same size for addition assignment");
    }
    for (size_t i = 0; i < size_; ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Vector& Vector::operator-=(const Vector& other) {
    if (size_ != other.size_) {
        throw std::invalid_argument("Vectors must be of the same size for subtraction assignment");
    }
    for (size_t i = 0; i < size_; ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

Vector& Vector::operator*=(double scalar) {
    for (size_t i = 0; i < size_; ++i) {
        data_[i] *= scalar;
    }
    return *this;
}

Vector& Vector::operator/=(double scalar) {
    if (std::abs(scalar) < 1e-15) {
        throw std::invalid_argument("Division by zero");
    }
    for (size_t i = 0; i < size_; ++i) {
        data_[i] /= scalar;
    }
    return *this;
}

// Mathematical operations
double Vector::dot(const Vector& other) const {
    if (size_ != other.size_) {
        throw std::invalid_argument("Vectors must be of the same size for dot product");
    }
    double result = 0.0;
    for (size_t i = 0; i < size_; ++i) {
        result += data_[i] * other.data_[i];
    }
    return result;
}

Vector Vector::cross(const Vector& other) const {
    if (size_ != 3 || other.size_ != 3) {
        throw std::invalid_argument("Cross product is only defined for 3D vectors");
    }
    Vector result(3);
    result.data_[0] = data_[1] * other.data_[2] - data_[2] * other.data_[1];
    result.data_[1] = data_[2] * other.data_[0] - data_[0] * other.data_[2];
    result.data_[2] = data_[0] * other.data_[1] - data_[1] * other.data_[0];
    return result;
}

double Vector::magnitude() const {
    return std::sqrt(magnitude_squared());
}

double Vector::magnitude_squared() const {
    double sum = 0.0;
    for (size_t i = 0; i < size_; ++i) {
        sum += data_[i] * data_[i];
    }
    return sum;
}

void Vector::normalize() {
    double mag = magnitude();
    if (std::abs(mag) < 1e-15) {
        throw std::runtime_error("Cannot normalize a zero vector");
    }
    for (size_t i = 0; i < size_; ++i) {
        data_[i] /= mag;
    }
}

Vector Vector::normalized() const {
    Vector result(*this);
    result.normalize();
    return result;
}

double Vector::distance(const Vector& other) const {
    if (size_ != other.size_) {
        throw std::invalid_argument("Vectors must be of the same size for distance calculation");
    }
    double sum = 0.0;
    for (size_t i = 0; i < size_; ++i) {
        double diff = data_[i] - other.data_[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

double Vector::angle(const Vector& other) const {
    if (size_ != other.size_) {
        throw std::invalid_argument("Vectors must be of the same size for angle calculation");
    }
    double dot_product = dot(other);
    double magnitudes = magnitude() * other.magnitude();
    if (std::abs(magnitudes) < 1e-15) {
        throw std::invalid_argument("Cannot calculate angle with a zero vector");
    }
    double cos_angle = dot_product / magnitudes;
    // Clamp the value to the range [-1, 1] to avoid NaN due to floating-point precision issues
    cos_angle = std::clamp(cos_angle, -1.0, 1.0);
    return std::acos(cos_angle);
}

//comparison operators
bool Vector::operator==(const Vector& other) const {
    if (size_ != other.size_) {
        return false;
    }
    for (size_t i = 0; i < size_; ++i) {
        if (data_[i] != other.data_[i]) {
            return false;
        }
    }
    return true;
}

bool Vector::operator!=(const Vector& other) const {
    return !(*this == other);
}

// Utility functions
void Vector::fill(double value) {
    std::fill(data_, data_ + size_, value);
}
Vector Vector::slice(size_t start, size_t end) const {
    if (start >= size_ || end > size_ || start >= end) {
        throw std::out_of_range("Invalid slice range");
    }
    Vector result(end - start);
    if (result.data_ != nullptr){
        for (size_t i = start; i < end; ++i) {
            result.data_[i - start] = data_[i];
        }
    }
    return result;
}

double Vector::sum() const {
    double total = 0.0;
    for (size_t i = 0; i < size_; ++i) {
        total += data_[i];
    }
    return total;
}

double Vector::mean() const {
    if (size_ == 0) {
        throw std::runtime_error("Cannot calculate mean of an empty vector");
    }
    return sum() / static_cast<double>(size_);
}

double Vector::min() const {
    if (size_ == 0) {
        throw std::runtime_error("Cannot find minimum of an empty vector");
    }
    double min_value = data_[0];
    for (size_t i = 1; i < size_; ++i) {
        if (data_[i] < min_value) {
            min_value = data_[i];
        }
    }
    return min_value;
}

double Vector::max() const {
    if (size_ == 0) {
        throw std::runtime_error("Cannot find maximum of an empty vector");
    }
    double max_value = data_[0];
    for (size_t i = 1; i < size_; ++i) {
        if (data_[i] > max_value) {
            max_value = data_[i];
        }
    }
    return max_value;
}

// Static utility functions
Vector Vector::zeros(size_t size) {
    return Vector(size, 0.0);
}

Vector Vector::ones(size_t size) {
    return Vector(size, 1.0);
}

Vector Vector::random(size_t size, double min, double max) {
    if (size == 0) {
        return Vector();
    }
    Vector result(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);
    for (size_t i = 0; i < size; ++i) {
        result.data_[i] = dis(gen);
    }
    return result;
}

Vector Vector::linspace(double start, double end, size_t num_points) {
    if (num_points == 0) {
        return Vector();
    }
    if (num_points == 1) {
        return Vector({start});
    }
    Vector result(num_points);
    double step = (end - start) / static_cast<double>(num_points - 1);
    for (size_t i = 0; i < num_points; ++i) {
        result.data_[i] = start + i * step;
    }
    return result;
}

// Non-member functions
Vector operator*(double scalar, const Vector& vec) {
    return vec * scalar; // Use the member function for scalar multiplication
}

std::ostream& operator<<(std::ostream& os, const Vector& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << std::fixed << std::setprecision(2) << vec[i];
        if (i < vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

} // namespace core
} // namespace mlzero
