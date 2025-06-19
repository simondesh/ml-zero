#include "mlzero/mlzero.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <random>

namespace mlzero {
namespace core {

// Private helper methods
void Matrix::allocate_memory() {
    if (rows_ == 0 || cols_ == 0) {
        data_ = nullptr;
        return;
    }
    
    data_ = new double*[rows_];
    data_[0] = new double[rows_ * cols_];
    
    for (size_t i = 1; i < rows_; ++i) {
        data_[i] = data_[0] + i * cols_;
    }
    
    // Initialize to zero
    std::fill(data_[0], data_[0] + rows_ * cols_, 0.0);
}

void Matrix::deallocate_memory() {
    if (data_ != nullptr) {
        delete[] data_[0];
        delete[] data_;
        data_ = nullptr;
    }
}

void Matrix::copy_from(const Matrix& other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    allocate_memory();
    
    if (data_ != nullptr && other.data_ != nullptr) {
        std::copy(other.data_[0], other.data_[0] + rows_ * cols_, data_[0]);
    }
}

void Matrix::check_bounds(size_t row, size_t col) const {
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("Matrix indices out of bounds");
    }
}

void Matrix::check_same_dimensions(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match");
    }
}

// Constructors and Destructor
Matrix::Matrix() : data_(nullptr), rows_(0), cols_(0) {}

Matrix::Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
    allocate_memory();
}

Matrix::Matrix(size_t rows, size_t cols, double value) : rows_(rows), cols_(cols) {
    allocate_memory();
    if (data_ != nullptr) {
        std::fill(data_[0], data_[0] + rows_ * cols_, value);
    }
}

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> values) {
    rows_ = values.size();
    cols_ = rows_ > 0 ? values.begin()->size() : 0;
    
    // Check that all rows have the same number of columns
    for (const auto& row : values) {
        if (row.size() != cols_) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
    }
    
    allocate_memory();
    
    if (data_ != nullptr) {
        size_t i = 0;
        for (const auto& row : values) {
            size_t j = 0;
            for (double value : row) {
                data_[i][j] = value;
                ++j;
            }
            ++i;
        }
    }
}

Matrix::Matrix(const std::vector<std::vector<double>>& data) {
    rows_ = data.size();
    cols_ = rows_ > 0 ? data[0].size() : 0;
    
    // Check that all rows have the same number of columns
    for (const auto& row : data) {
        if (row.size() != cols_) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
    }
    
    allocate_memory();
    
    if (data_ != nullptr) {
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                data_[i][j] = data[i][j];
            }
        }
    }
}

Matrix::Matrix(const Matrix& other) {
    copy_from(other);
}

Matrix::Matrix(Matrix&& other) noexcept 
    : data_(other.data_), rows_(other.rows_), cols_(other.cols_) {
    other.data_ = nullptr;
    other.rows_ = 0;
    other.cols_ = 0;
}

Matrix::~Matrix() {
    deallocate_memory();
}

// Assignment operators
Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        deallocate_memory();
        copy_from(other);
    }
    return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        deallocate_memory();
        data_ = other.data_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        other.data_ = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
    }
    return *this;
}

// Element access
double& Matrix::operator()(size_t row, size_t col) {
    return data_[row][col];
}

const double& Matrix::operator()(size_t row, size_t col) const {
    return data_[row][col];
}

double& Matrix::at(size_t row, size_t col) {
    check_bounds(row, col);
    return data_[row][col];
}

const double& Matrix::at(size_t row, size_t col) const {
    check_bounds(row, col);
    return data_[row][col];
}

// Row and column access
Vector Matrix::get_row(size_t row) const {
    if (row >= rows_) {
        throw std::out_of_range("Row index out of bounds");
    }
    
    Vector result(cols_);
    for (size_t j = 0; j < cols_; ++j) {
        result[j] = data_[row][j];
    }
    return result;
}

Vector Matrix::get_column(size_t col) const {
    if (col >= cols_) {
        throw std::out_of_range("Column index out of bounds");
    }
    
    Vector result(rows_);
    for (size_t i = 0; i < rows_; ++i) {
        result[i] = data_[i][col];
    }
    return result;
}

void Matrix::set_row(size_t row, const Vector& values) {
    if (row >= rows_) {
        throw std::out_of_range("Row index out of bounds");
    }
    if (values.size() != cols_) {
        throw std::invalid_argument("Vector size must match number of columns");
    }
    
    for (size_t j = 0; j < cols_; ++j) {
        data_[row][j] = values[j];
    }
}

void Matrix::set_column(size_t col, const Vector& values) {
    if (col >= cols_) {
        throw std::out_of_range("Column index out of bounds");
    }
    if (values.size() != rows_) {
        throw std::invalid_argument("Vector size must match number of rows");
    }
    
    for (size_t i = 0; i < rows_; ++i) {
        data_[i][col] = values[i];
    }
}

// Matrix operations
Matrix Matrix::operator+(const Matrix& other) const {
    check_same_dimensions(other);
    
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data_[i][j] = data_[i][j] + other.data_[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    check_same_dimensions(other);
    
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data_[i][j] = data_[i][j] - other.data_[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    Matrix result(rows_, other.cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < other.cols_; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < cols_; ++k) {
                sum += data_[i][k] * other.data_[k][j];
            }
            result.data_[i][j] = sum;
        }
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data_[i][j] = data_[i][j] * scalar;
        }
    }
    return result;
}

Matrix Matrix::operator/(double scalar) const {
    if (std::abs(scalar) < 1e-15) {
        throw std::invalid_argument("Division by zero");
    }
    
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data_[i][j] = data_[i][j] / scalar;
        }
    }
    return result;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    check_same_dimensions(other);
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            data_[i][j] += other.data_[i][j];
        }
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    check_same_dimensions(other);
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            data_[i][j] -= other.data_[i][j];
        }
    }
    return *this;
}

Matrix& Matrix::operator*=(const Matrix& other) {
    *this = *this * other;
    return *this;
}

Matrix& Matrix::operator*=(double scalar) {
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            data_[i][j] *= scalar;
        }
    }
    return *this;
}

Matrix& Matrix::operator/=(double scalar) {
    if (std::abs(scalar) < 1e-15) {
        throw std::invalid_argument("Division by zero");
    }
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            data_[i][j] /= scalar;
        }
    }
    return *this;
}

// Matrix-Vector operations
Vector Matrix::operator*(const Vector& vec) const {
    if (cols_ != vec.size()) {
        throw std::invalid_argument("Matrix columns must match vector size");
    }
    
    Vector result(rows_);
    for (size_t i = 0; i < rows_; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < cols_; ++j) {
            sum += data_[i][j] * vec[j];
        }
        result[i] = sum;
    }
    return result;
}

// Matrix properties and operations
Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.data_[j][i] = data_[i][j];
        }
    }
    return result;
}

Matrix& Matrix::transpose_inplace() {
    if (!is_square()) {
        throw std::runtime_error("In-place transpose only supported for square matrices");
    }
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = i + 1; j < cols_; ++j) {
            std::swap(data_[i][j], data_[j][i]);
        }
    }
    return *this;
}

double Matrix::trace() const {
    if (!is_square()) {
        throw std::runtime_error("Trace only defined for square matrices");
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < rows_; ++i) {
        sum += data_[i][i];
    }
    return sum;
}

double Matrix::determinant() const {
    if (!is_square()) {
        throw std::runtime_error("Determinant only defined for square matrices");
    }
    
    if (rows_ == 1) {
        return data_[0][0];
    }
    
    if (rows_ == 2) {
        return data_[0][0] * data_[1][1] - data_[0][1] * data_[1][0];
    }
    
    if (rows_ == 3) {
        return data_[0][0] * (data_[1][1] * data_[2][2] - data_[1][2] * data_[2][1])
             - data_[0][1] * (data_[1][0] * data_[2][2] - data_[1][2] * data_[2][0])
             + data_[0][2] * (data_[1][0] * data_[2][1] - data_[1][1] * data_[2][0]);
    }
    
    // For larger matrices, use LU decomposition
    auto lu = lu_decomposition();
    if (lu.is_singular) {
        return 0.0;
    }
    
    double det = 1.0;
    for (size_t i = 0; i < rows_; ++i) {
        det *= lu.U.data_[i][i];
    }
    
    // Account for row swaps in permutation
    size_t swaps = 0;
    for (size_t i = 0; i < rows_; ++i) {
        if (static_cast<size_t>(lu.permutation[i]) != i) {
            swaps++;
        }
    }
    
    return (swaps % 2 == 0) ? det : -det;
}

Matrix::LUDecomposition Matrix::lu_decomposition() const {
    if (!is_square()) {
        throw std::runtime_error("LU decomposition only defined for square matrices");
    }
    
    LUDecomposition result;
    result.L = Matrix::identity(rows_);
    result.U = *this;
    result.permutation = Vector(rows_);
    result.is_singular = false;
    
    // Initialize permutation
    for (size_t i = 0; i < rows_; ++i) {
        result.permutation[i] = static_cast<double>(i);
    }
    
    // Gaussian elimination with partial pivoting
    for (size_t k = 0; k < rows_ - 1; ++k) {
        // Find pivot
        size_t pivot_row = k;
        double max_val = std::abs(result.U.data_[k][k]);
        
        for (size_t i = k + 1; i < rows_; ++i) {
            double current_val = std::abs(result.U.data_[i][k]);
            if (current_val > max_val) {
                max_val = current_val;
                pivot_row = i;
            }
        }
        
        // Check for singularity
        if (max_val < 1e-12) { // 
            result.is_singular = true;
            return result;
        }
        
        // Perform row swapping if needed
        if (pivot_row != k) {
            // Swap rows in U
            result.U.swap_rows(k, pivot_row);
            
            // Swap rows in L (only the part below diagonal that's already computed)
            for (size_t j = 0; j < k; ++j) {
                std::swap(result.L.data_[k][j], result.L.data_[pivot_row][j]);
            }
            
            // Update permutation
            std::swap(result.permutation[k], result.permutation[pivot_row]);
        }
        
        // Double-check after swapping - the diagonal element should not be zero
        if (std::abs(result.U.data_[k][k]) < 1e-12) {
            result.is_singular = true;
            return result;
        }
        
        // Elimination
        for (size_t i = k + 1; i < rows_; ++i) {
            double factor = result.U.data_[i][k] / result.U.data_[k][k];
            result.L.data_[i][k] = factor;
            
            for (size_t j = k; j < cols_; ++j) {
                result.U.data_[i][j] -= factor * result.U.data_[k][j];
            }
        }
    }
    
    return result;
}

Matrix Matrix::inverse() const {
    if (!is_square()) {
        throw std::runtime_error("Inverse only defined for square matrices");
    }
    
    auto lu = lu_decomposition();
    if (lu.is_singular) {
        throw std::runtime_error("Matrix is singular and cannot be inverted");
    }
    
    Matrix inv = Matrix::identity(rows_);
    
    // Solve for each column of the inverse
    for (size_t col = 0; col < cols_; ++col) {
        Vector b = Matrix::identity(rows_).get_column(col);
        Vector x = solve(b);
        inv.set_column(col, x);
    }
    
    return inv;
}

Vector Matrix::solve(const Vector& b) const {
    if (!is_square()) {
        throw std::runtime_error("Can only solve square systems");
    }
    if (b.size() != rows_) {
        throw std::invalid_argument("Right-hand side vector size must match matrix rows");
    }
    
    auto lu = lu_decomposition();
    if (lu.is_singular) {
        throw std::runtime_error("Matrix is singular - system has no unique solution");
    }
    
    // Apply permutation to b
    Vector pb(rows_);
    for (size_t i = 0; i < rows_; ++i) {
        pb[i] = b[static_cast<size_t>(lu.permutation[i])];
    }
    
    // Forward substitution: solve Ly = Pb
    Vector y(rows_);
    for (size_t i = 0; i < rows_; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < i; ++j) {
            sum += lu.L.data_[i][j] * y[j];
        }
        y[i] = pb[i] - sum;
    }
    
    // Back substitution: solve Ux = y
    Vector x(rows_);
    for (int i = static_cast<int>(rows_) - 1; i >= 0; --i) {
        double sum = 0.0;
        for (size_t j = static_cast<size_t>(i) + 1; j < cols_; ++j) {
            sum += lu.U.data_[i][j] * x[j];
        }
        x[static_cast<size_t>(i)] = (y[static_cast<size_t>(i)] - sum) / lu.U.data_[i][i];
    }
    
    return x;
}

bool Matrix::is_symmetric(double tolerance) const {
    if (!is_square()) {
        return false;
    }
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            if (std::abs(data_[i][j] - data_[j][i]) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

bool Matrix::is_diagonal(double tolerance) const {
    if (!is_square()) {
        return false;
    }
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            if (i != j && std::abs(data_[i][j]) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

bool Matrix::is_identity(double tolerance) const {
    if (!is_square()) {
        return false;
    }
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (std::abs(data_[i][j] - expected) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

// Matrix norms
double Matrix::frobenius_norm() const {
    double sum = 0.0;
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            sum += data_[i][j] * data_[i][j];
        }
    }
    return std::sqrt(sum);
}

double Matrix::one_norm() const {
    double max_sum = 0.0;
    for (size_t j = 0; j < cols_; ++j) {
        double col_sum = 0.0;
        for (size_t i = 0; i < rows_; ++i) {
            col_sum += std::abs(data_[i][j]);
        }
        max_sum = std::max(max_sum, col_sum);
    }
    return max_sum;
}

double Matrix::infinity_norm() const {
    double max_sum = 0.0;
    for (size_t i = 0; i < rows_; ++i) {
        double row_sum = 0.0;
        for (size_t j = 0; j < cols_; ++j) {
            row_sum += std::abs(data_[i][j]);
        }
        max_sum = std::max(max_sum, row_sum);
    }
    return max_sum;
}

// Power method for largest eigenvalue
Matrix::EigenResult Matrix::power_method(double tolerance, size_t max_iterations) const {
    if (!is_square()) {
        throw std::runtime_error("Power method only works for square matrices");
    }
    
    EigenResult result;
    result.converged = false;
    
    // Start with random vector
    Vector x = Vector::random(rows_, -1.0, 1.0);
    x.normalize();
    
    double lambda_old = 0.0;
    
    for (size_t iter = 0; iter < max_iterations; ++iter) {
        Vector y = *this * x;
        double lambda = y.dot(x);
        
        // Avoid division by zero
        double y_magnitude = y.magnitude();
        if (y_magnitude < 1e-15) {
            break;
        }
        
        y.normalize();
        
        if (std::abs(lambda - lambda_old) < tolerance) {
            result.eigenvalue = lambda;
            result.eigenvector = y;
            result.converged = true;
            break;
        }
        
        lambda_old = lambda;
        x = y;
    }
    
    if (!result.converged) {
        result.eigenvalue = lambda_old;
        result.eigenvector = x;
    }
    
    return result;
}

// Comparison operators
bool Matrix::operator==(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        return false;
    }
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            if (std::abs(data_[i][j] - other.data_[i][j]) > 1e-15) {
                return false;
            }
        }
    }
    return true;
}

bool Matrix::operator!=(const Matrix& other) const {
    return !(*this == other);
}

// Utility functions
void Matrix::fill(double value) {
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            data_[i][j] = value;
        }
    }
}

void Matrix::fill_diagonal(double value) {
    if (!is_square()) {
        throw std::runtime_error("fill_diagonal only works for square matrices");
    }
    
    for (size_t i = 0; i < rows_; ++i) {
        data_[i][i] = value;
    }
}

Matrix Matrix::submatrix(size_t start_row, size_t end_row, size_t start_col, size_t end_col) const {
    if (start_row >= rows_ || end_row > rows_ || start_col >= cols_ || end_col > cols_ ||
        start_row >= end_row || start_col >= end_col) {
        throw std::out_of_range("Invalid submatrix range");
    }
    
    size_t sub_rows = end_row - start_row;
    size_t sub_cols = end_col - start_col;
    Matrix result(sub_rows, sub_cols);
    
    for (size_t i = 0; i < sub_rows; ++i) {
        for (size_t j = 0; j < sub_cols; ++j) {
            result.data_[i][j] = data_[start_row + i][start_col + j];
        }
    }
    
    return result;
}

void Matrix::resize(size_t new_rows, size_t new_cols, double fill_value) {
    Matrix new_matrix(new_rows, new_cols, fill_value);
    
    size_t copy_rows = std::min(rows_, new_rows);
    size_t copy_cols = std::min(cols_, new_cols);
    
    for (size_t i = 0; i < copy_rows; ++i) {
        for (size_t j = 0; j < copy_cols; ++j) {
            new_matrix.data_[i][j] = data_[i][j];
        }
    }
    
    *this = std::move(new_matrix);
}

void Matrix::swap_rows(size_t row1, size_t row2) {
    if (row1 >= rows_ || row2 >= rows_) {
        throw std::out_of_range("Row index out of bounds");
    }
    
    if (row1 != row2) {
        for (size_t j = 0; j < cols_; ++j) {
            std::swap(data_[row1][j], data_[row2][j]);
        }
    }
}

void Matrix::swap_cols(size_t col1, size_t col2) {
    if (col1 >= cols_ || col2 >= cols_) {
        throw std::out_of_range("Column index out of bounds");
    }
    
    if (col1 != col2) {
        for (size_t i = 0; i < rows_; ++i) {
            std::swap(data_[i][col1], data_[i][col2]);
        }
    }
}

// Statistical functions
Vector Matrix::row_sums() const {
    Vector result(rows_);
    for (size_t i = 0; i < rows_; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < cols_; ++j) {
            sum += data_[i][j];
        }
        result[i] = sum;
    }
    return result;
}

Vector Matrix::col_sums() const {
    Vector result(cols_);
    for (size_t j = 0; j < cols_; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < rows_; ++i) {
            sum += data_[i][j];
        }
        result[j] = sum;
    }
    return result;
}

Vector Matrix::row_means() const {
    Vector sums = row_sums();
    return sums / static_cast<double>(cols_);
}

Vector Matrix::col_means() const {
    Vector sums = col_sums();
    return sums / static_cast<double>(rows_);
}

double Matrix::sum() const {
    double total = 0.0;
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            total += data_[i][j];
        }
    }
    return total;
}

double Matrix::mean() const {
    if (rows_ == 0 || cols_ == 0) {
        throw std::runtime_error("Cannot compute mean of empty matrix");
    }
    return sum() / static_cast<double>(rows_ * cols_);
}

// Static factory methods
Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix Matrix::ones(size_t rows, size_t cols) {
    return Matrix(rows, cols, 1.0);
}

Matrix Matrix::identity(size_t size) {
    Matrix result(size, size, 0.0);
    for (size_t i = 0; i < size; ++i) {
        result.data_[i][i] = 1.0;
    }
    return result;
}

Matrix Matrix::diagonal(const Vector& diag_values) {
    size_t size = diag_values.size();
    Matrix result = Matrix::zeros(size, size);
    for (size_t i = 0; i < size; ++i) {
        result.data_[i][i] = diag_values[i];
    }
    return result;
}

Matrix Matrix::random(size_t rows, size_t cols, double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data_[i][j] = dis(gen);
        }
    }
    return result;
}

Matrix Matrix::random_symmetric(size_t size, double min, double max) {
    Matrix result = random(size, size, min, max);
    
    // Make it symmetric
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = i + 1; j < size; ++j) {
            result.data_[j][i] = result.data_[i][j];
        }
    }
    
    return result;
}

Matrix Matrix::vandermonde(const Vector& x, size_t degree) {
    size_t n = x.size();
    Matrix result(n, degree + 1);
    
    for (size_t i = 0; i < n; ++i) {
        double power = 1.0;
        for (size_t j = 0; j <= degree; ++j) {
            result.data_[i][j] = power;
            power *= x[i];
        }
    }
    
    return result;
}

// QR Decomposition using Gram-Schmidt
Matrix::QRDecomposition Matrix::qr_decomposition() const {
    QRDecomposition result;
    result.Q = Matrix::zeros(rows_, cols_);
    result.R = Matrix::zeros(cols_, cols_);
    
    // Gram-Schmidt process
    for (size_t j = 0; j < cols_; ++j) {
        Vector a_j = get_column(j);
        Vector q_j = a_j;
        
        // Subtract projections onto previous q vectors
        for (size_t i = 0; i < j; ++i) {
            Vector q_i = result.Q.get_column(i);
            double projection = a_j.dot(q_i);
            result.R(i, j) = projection;
            q_j = q_j - q_i * projection;
        }
        
        // Normalize
        double norm = q_j.magnitude();
        if (norm > 1e-15) {
            q_j = q_j / norm;
            result.R(j, j) = norm;
        } else {
            // Handle linearly dependent vectors
            result.R(j, j) = 0.0;
        }
        
        result.Q.set_column(j, q_j);
    }
    
    return result;
}

Matrix Matrix::solve(const Matrix& B) const {
    if (!is_square()) {
        throw std::runtime_error("Can only solve square systems");
    }
    if (B.rows() != rows_) {
        throw std::invalid_argument("Right-hand side matrix rows must match matrix rows");
    }
    
    Matrix result(rows_, B.cols());
    
    // Solve for each column of B
    for (size_t col = 0; col < B.cols(); ++col) {
        Vector b = B.get_column(col);
        Vector x = solve(b);
        result.set_column(col, x);
    }
    
    return result;
}

// Non-member operators
Matrix operator*(double scalar, const Matrix& matrix) {
    return matrix * scalar;
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    os << "[\n";
    for (size_t i = 0; i < matrix.rows(); ++i) {
        os << "  [";
        for (size_t j = 0; j < matrix.cols(); ++j) {
            if (j > 0) os << ", ";
            os << std::fixed << std::setprecision(6) << matrix(i, j);
        }
        os << "]";
        if (i < matrix.rows() - 1) os << ",";
        os << "\n";
    }
    os << "]";
    return os;
}

} // namespace core
} // namespace mlzero