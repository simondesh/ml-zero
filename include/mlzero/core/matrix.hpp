#ifndef MLZERO_CORE_MATRIX_HPP
#define MLZERO_CORE_MATRIX_HPP

#include "mlzero/core/vector.hpp"
#include <cstddef>
#include <initializer_list>
#include <vector>
#include <stdexcept>

namespace mlzero {
namespace core {

class Matrix {
private:
    double** data_;
    size_t rows_;
    size_t cols_;
    
    void allocate_memory();
    void deallocate_memory();
    void copy_from(const Matrix& other);
    void check_bounds(size_t row, size_t col) const;
    void check_same_dimensions(const Matrix& other) const;

public:
    // Constructors and Destructor
    Matrix();                                                    // Default constructor
    Matrix(size_t rows, size_t cols);                           // Size constructor
    Matrix(size_t rows, size_t cols, double value);             // Size with initial value
    Matrix(std::initializer_list<std::initializer_list<double>> values); // 2D initializer list
    Matrix(const std::vector<std::vector<double>>& data);       // From 2D std::vector for convinience
    Matrix(const Matrix& other);                                // Copy constructor
    Matrix(Matrix&& other) noexcept;                           // Move constructor
    ~Matrix();                                                  // Destructor

    // Assignment operators
    Matrix& operator=(const Matrix& other);                    // Copy assignment
    Matrix& operator=(Matrix&& other) noexcept;               // Move assignment

    // Element access
    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;
    double& at(size_t row, size_t col);
    const double& at(size_t row, size_t col) const;
    
    // Row and column access
    Vector get_row(size_t row) const;
    Vector get_column(size_t col) const;
    void set_row(size_t row, const Vector& values);
    void set_column(size_t col, const Vector& values);

    // Dimensions
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return rows_ * cols_; }
    bool empty() const { return rows_ == 0 || cols_ == 0; }
    bool is_square() const { return rows_ == cols_; }
    
    // Matrix operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;               // Matrix multiplication
    Matrix operator*(double scalar) const;                     // Scalar multiplication
    Matrix operator/(double scalar) const;                     // Scalar division
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(const Matrix& other);
    Matrix& operator*=(double scalar);
    Matrix& operator/=(double scalar);

    // Matrix-Vector operations
    Vector operator*(const Vector& vec) const;                 // Matrix-vector multiplication
    
    // Matrix properties and operations
    Matrix transpose() const;
    Matrix& transpose_inplace();
    double trace() const;                                       // Sum of diagonal elements
    double determinant() const;                                 // Determinant calculation
    Matrix inverse() const;                                     // Matrix inversion
    bool is_symmetric(double tolerance = 1e-10) const;
    bool is_diagonal(double tolerance = 1e-10) const;
    bool is_identity(double tolerance = 1e-10) const;
    
    // Decompositions
    struct LUDecomposition;
    LUDecomposition lu_decomposition() const;
    
    struct QRDecomposition;
    QRDecomposition qr_decomposition() const;
    
    // System solving
    Vector solve(const Vector& b) const;                       // Solve Ax = b
    Matrix solve(const Matrix& B) const;                       // Solve AX = B
    
    // Eigenvalues (using power method for largest eigenvalue)
    struct EigenResult {
        double eigenvalue;
        Vector eigenvector;
        bool converged;
    };
    EigenResult power_method(double tolerance = 1e-10, size_t max_iterations = 1000) const;
    
    // Matrix norms
    double frobenius_norm() const;
    double one_norm() const;                                   // Maximum column sum
    double infinity_norm() const;                              // Maximum row sum
    
    // Comparison operators
    bool operator==(const Matrix& other) const;
    bool operator!=(const Matrix& other) const;
    
    // Utility functions
    void fill(double value);
    void fill_diagonal(double value);
    Matrix submatrix(size_t start_row, size_t end_row, size_t start_col, size_t end_col) const;
    void resize(size_t new_rows, size_t new_cols, double fill_value = 0.0);
    void swap_rows(size_t row1, size_t row2);
    void swap_cols(size_t col1, size_t col2);
    
    // Statistical functions
    Vector row_sums() const;
    Vector col_sums() const;
    Vector row_means() const;
    Vector col_means() const;
    double sum() const;
    double mean() const;
    
    // Static factory methods
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);
    static Matrix identity(size_t size);
    static Matrix diagonal(const Vector& diag_values);
    static Matrix random(size_t rows, size_t cols, double min = 0.0, double max = 1.0);
    static Matrix random_symmetric(size_t size, double min = 0.0, double max = 1.0);
    static Matrix vandermonde(const Vector& x, size_t degree);
    
    // Iterator support for range-based loops (row-wise iteration)
    class iterator {
    private:
        Matrix* matrix_;
        size_t index_;
    public:
        iterator(Matrix* matrix, size_t index) : matrix_(matrix), index_(index) {}
        double& operator*() { return matrix_->data_[index_ / matrix_->cols_][index_ % matrix_->cols_]; }
        iterator& operator++() { ++index_; return *this; }
        bool operator!=(const iterator& other) const { return index_ != other.index_; }
    };
    
    class const_iterator {
    private:
        const Matrix* matrix_;
        size_t index_;
    public:
        const_iterator(const Matrix* matrix, size_t index) : matrix_(matrix), index_(index) {}
        const double& operator*() const { return matrix_->data_[index_ / matrix_->cols_][index_ % matrix_->cols_]; }
        const_iterator& operator++() { ++index_; return *this; }
        bool operator!=(const const_iterator& other) const { return index_ != other.index_; }
    };
    
    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, rows_ * cols_); }
    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, rows_ * cols_); }
};

// LU and QR decomposition structs moved outside the Matrix class
struct Matrix::LUDecomposition {
    Matrix L, U;
    Vector permutation;
    bool is_singular;

    // Constructor for easier initialization
    LUDecomposition() : is_singular(false) {}
};

struct Matrix::QRDecomposition {
    Matrix Q, R;

    QRDecomposition() = default;
};

// Non-member operators
Matrix operator*(double scalar, const Matrix& matrix);
std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

} // namespace core
} // namespace mlzero

#endif // MLZERO_CORE_MATRIX_HPP