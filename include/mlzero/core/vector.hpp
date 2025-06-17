#ifndef MLZERO_CORE_VECTOR_HPP
#define MLZERO_CORE_VECTOR_HPP

#include <cstddef>
#include <initializer_list>
#include <stdexcept>

namespace mlzero {
namespace core {
class Vector {
private: 
    double* data_;
    size_t size_;
    size_t capacity_;

    void reallocate(size_t new_capacity);
    void copy_from(const Vector& other);
    void cleanup();

public: 
    // constructors and destructor
    Vector();                                               // Default constructor
    explicit Vector(size_t size);                           // Constructor with size 
    Vector(size_t size, double value);                      // Constructor with size and initial value
    Vector(std::initializer_list<double> values);           // Constructor with initializer list
    Vector(const Vector& other);                            // Copy constructor
    Vector(Vector&& other) noexcept;                        // Move constructor
    ~Vector();                                              // Destructor

    // assignment operators
    Vector& operator=(const Vector& other);                 // Copy assignment operator
    Vector& operator=(Vector&& other) noexcept;             // Move assignment operator

    // element access
    double& operator[](size_t index);                       // Non-const element access
    const double& operator[](size_t index) const;           // Const element access
    double& at(size_t index) ;                              // Non-const element access with bounds checking
    const double& at(size_t index) const;                   // Const element access with bounds checking

    // size and capacity
    size_t size() const {return size_;};                    // Get the number of elements
    size_t capacity() const {return capacity_;};            // Get the capacity of the vector
    bool empty() const {return size_ == 0;};                // Check if the vector is empty
    void reserve(size_t new_capacity);                      // Reserve capacity
    void resize(size_t new_size, double value = 0.0);       // Resize the vector

    // modifiers
    void push_back(double value);                           // Add an element to the end
    void pop_back();                                        // Remove the last element
    void clear();                                           // Clear the vector

    // operations
    Vector operator+(const Vector& other) const;            // Vector addition
    Vector operator-(const Vector& other) const;            // Vector subtraction
    Vector operator*(double scalar) const;                  // Scalar multiplication
    Vector operator/(double scalar) const;                  // Scalar division
    Vector& operator+=(const Vector& other);                // Vector addition assignment
    Vector& operator-=(const Vector& other);                // Vector subtraction assignment
    Vector& operator*=(double scalar);                      // Scalar multiplication assignment
    Vector& operator/=(double scalar);                      // Scalar division assignment

    // mathematical operations
    double dot(const Vector& other) const;                  // Dot product
    Vector cross(const Vector& other) const;                // Cross product (only for 3D vectors)
    double magnitude() const;                               // Magnitude of the vector
    double magnitude_squared() const;                       // Squared magnitude of the vector
    void normalize();                                       // Normalize the vector
    Vector normalized() const;                              // Return a normalized copy of the vector
    double distance(const Vector& other) const;             // Euclidean distance to another vector
    double angle(const Vector& other) const;                // Angle between two vectors in radians

    // comparison operators
    bool operator==(const Vector& other) const;             // Equality comparison
    bool operator!=(const Vector& other) const;             // Inequality comparison

    // utility functions
    void fill(double value);                                // Fill the vector with a specific value
    Vector slice(size_t start, size_t end) const;           // Get a slice of the vector
    double sum() const;                                     // Sum of all elements
    double mean() const;                                    // Mean of the elements
    double min() const;                                     // Minimum element
    double max() const;                                     // Maximum element

    // iterators
    double* begin() { return data_; }                       // Non-const iterator to the beginning
    const double* begin() const { return data_; }           // Const iterator to the beginning
    double* end() { return data_ + size_; }                 // Non-const iterator to the end
    const double* end() const { return data_ + size_; }     // Const iterator to the end

    //static utility functions
    static Vector zeros(size_t size);                       // Create a vector of zeros
    static Vector ones(size_t size);                        // Create a vector of ones
    static Vector random(size_t size, double min = 0.0, double max = 1.0); // Create a vector with random values
    static Vector linspace(double start, double end, size_t num_points); // Create a vector with evenly spaced values
};

// Non-member functions
Vector operator*(double scalar, const Vector& vec); // Scalar multiplication from the left
std::ostream& operator<<(std::ostream& os, const Vector& vec); // Output stream operator

} // namespace core
} // namespace mlzero

#endif // MLZERO_CORE_VECTOR_HPP