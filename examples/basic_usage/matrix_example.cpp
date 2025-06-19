
#include "mlzero/mlzero.hpp"
#include <iostream>
#include <iomanip>

using namespace mlzero;


int main() {
    std::cout << "mlzero ML Framework - Linear Algebra Example\n";
    std::cout << "Demonstrating Vector and Matrix working together\n\n";
    
    // Create some data points for a simple linear regression
    std::cout << "=== Simple Linear Regression Setup ===\n";
    
    // Data: y = 2x + 1 + noise
    Vector x_data{1.0, 2.0, 3.0, 4.0, 5.0};
    Vector y_data{3.1, 4.9, 7.2, 9.1, 10.8};
    
    std::cout << "x data: " << x_data << "\n";
    std::cout << "y data: " << y_data << "\n\n";
    
    // Create design matrix for linear regression: [1, x]
    Matrix X(x_data.size(), 2);
    for (size_t i = 0; i < x_data.size(); ++i) {
        X(i, 0) = 1.0;      // Intercept term
        X(i, 1) = x_data[i]; // x value
    }
    
    std::cout << "Design matrix X:\n" << X << "\n\n";
    
    // Solve normal equations: X^T * X * β = X^T * y
    Matrix XtX = X.transpose() * X;
    Vector Xty = X.transpose() * y_data;
    
    std::cout << "X^T * X:\n" << XtX << "\n\n";
    std::cout << "X^T * y: " << Xty << "\n\n";
    
    Vector beta = XtX.solve(Xty);
    
    std::cout << "Solution β (coefficients): " << beta << "\n";
    std::cout << "Linear model: y = " << beta[1] << " * x + " << beta[0] << "\n\n";
    
    // Make predictions and compute residuals
    Vector predictions = X * beta;
    Vector residuals = y_data - predictions;
    
    std::cout << "Predictions: " << predictions << "\n";
    std::cout << "Residuals: " << residuals << "\n";
    std::cout << "Sum of squared residuals: " << residuals.dot(residuals) << "\n\n";
    
    // Demonstrate matrix-vector operations in ML context
    std::cout << "=== ML Operations with Vectors and Matrices ===\n";
    
    // Simulate a small neural network forward pass
    Matrix W1{{0.5, -0.3, 0.2},
              {0.1, 0.4, -0.1}};  // 2x3 weight matrix
    
    Vector b1{0.1, -0.2};           // 2x1 bias vector
    Vector input{1.0, 0.5, -0.3};  // 3x1 input vector
    
    std::cout << "Neural network layer:\n";
    std::cout << "Input: " << input << "\n";
    std::cout << "Weights W1:\n" << W1 << "\n\n";
    std::cout << "Bias b1: " << b1 << "\n\n";
    
    // Forward pass: output = W * input + bias
    Vector hidden = W1 * input + b1;
    std::cout << "Hidden layer output (before activation): " << hidden << "\n\n";
    
    // Apply ReLU activation (manually for demonstration)
    Vector activated(hidden.size());
    for (size_t i = 0; i < hidden.size(); ++i) {
        activated[i] = std::max(0.0, hidden[i]);
    }
    std::cout << "After ReLU activation: " << activated << "\n\n";
    
    // Demonstrate covariance matrix calculation
    std::cout << "=== Covariance Matrix Calculation ===\n";
    
    Matrix data{{1.0, 2.0, 3.0},
               {2.0, 4.0, 1.0},
               {3.0, 1.0, 5.0},
               {4.0, 3.0, 2.0}};  // 4 samples, 3 features
    
    std::cout << "Data matrix (4 samples, 3 features):\n" << data << "\n\n";
    
    // Center the data (subtract mean)
    Vector means = data.col_means();
    std::cout << "Feature means: " << means << "\n";
    
    Matrix centered(data.rows(), data.cols());
    for (size_t i = 0; i < data.rows(); ++i) {
        for (size_t j = 0; j < data.cols(); ++j) {
            centered(i, j) = data(i, j) - means[j];
        }
    }
    
    std::cout << "Centered data:\n" << centered << "\n\n";
    
    // Compute covariance matrix: C = (1/(n-1)) * X^T * X
    Matrix covariance = (centered.transpose() * centered) / (data.rows() - 1);
    
    std::cout << "Covariance matrix:\n" << covariance << "\n\n";
    
    // Principal component analysis (simplified)
    std::cout << "=== Principal Component Analysis ===\n";
    
    auto eigen_result = covariance.power_method();
    if (eigen_result.converged) {
        std::cout << "First principal component (largest eigenvalue): " << eigen_result.eigenvalue << "\n";
        std::cout << "First principal direction: " << eigen_result.eigenvector << "\n\n";
        
        // Project data onto first principal component
        Vector projections(data.rows());
        for (size_t i = 0; i < data.rows(); ++i) {
            Vector sample = centered.get_row(i);
            projections[i] = sample.dot(eigen_result.eigenvector);
        }
        
        std::cout << "Data projected onto first PC: " << projections << "\n\n";
    }
    
    // Distance calculations (useful for clustering)
    std::cout << "=== Distance Calculations ===\n";
    
    Vector point1{1.0, 2.0, 3.0};
    Vector point2{4.0, 5.0, 6.0};
    Vector point3{1.5, 2.5, 3.5};
    
    std::cout << "Point 1: " << point1 << "\n";
    std::cout << "Point 2: " << point2 << "\n";
    std::cout << "Point 3: " << point3 << "\n\n";
    
    std::cout << "Distance 1->2: " << point1.distance(point2) << "\n";
    std::cout << "Distance 1->3: " << point1.distance(point3) << "\n";
    std::cout << "Distance 2->3: " << point2.distance(point3) << "\n\n";
    
    // Angle between vectors
    std::cout << "Angle between point1 and point2: " << point1.angle(point2) << " radians\n";
    std::cout << "Angle between point1 and point3: " << point1.angle(point3) << " radians\n\n";
    
    std::cout << "Linear algebra example completed successfully!\n";
    std::cout << "Ready to implement machine learning algorithms!\n";
    

    return 0;
}
