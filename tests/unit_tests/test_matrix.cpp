
#include "catch_amalgamated.hpp"
#include "mlzero/core/matrix.hpp"
#include <sstream>
#include <mlzero/mlzero.hpp>

using namespace mlzero::core;

TEST_CASE("Matrix Construction", "[matrix][construction]") {
    SECTION("Default constructor") {
        Matrix m;
        REQUIRE(m.rows() == 0);
        REQUIRE(m.cols() == 0);
        REQUIRE(m.empty());
    }
    
    SECTION("Size constructor") {
        Matrix m(3, 4);
        REQUIRE(m.rows() == 3);
        REQUIRE(m.cols() == 4);
        REQUIRE(!m.empty());
        REQUIRE(m.size() == 12);
        
        // Should be initialized to zero
        for (size_t i = 0; i < m.rows(); ++i) {
            for (size_t j = 0; j < m.cols(); ++j) {
                REQUIRE(m(i, j) == Catch::Approx(0.0));
            }
        }
    }
    
    SECTION("Size with value constructor") {
        Matrix m(2, 3, 5.5);
        REQUIRE(m.rows() == 2);
        REQUIRE(m.cols() == 3);
        
        for (size_t i = 0; i < m.rows(); ++i) {
            for (size_t j = 0; j < m.cols(); ++j) {
                REQUIRE(m(i, j) == Catch::Approx(5.5));
            }
        }
    }
    
    SECTION("Initializer list constructor") {
        Matrix m{{1.0, 2.0, 3.0},
                 {4.0, 5.0, 6.0}};
        
        REQUIRE(m.rows() == 2);
        REQUIRE(m.cols() == 3);
        REQUIRE(m(0, 0) == Catch::Approx(1.0));
        REQUIRE(m(0, 1) == Catch::Approx(2.0));
        REQUIRE(m(0, 2) == Catch::Approx(3.0));
        REQUIRE(m(1, 0) == Catch::Approx(4.0));
        REQUIRE(m(1, 1) == Catch::Approx(5.0));
        REQUIRE(m(1, 2) == Catch::Approx(6.0));
    }
    
    SECTION("Vector constructor") {
        std::vector<std::vector<double>> data = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
        Matrix m(data);
        
        REQUIRE(m.rows() == 3);
        REQUIRE(m.cols() == 2);
        REQUIRE(m(0, 0) == Catch::Approx(1.0));
        REQUIRE(m(1, 0) == Catch::Approx(3.0));
        REQUIRE(m(2, 1) == Catch::Approx(6.0));
    }
    
    SECTION("Copy constructor") {
        Matrix original{{1.0, 2.0}, {3.0, 4.0}};
        Matrix copy(original);
        
        REQUIRE(copy.rows() == original.rows());
        REQUIRE(copy.cols() == original.cols());
        for (size_t i = 0; i < copy.rows(); ++i) {
            for (size_t j = 0; j < copy.cols(); ++j) {
                REQUIRE(copy(i, j) == Catch::Approx(original(i, j)));
            }
        }
        
        // Ensure deep copy
        copy(0, 0) = 99.0;
        REQUIRE(original(0, 0) == Catch::Approx(1.0));
    }
    
    SECTION("Move constructor") {
        Matrix original{{1.0, 2.0}, {3.0, 4.0}};
        size_t orig_rows = original.rows();
        size_t orig_cols = original.cols();
        
        Matrix moved(std::move(original));
        
        REQUIRE(moved.rows() == orig_rows);
        REQUIRE(moved.cols() == orig_cols);
        REQUIRE(moved(0, 0) == Catch::Approx(1.0));
        REQUIRE(moved(1, 1) == Catch::Approx(4.0));
        
        // Original should be empty after move
        REQUIRE(original.empty());
    }
    
    SECTION("Invalid initializer list should throw") {
        REQUIRE_THROWS_AS(Matrix({{1.0, 2.0}, {3.0}}), std::invalid_argument);
    }
}

TEST_CASE("Matrix Element Access", "[matrix][access]") {
    Matrix m{{1.0, 2.0, 3.0},
             {4.0, 5.0, 6.0}};
    
    SECTION("Operator() access") {
        REQUIRE(m(0, 0) == Catch::Approx(1.0));
        REQUIRE(m(0, 2) == Catch::Approx(3.0));
        REQUIRE(m(1, 1) == Catch::Approx(5.0));
        
        // Modification
        m(1, 2) = 99.0;
        REQUIRE(m(1, 2) == Catch::Approx(99.0));
    }
    
    SECTION("At() access with bounds checking") {
        REQUIRE(m.at(0, 0) == Catch::Approx(1.0));
        REQUIRE(m.at(1, 2) == Catch::Approx(6.0));
        
        // Out of bounds should throw
        REQUIRE_THROWS_AS(m.at(2, 0), std::out_of_range);
        REQUIRE_THROWS_AS(m.at(0, 3), std::out_of_range);
        REQUIRE_THROWS_AS(m.at(5, 5), std::out_of_range);
    }
    
    SECTION("Row and column access") {
        Vector row0 = m.get_row(0);
        REQUIRE(row0.size() == 3);
        REQUIRE(row0[0] == Catch::Approx(1.0));
        REQUIRE(row0[1] == Catch::Approx(2.0));
        REQUIRE(row0[2] == Catch::Approx(3.0));
        
        Vector col1 = m.get_column(1);
        REQUIRE(col1.size() == 2);
        REQUIRE(col1[0] == Catch::Approx(2.0));
        REQUIRE(col1[1] == Catch::Approx(5.0));
        
        // Out of bounds should throw
        REQUIRE_THROWS_AS(m.get_row(2), std::out_of_range);
        REQUIRE_THROWS_AS(m.get_column(3), std::out_of_range);
    }
    
    SECTION("Row and column setting") {
        Vector new_row{10.0, 20.0, 30.0};
        m.set_row(0, new_row);
        REQUIRE(m(0, 0) == Catch::Approx(10.0));
        REQUIRE(m(0, 1) == Catch::Approx(20.0));
        REQUIRE(m(0, 2) == Catch::Approx(30.0));
        
        Vector new_col{100.0, 200.0};
        m.set_column(2, new_col);
        REQUIRE(m(0, 2) == Catch::Approx(100.0));
        REQUIRE(m(1, 2) == Catch::Approx(200.0));
        
        // Size mismatch should throw
        Vector wrong_size{1.0, 2.0};
        REQUIRE_THROWS_AS(m.set_row(0, wrong_size), std::invalid_argument);
        
        Vector wrong_size2{1.0, 2.0, 3.0};
        REQUIRE_THROWS_AS(m.set_column(0, wrong_size2), std::invalid_argument);
    }
}

TEST_CASE("Matrix Properties", "[matrix][properties]") {
    SECTION("Square matrix detection") {
        Matrix square(3, 3);
        Matrix rect(3, 4);
        
        REQUIRE(square.is_square());
        REQUIRE_FALSE(rect.is_square());
    }
    
    SECTION("Identity matrix detection") {
        Matrix identity = Matrix::identity(3);
        Matrix non_identity{{1.0, 0.0}, {0.0, 2.0}};
        
        REQUIRE(identity.is_identity());
        REQUIRE_FALSE(non_identity.is_identity());
    }
    
    SECTION("Symmetric matrix detection") {
        Matrix symmetric{{1.0, 2.0, 3.0},
                        {2.0, 4.0, 5.0},
                        {3.0, 5.0, 6.0}};
        
        Matrix non_symmetric{{1.0, 2.0},
                            {3.0, 4.0}};
        
        REQUIRE(symmetric.is_symmetric());
        REQUIRE_FALSE(non_symmetric.is_symmetric());
    }
    
    SECTION("Diagonal matrix detection") {
        Matrix diagonal{{2.0, 0.0, 0.0},
                       {0.0, 3.0, 0.0},
                       {0.0, 0.0, 4.0}};
        
        Matrix non_diagonal{{1.0, 2.0},
                           {0.0, 3.0}};
        
        REQUIRE(diagonal.is_diagonal());
        REQUIRE_FALSE(non_diagonal.is_diagonal());
    }
}

TEST_CASE("Matrix Arithmetic Operations", "[matrix][arithmetic]") {
    Matrix m1{{1.0, 2.0},
              {3.0, 4.0}};
    
    Matrix m2{{5.0, 6.0},
              {7.0, 8.0}};
    
    SECTION("Matrix addition") {
        Matrix result = m1 + m2;
        REQUIRE(result(0, 0) == Catch::Approx(6.0));
        REQUIRE(result(0, 1) == Catch::Approx(8.0));
        REQUIRE(result(1, 0) == Catch::Approx(10.0));
        REQUIRE(result(1, 1) == Catch::Approx(12.0));
        
        // Original matrices should be unchanged
        REQUIRE(m1(0, 0) == Catch::Approx(1.0));
        REQUIRE(m2(0, 0) == Catch::Approx(5.0));
    }
    
    SECTION("Matrix subtraction") {
        Matrix result = m2 - m1;
        REQUIRE(result(0, 0) == Catch::Approx(4.0));
        REQUIRE(result(0, 1) == Catch::Approx(4.0));
        REQUIRE(result(1, 0) == Catch::Approx(4.0));
        REQUIRE(result(1, 1) == Catch::Approx(4.0));
    }
    
    SECTION("Scalar multiplication") {
        Matrix result = m1 * 2.0;
        REQUIRE(result(0, 0) == Catch::Approx(2.0));
        REQUIRE(result(0, 1) == Catch::Approx(4.0));
        REQUIRE(result(1, 0) == Catch::Approx(6.0));
        REQUIRE(result(1, 1) == Catch::Approx(8.0));
        
        // Test commutative property
        Matrix result2 = 2.0 * m1;
        REQUIRE(result == result2);
    }
    
    SECTION("Scalar division") {
        Matrix result = m1 / 2.0;
        REQUIRE(result(0, 0) == Catch::Approx(0.5));
        REQUIRE(result(0, 1) == Catch::Approx(1.0));
        REQUIRE(result(1, 0) == Catch::Approx(1.5));
        REQUIRE(result(1, 1) == Catch::Approx(2.0));
        
        // Division by zero should throw
        REQUIRE_THROWS_AS(m1 / 0.0, std::invalid_argument);
    }
    
    SECTION("Matrix multiplication") {
        Matrix result = m1 * m2;
        // [1 2] * [5 6] = [19 22]
        // [3 4]   [7 8]   [43 50]
        REQUIRE(result(0, 0) == Catch::Approx(19.0)); // 1*5 + 2*7
        REQUIRE(result(0, 1) == Catch::Approx(22.0)); // 1*6 + 2*8
        REQUIRE(result(1, 0) == Catch::Approx(43.0)); // 3*5 + 4*7
        REQUIRE(result(1, 1) == Catch::Approx(50.0)); // 3*6 + 4*8
    }
    
    SECTION("Compound assignment operators") {
        Matrix m = m1; // Copy
        
        m += m2;
        REQUIRE(m(0, 0) == Catch::Approx(6.0));
        REQUIRE(m(1, 1) == Catch::Approx(12.0));
        
        m -= m2;
        REQUIRE(m(0, 0) == Catch::Approx(1.0));
        REQUIRE(m(1, 1) == Catch::Approx(4.0));
        
        m *= 3.0;
        REQUIRE(m(0, 0) == Catch::Approx(3.0));
        REQUIRE(m(1, 1) == Catch::Approx(12.0));
        
        m /= 3.0;
        REQUIRE(m(0, 0) == Catch::Approx(1.0));
        REQUIRE(m(1, 1) == Catch::Approx(4.0));
    }
    
    SECTION("Dimension mismatch should throw") {
        Matrix m3(3, 2);
        REQUIRE_THROWS_AS(m1 + m3, std::invalid_argument);
        REQUIRE_THROWS_AS(m1 - m3, std::invalid_argument);
        
        Matrix m4(3, 3);
        REQUIRE_THROWS_AS(m1 * m4, std::invalid_argument); // 2x2 * 3x3 invalid
    }
}

TEST_CASE("Matrix-Vector Operations", "[matrix][vector]") {
    Matrix m{{1.0, 2.0, 3.0},
             {4.0, 5.0, 6.0}};
    
    Vector v{1.0, 2.0, 3.0};
    
    SECTION("Matrix-vector multiplication") {
        Vector result = m * v;
        REQUIRE(result.size() == 2);
        REQUIRE(result[0] == Catch::Approx(14.0)); // 1*1 + 2*2 + 3*3
        REQUIRE(result[1] == Catch::Approx(32.0)); // 4*1 + 5*2 + 6*3
    }
    
    SECTION("Dimension mismatch should throw") {
        Vector wrong_v{1.0, 2.0}; // Size 2, but matrix has 3 columns
        REQUIRE_THROWS_AS(m * wrong_v, std::invalid_argument);
    }
}

TEST_CASE("Matrix Transpose", "[matrix][transpose]") {
    Matrix m{{1.0, 2.0, 3.0},
             {4.0, 5.0, 6.0}};
    
    SECTION("Transpose operation") {
        Matrix t = m.transpose();
        REQUIRE(t.rows() == 3);
        REQUIRE(t.cols() == 2);
        REQUIRE(t(0, 0) == Catch::Approx(1.0));
        REQUIRE(t(0, 1) == Catch::Approx(4.0));
        REQUIRE(t(1, 0) == Catch::Approx(2.0));
        REQUIRE(t(1, 1) == Catch::Approx(5.0));
        REQUIRE(t(2, 0) == Catch::Approx(3.0));
        REQUIRE(t(2, 1) == Catch::Approx(6.0));
        
        // Original should be unchanged
        REQUIRE(m.rows() == 2);
        REQUIRE(m.cols() == 3);
    }
    
    SECTION("In-place transpose for square matrices") {
        Matrix square{{1.0, 2.0, 3.0},
                      {4.0, 5.0, 6.0},
                      {7.0, 8.0, 9.0}};
        
        square.transpose_inplace();
        REQUIRE(square(0, 0) == Catch::Approx(1.0));
        REQUIRE(square(0, 1) == Catch::Approx(4.0));
        REQUIRE(square(0, 2) == Catch::Approx(7.0));
        REQUIRE(square(1, 0) == Catch::Approx(2.0));
        REQUIRE(square(2, 0) == Catch::Approx(3.0));
        
        // Non-square matrices should throw
        REQUIRE_THROWS_AS(m.transpose_inplace(), std::runtime_error);
    }
    
    SECTION("Double transpose should return original") {
        Matrix original = m;
        Matrix double_transpose = m.transpose().transpose();
        REQUIRE(double_transpose == original);
    }
}

TEST_CASE("Matrix Determinant", "[matrix][determinant]") {
    SECTION("1x1 matrix") {
        Matrix m1{{5.0}};
        REQUIRE(m1.determinant() == Catch::Approx(5.0));
    }
    
    SECTION("2x2 matrix") {
        Matrix m2{{1.0, 2.0},
                  {3.0, 4.0}};
        REQUIRE(m2.determinant() == Catch::Approx(-2.0)); // 1*4 - 2*3
    }
    
    SECTION("3x3 matrix") {
        Matrix m3{{1.0, 2.0, 3.0},
                  {4.0, 5.0, 6.0},
                  {7.0, 8.0, 9.0}};
        REQUIRE(m3.determinant() == Catch::Approx(0.0)); // Singular matrix
        
        Matrix m3_regular{{1.0, 0.0, 2.0},
                         {-1.0, 5.0, 0.0},
                         {0.0, 3.0, -9.0}};
        REQUIRE(m3_regular.determinant() == Catch::Approx(-51.0));
    }
    
    SECTION("Identity matrix") {
        Matrix identity = Matrix::identity(4);
        REQUIRE(identity.determinant() == Catch::Approx(1.0));
    }
    
    SECTION("Non-square matrix should throw") {
        Matrix rect(2, 3);
        REQUIRE_THROWS_AS(rect.determinant(), std::runtime_error);
    }
}

TEST_CASE("Matrix Trace", "[matrix][trace]") {
    SECTION("Square matrix trace") {
        Matrix m{{1.0, 2.0, 3.0},
                 {4.0, 5.0, 6.0},
                 {7.0, 8.0, 9.0}};
        
        REQUIRE(m.trace() == Catch::Approx(15.0)); // 1 + 5 + 9
    }
    
    SECTION("Identity matrix trace") {
        Matrix identity = Matrix::identity(5);
        REQUIRE(identity.trace() == Catch::Approx(5.0));
    }
    
    SECTION("Non-square matrix should throw") {
        Matrix rect(2, 3);
        REQUIRE_THROWS_AS(rect.trace(), std::runtime_error);
    }
}

TEST_CASE("Matrix LU Decomposition", "[matrix][lu]") {
    SECTION("Regular matrix LU decomposition") {
        Matrix m{{2.0, 1.0, 1.0},
                 {4.0, 3.0, 3.0},
                 {8.0, 7.0, 9.0}};
        
        auto lu = m.lu_decomposition();
        REQUIRE_FALSE(lu.is_singular);
        
        // Verify that P*A = L*U (approximately)
        Matrix reconstructed = lu.L * lu.U;
        
        // Apply permutation to original matrix
        Matrix permuted(m.rows(), m.cols());
        for (size_t i = 0; i < m.rows(); ++i) {
            for (size_t j = 0; j < m.cols(); ++j) {
                permuted(i, j) = m(lu.permutation[i], j);
            }
        }
        
        // Check if L*U equals permuted A
        for (size_t i = 0; i < m.rows(); ++i) {
            for (size_t j = 0; j < m.cols(); ++j) {
                REQUIRE(reconstructed(i, j) == Catch::Approx(permuted(i, j)).margin(1e-10));
            }
        }
    }
    
    SECTION("Singular matrix") {
        Matrix singular{{1.0, 2.0, 3.0},
                       {4.0, 5.0, 6.0},
                       {7.0, 8.0, 9.0}};
        
        auto lu = singular.lu_decomposition();
        bool is_singular = lu.is_singular;
        REQUIRE(is_singular);
    }
    
    SECTION("Non-square matrix should throw") {
        Matrix rect(2, 3);
        REQUIRE_THROWS_AS(rect.lu_decomposition(), std::runtime_error);
    }
}

TEST_CASE("Matrix Inverse", "[matrix][inverse]") {
    SECTION("Invertible matrix") {
        Matrix m{{2.0, 1.0},
                 {1.0, 1.0}};
        Matrix inv = m.inverse();
        Matrix product = m * inv;
        Matrix identity = Matrix::identity(2);
        
        // m * m^(-1) should equal identity
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                REQUIRE(product(i, j) == Catch::Approx(identity(i, j)).margin(1e-10));
            }
        }
    }
    
    SECTION("Singular matrix should throw") {
        Matrix singular{{1.0, 2.0},
                       {2.0, 4.0}};

        REQUIRE(singular.rows() == 2 );
        REQUIRE(singular.cols() == 2 );

        REQUIRE(singular.get_row(0) == mlzero::Vector({1.0, 2.0}));
        
        REQUIRE_THROWS_AS(singular.inverse(), std::runtime_error);
    }
    
    SECTION("Non-square matrix should throw") {
        Matrix rect(2, 3);
        REQUIRE_THROWS_AS(rect.inverse(), std::runtime_error);
    }
}

TEST_CASE("Matrix System Solving", "[matrix][solve]") {
    SECTION("Solve linear system Ax = b") {
        Matrix A{{2.0, 1.0},
                 {1.0, 1.0}};
        Vector b{3.0, 2.0};
        
        Vector x = A.solve(b);
        Vector Ax = A * x;
        
        // Check that A*x = b
        REQUIRE(Ax[0] == Catch::Approx(b[0]).margin(1e-10));
        REQUIRE(Ax[1] == Catch::Approx(b[1]).margin(1e-10));
    }
    
    SECTION("Solve matrix equation AX = B") {
        Matrix A{{2.0, 1.0},
                 {1.0, 1.0}};
        Matrix B{{3.0, 1.0},
                 {2.0, 1.0}};
        
        Matrix X = A.solve(B);
        Matrix AX = A * X;
        
        // Check that A*X = B
        for (size_t i = 0; i < A.rows(); ++i) {
            for (size_t j = 0; j < B.cols(); ++j) {
                REQUIRE(AX(i, j) == Catch::Approx(B(i, j)).margin(1e-10));
            }
        }
    }
    
    SECTION("Singular system should throw") {
        Matrix singular{{1.0, 2.0},
                       {2.0, 4.0}};
        Vector b{1.0, 2.0};
        
        REQUIRE_THROWS_AS(singular.solve(b), std::runtime_error);
    }
    
    SECTION("Dimension mismatch should throw") {
        Matrix A(3, 3);
        Vector wrong_b(2);
        
        REQUIRE_THROWS_AS(A.solve(wrong_b), std::invalid_argument);
    }
}

TEST_CASE("Matrix Norms", "[matrix][norms]") {
    Matrix m{{1.0, -2.0},
             {3.0, -4.0}};
    
    SECTION("Frobenius norm") {
        double frobenius = m.frobenius_norm();
        REQUIRE(frobenius == Catch::Approx(std::sqrt(30.0))); // sqrt(1² + 2² + 3² + 4²)
    }
    
    SECTION("One norm (maximum column sum)") {
        double one_norm = m.one_norm();
        REQUIRE(one_norm == Catch::Approx(6.0)); // max(|1|+|3|, |-2|+|-4|) = max(4, 6) = 6
    }
    
    SECTION("Infinity norm (maximum row sum)") {
        double inf_norm = m.infinity_norm();
        REQUIRE(inf_norm == Catch::Approx(7.0)); // max(|1|+|-2|, |3|+|-4|) = max(3, 7) = 7
    }
}

TEST_CASE("Matrix Power Method", "[matrix][eigenvalues]") {
    SECTION("Power method for dominant eigenvalue") {
        Matrix A{{3.0, 1.0},
                 {1.0, 2.0}};
        
        auto result = A.power_method(1e-6, 1000);
        
        // For this matrix, largest eigenvalue should be approximately (5 + sqrt(5))/2 ≈ 3.618
        double expected_eigenvalue = (5.0 + std::sqrt(5.0)) / 2.0;
        REQUIRE(result.converged);
        REQUIRE(result.eigenvalue == Catch::Approx(expected_eigenvalue).margin(1e-3));
        
        // Check eigenvector property: A*v = λ*v
        Vector Av = A * result.eigenvector;
        Vector lambda_v = result.eigenvector * result.eigenvalue;
        
        for (size_t i = 0; i < Av.size(); ++i) {
            REQUIRE(Av[i] == Catch::Approx(lambda_v[i]).margin(1e-3));
        }
    }
    
    SECTION("Non-square matrix should throw") {
        Matrix rect(2, 3);
        REQUIRE_THROWS_AS(rect.power_method(), std::runtime_error);
    }
}

TEST_CASE("Matrix QR Decomposition", "[matrix][qr]") {
    SECTION("QR decomposition") {
        Matrix A{{1.0, 1.0, 0.0},
                 {1.0, 0.0, 1.0},
                 {0.0, 1.0, 1.0}};
        
        auto qr = A.qr_decomposition();
        
        // Check that Q is orthogonal (Q^T * Q = I)
        Matrix QtQ = qr.Q.transpose() * qr.Q;
        Matrix identity = Matrix::identity(3);
        
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                REQUIRE(QtQ(i, j) == Catch::Approx(identity(i, j)).margin(1e-10));
            }
        }
        
        // Check that Q * R = A
        Matrix QR = qr.Q * qr.R;
        for (size_t i = 0; i < A.rows(); ++i) {
            for (size_t j = 0; j < A.cols(); ++j) {
                REQUIRE(QR(i, j) == Catch::Approx(A(i, j)).margin(1e-10));
            }
        }
    }
}

TEST_CASE("Matrix Utility Functions", "[matrix][utility]") {
    Matrix m{{1.0, 2.0, 3.0},
             {4.0, 5.0, 6.0}};
    
    SECTION("Fill operations") {
        Matrix test(2, 2);
        test.fill(7.5);
        
        for (size_t i = 0; i < test.rows(); ++i) {
            for (size_t j = 0; j < test.cols(); ++j) {
                REQUIRE(test(i, j) == Catch::Approx(7.5));
            }
        }
        
        Matrix square(3, 3);
        square.fill_diagonal(9.0);
        REQUIRE(square(0, 0) == Catch::Approx(9.0));
        REQUIRE(square(1, 1) == Catch::Approx(9.0));
        REQUIRE(square(2, 2) == Catch::Approx(9.0));
        REQUIRE(square(0, 1) == Catch::Approx(0.0));
        
        // fill_diagonal on non-square should throw
        REQUIRE_THROWS_AS(m.fill_diagonal(1.0), std::runtime_error);
    }
    
    SECTION("Submatrix extraction") {
        Matrix sub = m.submatrix(0, 2, 1, 3);
        REQUIRE(sub.rows() == 2);
        REQUIRE(sub.cols() == 2);
        REQUIRE(sub(0, 0) == Catch::Approx(2.0));
        REQUIRE(sub(0, 1) == Catch::Approx(3.0));
        REQUIRE(sub(1, 0) == Catch::Approx(5.0));
        REQUIRE(sub(1, 1) == Catch::Approx(6.0));
        
        // Invalid ranges should throw
        REQUIRE_THROWS_AS(m.submatrix(1, 0, 0, 2), std::out_of_range);
        REQUIRE_THROWS_AS(m.submatrix(0, 3, 0, 2), std::out_of_range);
    }
    
    SECTION("Row and column swapping") {
        Matrix test = m;
        test.swap_rows(0, 1);
        REQUIRE(test(0, 0) == Catch::Approx(4.0));
        REQUIRE(test(1, 0) == Catch::Approx(1.0));
        
        test.swap_cols(0, 2);
        REQUIRE(test(0, 0) == Catch::Approx(6.0));
        REQUIRE(test(0, 2) == Catch::Approx(4.0));
        
        // Out of bounds should throw
        REQUIRE_THROWS_AS(test.swap_rows(0, 5), std::out_of_range);
        REQUIRE_THROWS_AS(test.swap_cols(0, 5), std::out_of_range);
    }
    
    SECTION("Resize operation") {
        Matrix test(2, 2, 1.0);
        test.resize(3, 4, 9.0);
        
        REQUIRE(test.rows() == 3);
        REQUIRE(test.cols() == 4);
        
        // Original data should be preserved
        REQUIRE(test(0, 0) == Catch::Approx(1.0));
        REQUIRE(test(1, 1) == Catch::Approx(1.0));
        
        // New elements should have fill value
        REQUIRE(test(0, 2) == Catch::Approx(9.0));
        REQUIRE(test(2, 0) == Catch::Approx(9.0));
    }
}

TEST_CASE("Matrix Statistical Functions", "[matrix][statistics]") {
    Matrix m{{1.0, 2.0, 3.0},
             {4.0, 5.0, 6.0}};
    
    SECTION("Row and column sums") {
        Vector row_sums = m.row_sums();
        REQUIRE(row_sums.size() == 2);
        REQUIRE(row_sums[0] == Catch::Approx(6.0));  // 1+2+3
        REQUIRE(row_sums[1] == Catch::Approx(15.0)); // 4+5+6
        
        Vector col_sums = m.col_sums();
        REQUIRE(col_sums.size() == 3);
        REQUIRE(col_sums[0] == Catch::Approx(5.0));  // 1+4
        REQUIRE(col_sums[1] == Catch::Approx(7.0));  // 2+5
        REQUIRE(col_sums[2] == Catch::Approx(9.0));  // 3+6
    }
    
    SECTION("Row and column means") {
        Vector row_means = m.row_means();
        REQUIRE(row_means.size() == 2);
        REQUIRE(row_means[0] == Catch::Approx(2.0));  // 6/3
        REQUIRE(row_means[1] == Catch::Approx(5.0));  // 15/3
        
        Vector col_means = m.col_means();
        REQUIRE(col_means.size() == 3);
        REQUIRE(col_means[0] == Catch::Approx(2.5));  // 5/2
        REQUIRE(col_means[1] == Catch::Approx(3.5));  // 7/2
        REQUIRE(col_means[2] == Catch::Approx(4.5));  // 9/2
    }
    
    SECTION("Total sum and mean") {
        REQUIRE(m.sum() == Catch::Approx(21.0));   // 1+2+3+4+5+6
        REQUIRE(m.mean() == Catch::Approx(3.5));   // 21/6
        
        Matrix empty;
        REQUIRE_THROWS_AS(empty.mean(), std::runtime_error);
    }
}

TEST_CASE("Matrix Factory Methods", "[matrix][factory]") {
    SECTION("zeros") {
        Matrix zeros = Matrix::zeros(2, 3);
        REQUIRE(zeros.rows() == 2);
        REQUIRE(zeros.cols() == 3);
        
        for (size_t i = 0; i < zeros.rows(); ++i) {
            for (size_t j = 0; j < zeros.cols(); ++j) {
                REQUIRE(zeros(i, j) == Catch::Approx(0.0));
            }
        }
    }
    
    SECTION("ones") {
        Matrix ones = Matrix::ones(3, 2);
        REQUIRE(ones.rows() == 3);
        REQUIRE(ones.cols() == 2);
        
        for (size_t i = 0; i < ones.rows(); ++i) {
            for (size_t j = 0; j < ones.cols(); ++j) {
                REQUIRE(ones(i, j) == Catch::Approx(1.0));
            }
        }
    }
    
    SECTION("identity") {
        Matrix identity = Matrix::identity(3);
        REQUIRE(identity.rows() == 3);
        REQUIRE(identity.cols() == 3);
        REQUIRE(identity.is_identity());
    }
    
    SECTION("diagonal") {
        Vector diag_vals{1.0, 2.0, 3.0};
        Matrix diag = Matrix::diagonal(diag_vals);
        
        REQUIRE(diag.rows() == 3);
        REQUIRE(diag.cols() == 3);
        REQUIRE(diag(0, 0) == Catch::Approx(1.0));
        REQUIRE(diag(1, 1) == Catch::Approx(2.0));
        REQUIRE(diag(2, 2) == Catch::Approx(3.0));
        REQUIRE(diag(0, 1) == Catch::Approx(0.0));
        REQUIRE(diag.is_diagonal());
    }
    
    SECTION("random") {
        Matrix random = Matrix::random(3, 4, -1.0, 1.0);
        REQUIRE(random.rows() == 3);
        REQUIRE(random.cols() == 4);
        
        // Check that values are in range
        for (size_t i = 0; i < random.rows(); ++i) {
            for (size_t j = 0; j < random.cols(); ++j) {
                REQUIRE(random(i, j) >= -1.0);
                REQUIRE(random(i, j) <= 1.0);
            }
        }
    }
    
    SECTION("random_symmetric") {
        Matrix sym = Matrix::random_symmetric(4, 0.0, 1.0);
        REQUIRE(sym.rows() == 4);
        REQUIRE(sym.cols() == 4);
        REQUIRE(sym.is_symmetric());
    }
    
    SECTION("vandermonde") {
        Vector x{1.0, 2.0, 3.0};
        Matrix vand = Matrix::vandermonde(x, 2);
        
        REQUIRE(vand.rows() == 3);
        REQUIRE(vand.cols() == 3);
        
        // Check structure: [1 x x²]
        REQUIRE(vand(0, 0) == Catch::Approx(1.0));
        REQUIRE(vand(0, 1) == Catch::Approx(1.0));
        REQUIRE(vand(0, 2) == Catch::Approx(1.0));
        
        REQUIRE(vand(1, 0) == Catch::Approx(1.0));
        REQUIRE(vand(1, 1) == Catch::Approx(2.0));
        REQUIRE(vand(1, 2) == Catch::Approx(4.0));
        
        REQUIRE(vand(2, 0) == Catch::Approx(1.0));
        REQUIRE(vand(2, 1) == Catch::Approx(3.0));
        REQUIRE(vand(2, 2) == Catch::Approx(9.0));
    }
}

TEST_CASE("Matrix Comparison", "[matrix][comparison]") {
    Matrix m1{{1.0, 2.0}, {3.0, 4.0}};
    Matrix m2{{1.0, 2.0}, {3.0, 4.0}};
    Matrix m3{{1.0, 2.0}, {3.0, 4.1}};
    Matrix m4{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    
    SECTION("Equality comparison") {
        REQUIRE(m1 == m2);
        REQUIRE_FALSE(m1 == m3);
        REQUIRE_FALSE(m1 == m4); // Different dimensions
        
        // Test numerical tolerance
        Matrix m5{{1.0, 2.0}, {3.0, 4.0 + 1e-16}};
        REQUIRE(m1 == m5); // Should be equal within tolerance
    }
    
    SECTION("Inequality comparison") {
        REQUIRE_FALSE(m1 != m2);
        REQUIRE(m1 != m3);
        REQUIRE(m1 != m4);
    }
}

TEST_CASE("Matrix Iterator Support", "[matrix][iterator]") {
    Matrix m{{1.0, 2.0, 3.0},
             {4.0, 5.0, 6.0}};
    
    SECTION("Range-based for loop") {
        double sum = 0.0;
        for (double value : m) {
            sum += value;
        }
        REQUIRE(sum == Catch::Approx(21.0));
    }
    
    SECTION("Iterator modification") {
        for (double& value : m) {
            value *= 2.0;
        }
        
        REQUIRE(m(0, 0) == Catch::Approx(2.0));
        REQUIRE(m(0, 1) == Catch::Approx(4.0));
        REQUIRE(m(1, 2) == Catch::Approx(12.0));
    }
    
    SECTION("Const iterator") {
        const Matrix& const_m = m;
        double sum = 0.0;
        for (const double& value : const_m) {
            sum += value;
        }
        REQUIRE(sum == Catch::Approx(21.0));
    }
}

TEST_CASE("Matrix Stream Output", "[matrix][io]") {
    Matrix m{{1.234567, 2.345678},
             {3.456789, 4.567890}};
    
    std::ostringstream oss;
    oss << m;
    
    std::string output = oss.str();
    
    // Should contain brackets and structure
    REQUIRE(output.find('[') != std::string::npos);
    REQUIRE(output.find(']') != std::string::npos);
    REQUIRE(output.find(',') != std::string::npos);
    REQUIRE(output.find('\n') != std::string::npos);
    
    // Should contain the values (approximately)
    REQUIRE(output.find("1.234567") != std::string::npos);
    REQUIRE(output.find("4.567890") != std::string::npos);
}

TEST_CASE("Matrix Performance and Memory", "[matrix][performance]") {
    SECTION("Large matrix operations") {
        const size_t size = 100;
        
        Matrix m1 = Matrix::random(size, size, -1.0, 1.0);
        Matrix m2 = Matrix::random(size, size, -1.0, 1.0);
        
        REQUIRE(m1.rows() == size);
        REQUIRE(m1.cols() == size);
        
        // Basic operations should work
        Matrix sum = m1 + m2;
        REQUIRE(sum.rows() == size);
        REQUIRE(sum.cols() == size);
        
        // Matrix multiplication
        Matrix product = m1 * m2;
        REQUIRE(product.rows() == size);
        REQUIRE(product.cols() == size);
        
        double norm = m1.frobenius_norm();
        REQUIRE(norm > 0.0);
        REQUIRE(std::isfinite(norm));
    }
    
    SECTION("Move semantics efficiency") {
        Matrix m1{{1.0, 2.0}, {3.0, 4.0}};
        Matrix m2;
        
        m2 = std::move(m1);
        
        REQUIRE(m2.rows() == 2);
        REQUIRE(m2.cols() == 2);
        REQUIRE(m1.empty()); // m1 should be empty after move
    }
}