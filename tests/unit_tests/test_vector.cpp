#include "catch_amalgamated.hpp"
#include "mlzero/core/vector.hpp"
#include <sstream>

using namespace mlzero::core;

TEST_CASE("Vector Construction", "[vector][construction]") {
    SECTION("Default constructor") {
        Vector v;
        REQUIRE(v.size() == 0);
        REQUIRE(v.empty());
    }
    
    SECTION("Size constructor") {
        Vector v(5);
        REQUIRE(v.size() == 5);
        REQUIRE(!v.empty());
        
        // Should be initialized to zero
        for (size_t i = 0; i < v.size(); ++i) {
            REQUIRE(v[i] == Catch::Approx(0.0));
        }
    }
    
    SECTION("Size with value constructor") {
        Vector v(3, 2.5);
        REQUIRE(v.size() == 3);
        
        for (size_t i = 0; i < v.size(); ++i) {
            REQUIRE(v[i] == Catch::Approx(2.5));
        }
    }
    
    SECTION("Initializer list constructor") {
        Vector v{1.0, 2.0, 3.0, 4.0};
        REQUIRE(v.size() == 4);
        REQUIRE(v[0] == Catch::Approx(1.0));
        REQUIRE(v[1] == Catch::Approx(2.0));
        REQUIRE(v[2] == Catch::Approx(3.0));
        REQUIRE(v[3] == Catch::Approx(4.0));
    }
    
    SECTION("Copy constructor") {
        Vector original{1.0, 2.0, 3.0};
        Vector copy(original);
        
        REQUIRE(copy.size() == original.size());
        for (size_t i = 0; i < copy.size(); ++i) {
            REQUIRE(copy[i] == Catch::Approx(original[i]));
        }
        
        // Ensure deep copy
        copy[0] = 99.0;
        REQUIRE(original[0] == Catch::Approx(1.0));
    }
    
    SECTION("Move constructor") {
        Vector original{1.0, 2.0, 3.0};
        size_t original_size = original.size();
        Vector moved(std::move(original));
        
        REQUIRE(moved.size() == original_size);
        REQUIRE(moved[0] == Catch::Approx(1.0));
        REQUIRE(moved[1] == Catch::Approx(2.0));
        REQUIRE(moved[2] == Catch::Approx(3.0));
        
        // Original should be empty after move
        REQUIRE(original.size() == 0);
    }
}
TEST_CASE("Vector Assignment", "[vector][assignment]") {
    SECTION("Copy assignment") {
        Vector v1{1.0, 2.0, 3.0};
        Vector v2;
        
        v2 = v1;
        
        REQUIRE(v2.size() == v1.size());
        for (size_t i = 0; i < v2.size(); ++i) {
            REQUIRE(v2[i] == Catch::Approx(v1[i]));
        }
        
        // Ensure deep copy
        v2[0] = 99.0;
        REQUIRE(v1[0] == Catch::Approx(1.0));
    }
    
    SECTION("Move assignment") {
        Vector v1{1.0, 2.0, 3.0};
        Vector v2;
        size_t original_size = v1.size();
        
        v2 = std::move(v1);
        
        REQUIRE(v2.size() == original_size);
        REQUIRE(v2[0] == Catch::Approx(1.0));
        REQUIRE(v2[1] == Catch::Approx(2.0));
        REQUIRE(v2[2] == Catch::Approx(3.0));
        
        // Original should be empty after move
        REQUIRE(v1.size() == 0);
    }
    
    SECTION("Self assignment") {
        
        Vector v{1.0, 2.0, 3.0};
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif
        v = v; // Self assignment
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
        
        REQUIRE(v.size() == 3);
        REQUIRE(v[0] == Catch::Approx(1.0));
        REQUIRE(v[1] == Catch::Approx(2.0));
        REQUIRE(v[2] == Catch::Approx(3.0));
    }
}

TEST_CASE("Vector Element Access", "[vector][access]") {
    Vector v{10.0, 20.0, 30.0};
    
    SECTION("Operator[] access") {
        REQUIRE(v[0] == Catch::Approx(10.0));
        REQUIRE(v[1] == Catch::Approx(20.0));
        REQUIRE(v[2] == Catch::Approx(30.0));
        
        // Modification
        v[1] = 25.0;
        REQUIRE(v[1] == Catch::Approx(25.0));
    }
    
    SECTION("At() access with bounds checking") {
        REQUIRE(v.at(0) == Catch::Approx(10.0));
        REQUIRE(v.at(1) == Catch::Approx(20.0));
        REQUIRE(v.at(2) == Catch::Approx(30.0));
        
        // Out of bounds should throw
        REQUIRE_THROWS_AS(v.at(3), std::out_of_range);
        REQUIRE_THROWS_AS(v.at(100), std::out_of_range);
    }
    
    SECTION("Const access") {
        const Vector& const_v = v;
        REQUIRE(const_v[0] == Catch::Approx(10.0));
        REQUIRE(const_v.at(1) == Catch::Approx(20.0));
        REQUIRE_THROWS_AS(const_v.at(3), std::out_of_range);
    }
}

TEST_CASE("Vector Capacity Operations", "[vector][capacity]") {
    SECTION("Basic capacity operations") {
        Vector v;
        REQUIRE(v.size() == 0);
        REQUIRE(v.empty());
        
        v.push_back(1.0);
        REQUIRE(v.size() == 1);
        REQUIRE(!v.empty());
        REQUIRE(v[0] == Catch::Approx(1.0));
    }
    
    SECTION("Multiple push_back operations") {
        Vector v;
        for (int i = 0; i < 10; ++i) {
            v.push_back(static_cast<double>(i));
        }
        
        REQUIRE(v.size() == 10);
        for (size_t i = 0; i < v.size(); ++i) {
            REQUIRE(v[i] == Catch::Approx(static_cast<double>(i)));
        }
    }
    
    SECTION("Pop back operations") {
        Vector v{1.0, 2.0, 3.0};
        
        v.pop_back();
        REQUIRE(v.size() == 2);
        REQUIRE(v[0] == Catch::Approx(1.0));
        REQUIRE(v[1] == Catch::Approx(2.0));
        
        v.pop_back();
        v.pop_back();
        REQUIRE(v.size() == 0);
        REQUIRE(v.empty());
        
        // Pop from empty vector should be safe
        REQUIRE_THROWS_AS(v.pop_back(), std::out_of_range);
        REQUIRE(v.size() == 0);
    }
    
    SECTION("Resize operations") {
        Vector v{1.0, 2.0, 3.0};
        
        // Resize larger
        v.resize(5, 9.0);
        REQUIRE(v.size() == 5);
        REQUIRE(v[0] == Catch::Approx(1.0));
        REQUIRE(v[1] == Catch::Approx(2.0));
        REQUIRE(v[2] == Catch::Approx(3.0));
        REQUIRE(v[3] == Catch::Approx(9.0));
        REQUIRE(v[4] == Catch::Approx(9.0));
        
        // Resize smaller
        v.resize(2);
        REQUIRE(v.size() == 2);
        REQUIRE(v[0] == Catch::Approx(1.0));
        REQUIRE(v[1] == Catch::Approx(2.0));
    }
    
    SECTION("Clear operation") {
        Vector v{1.0, 2.0, 3.0};
        v.clear();
        REQUIRE(v.size() == 0);
        REQUIRE(v.empty());
    }
}

TEST_CASE("Vector Arithmetic Operations", "[vector][arithmetic]") {
    Vector v1{1.0, 2.0, 3.0};
    Vector v2{4.0, 5.0, 6.0};
    
    SECTION("Vector addition") {
        Vector result = v1 + v2;
        REQUIRE(result.size() == 3);
        REQUIRE(result[0] == Catch::Approx(5.0));
        REQUIRE(result[1] == Catch::Approx(7.0));
        REQUIRE(result[2] == Catch::Approx(9.0));
        
        // Original vectors should be unchanged
        REQUIRE(v1[0] == Catch::Approx(1.0));
        REQUIRE(v2[0] == Catch::Approx(4.0));
    }
    
    SECTION("Vector subtraction") {
        Vector result = v2 - v1;
        REQUIRE(result.size() == 3);
        REQUIRE(result[0] == Catch::Approx(3.0));
        REQUIRE(result[1] == Catch::Approx(3.0));
        REQUIRE(result[2] == Catch::Approx(3.0));
    }
    
    SECTION("Scalar multiplication") {
        Vector result = v1 * 2.0;
        REQUIRE(result.size() == 3);
        REQUIRE(result[0] == Catch::Approx(2.0));
        REQUIRE(result[1] == Catch::Approx(4.0));
        REQUIRE(result[2] == Catch::Approx(6.0));
        
        // Test commutative property
        Vector result2 = 2.0 * v1;
        REQUIRE(result2.size() == 3);
        REQUIRE(result2[0] == Catch::Approx(2.0));
        REQUIRE(result2[1] == Catch::Approx(4.0));
        REQUIRE(result2[2] == Catch::Approx(6.0));
    }
    
    SECTION("Scalar division") {
        Vector result = v1 / 2.0;
        REQUIRE(result.size() == 3);
        REQUIRE(result[0] == Catch::Approx(0.5));
        REQUIRE(result[1] == Catch::Approx(1.0));
        REQUIRE(result[2] == Catch::Approx(1.5));
        
        // Division by zero should throw
        REQUIRE_THROWS_AS(v1 / 0.0, std::invalid_argument);
    }
    
    SECTION("Compound assignment operators") {
        Vector v = v1; // Copy
        
        v += v2;
        REQUIRE(v[0] == Catch::Approx(5.0));
        REQUIRE(v[1] == Catch::Approx(7.0));
        REQUIRE(v[2] == Catch::Approx(9.0));
        
        v -= v2;
        REQUIRE(v[0] == Catch::Approx(1.0));
        REQUIRE(v[1] == Catch::Approx(2.0));
        REQUIRE(v[2] == Catch::Approx(3.0));
        
        v *= 3.0;
        REQUIRE(v[0] == Catch::Approx(3.0));
        REQUIRE(v[1] == Catch::Approx(6.0));
        REQUIRE(v[2] == Catch::Approx(9.0));
        
        v /= 3.0;
        REQUIRE(v[0] == Catch::Approx(1.0));
        REQUIRE(v[1] == Catch::Approx(2.0));
        REQUIRE(v[2] == Catch::Approx(3.0));
    }
    
    SECTION("Size mismatch should throw") {
        Vector v3{1.0, 2.0}; // Different size
        REQUIRE_THROWS_AS(v1 + v3, std::invalid_argument);
        REQUIRE_THROWS_AS(v1 - v3, std::invalid_argument);
        REQUIRE_THROWS_AS(v1 += v3, std::invalid_argument);
        REQUIRE_THROWS_AS(v1 -= v3, std::invalid_argument);
    }
}

TEST_CASE("Vector Mathematical Operations", "[vector][mathematics]") {
    SECTION("Dot product") {
        Vector v1{1.0, 2.0, 3.0};
        Vector v2{4.0, 5.0, 6.0};
        
        double dot = v1.dot(v2);
        REQUIRE(dot == Catch::Approx(32.0)); // 1*4 + 2*5 + 3*6 = 32
        
        // Dot product should be commutative
        REQUIRE(v2.dot(v1) == Catch::Approx(32.0));
        
        // Dot product with itself equals magnitude squared
        double mag_sq = v1.magnitude_squared();
        REQUIRE(v1.dot(v1) == Catch::Approx(mag_sq));
        
        // Size mismatch should throw
        Vector v3{1.0, 2.0};
        REQUIRE_THROWS_AS(v1.dot(v3), std::invalid_argument);
    }
    
    SECTION("Cross product") {
        Vector v1{1.0, 0.0, 0.0};
        Vector v2{0.0, 1.0, 0.0};
        
        Vector cross = v1.cross(v2);
        REQUIRE(cross.size() == 3);
        REQUIRE(cross[0] == Catch::Approx(0.0));
        REQUIRE(cross[1] == Catch::Approx(0.0));
        REQUIRE(cross[2] == Catch::Approx(1.0));
        
        // Cross product should be anti-commutative
        Vector cross2 = v2.cross(v1);
        REQUIRE(cross2[0] == Catch::Approx(0.0));
        REQUIRE(cross2[1] == Catch::Approx(0.0));
        REQUIRE(cross2[2] == Catch::Approx(-1.0));
        
        // Non-3D vectors should throw
        Vector v3{1.0, 2.0};
        REQUIRE_THROWS_AS(v1.cross(v3), std::invalid_argument);
        REQUIRE_THROWS_AS(v3.cross(v1), std::invalid_argument);
    }
    
    SECTION("Magnitude calculations") {
        Vector v{3.0, 4.0};
        
        double mag_sq = v.magnitude_squared();
        REQUIRE(mag_sq == Catch::Approx(25.0)); // 3² + 4² = 25
        
        double mag = v.magnitude();
        REQUIRE(mag == Catch::Approx(5.0)); // √25 = 5
        
        // Zero vector
        Vector zero{0.0, 0.0, 0.0};
        REQUIRE(zero.magnitude() == Catch::Approx(0.0));
        REQUIRE(zero.magnitude_squared() == Catch::Approx(0.0));
    }
    
    SECTION("Normalization") {
        Vector v{3.0, 4.0};
        
        // Test normalized() - should not modify original
        Vector norm = v.normalized();
        REQUIRE(norm.magnitude() == Catch::Approx(1.0));
        REQUIRE(norm[0] == Catch::Approx(0.6)); // 3/5
        REQUIRE(norm[1] == Catch::Approx(0.8)); // 4/5
        
        // Original should be unchanged
        REQUIRE(v.magnitude() == Catch::Approx(5.0));
        
        // Test normalize() - should modify original
        v.normalize();
        REQUIRE(v.magnitude() == Catch::Approx(1.0));
        REQUIRE(v[0] == Catch::Approx(0.6));
        REQUIRE(v[1] == Catch::Approx(0.8));
        
        // Zero vector should throw
        Vector zero{0.0, 0.0};
        REQUIRE_THROWS_AS(zero.normalize(), std::runtime_error);
        REQUIRE_THROWS_AS(zero.normalized(), std::runtime_error);
    }
    
    SECTION("Distance calculation") {
        Vector v1{0.0, 0.0};
        Vector v2{3.0, 4.0};
        
        double dist = v1.distance(v2);
        REQUIRE(dist == Catch::Approx(5.0));
        
        // Distance should be symmetric
        REQUIRE(v2.distance(v1) == Catch::Approx(5.0));
        
        // Distance to self should be zero
        REQUIRE(v1.distance(v1) == Catch::Approx(0.0));
        
        // Size mismatch should throw
        Vector v3{1.0, 2.0, 3.0};
        REQUIRE_THROWS_AS(v1.distance(v3), std::invalid_argument);
    }
    
    SECTION("Angle calculation") {
        Vector v1{1.0, 0.0};
        Vector v2{0.0, 1.0};
        
        double angle = v1.angle(v2);
        REQUIRE(angle == Catch::Approx(M_PI / 2.0)); // 90 degrees
        
        // Parallel vectors
        Vector v3{2.0, 0.0};
        double angle_parallel = v1.angle(v3);
        REQUIRE(angle_parallel == Catch::Approx(0.0));
        
        // Anti-parallel vectors
        Vector v4{-1.0, 0.0};
        double angle_anti = v1.angle(v4);
        REQUIRE(angle_anti == Catch::Approx(M_PI));
        
        // Zero vector should throw
        Vector zero{0.0, 0.0};
        REQUIRE_THROWS_AS(v1.angle(zero), std::invalid_argument);
        REQUIRE_THROWS_AS(zero.angle(v1), std::invalid_argument);
    }
}

TEST_CASE("Vector Comparison Operations", "[vector][comparison]") {
    Vector v1{1.0, 2.0, 3.0};
    Vector v2{1.0, 2.0, 3.0};
    Vector v3{1.0, 2.0, 3.1};
    Vector v4{1.0, 2.0}; // Different size
    
    SECTION("Equality comparison") {
        REQUIRE(v1 == v2);
        REQUIRE_FALSE(v1 == v3);
        REQUIRE_FALSE(v1 == v4); // Different sizes
        
        // Test numerical tolerance
        Vector v5{1.0, 2.0, 3.0 + 1e-16}; // Very small difference
        REQUIRE(v1 == v5); // Should be equal within tolerance
    }
    
    SECTION("Inequality comparison") {
        REQUIRE_FALSE(v1 != v2);
        REQUIRE(v1 != v3);
        REQUIRE(v1 != v4);
    }
}

TEST_CASE("Vector Utility Functions", "[vector][utility]") {
    SECTION("Fill operation") {
        Vector v(5);
        v.fill(7.5);
        
        for (size_t i = 0; i < v.size(); ++i) {
            REQUIRE(v[i] == Catch::Approx(7.5));
        }
    }
    
    SECTION("Slice operation") {
        Vector v{1.0, 2.0, 3.0, 4.0, 5.0};
        
        Vector slice = v.slice(1, 4);
        REQUIRE(slice.size() == 3);
        REQUIRE(slice[0] == Catch::Approx(2.0));
        REQUIRE(slice[1] == Catch::Approx(3.0));
        REQUIRE(slice[2] == Catch::Approx(4.0));
        
        // Invalid slice should throw
        REQUIRE_THROWS_AS(v.slice(3, 2), std::out_of_range); // start >= end
        REQUIRE_THROWS_AS(v.slice(5, 6), std::out_of_range); // start >= size
        REQUIRE_THROWS_AS(v.slice(0, 6), std::out_of_range); // end > size
    }
    
    SECTION("Statistical functions") {
        Vector v{1.0, 2.0, 3.0, 4.0, 5.0};
        
        REQUIRE(v.sum() == Catch::Approx(15.0));
        REQUIRE(v.mean() == Catch::Approx(3.0));
        REQUIRE(v.min() == Catch::Approx(1.0));
        REQUIRE(v.max() == Catch::Approx(5.0));
        
        // Empty vector edge cases
        Vector empty;
        REQUIRE_THROWS_AS(empty.mean(), std::runtime_error);
        REQUIRE_THROWS_AS(empty.min(), std::runtime_error);
        REQUIRE_THROWS_AS(empty.max(), std::runtime_error);
        REQUIRE(empty.sum() == Catch::Approx(0.0)); // Sum of empty is 0
    }
}

TEST_CASE("Vector Static Factory Methods", "[vector][factory]") {
    SECTION("zeros") {
        Vector v = Vector::zeros(4);
        REQUIRE(v.size() == 4);
        for (size_t i = 0; i < v.size(); ++i) {
            REQUIRE(v[i] == Catch::Approx(0.0));
        }
    }
    
    SECTION("ones") {
        Vector v = Vector::ones(3);
        REQUIRE(v.size() == 3);
        for (size_t i = 0; i < v.size(); ++i) {
            REQUIRE(v[i] == Catch::Approx(1.0));
        }
    }
    
    SECTION("random") {
        Vector v = Vector::random(100, 0.0, 1.0);
        REQUIRE(v.size() == 100);
        
        // Check that values are in range
        for (size_t i = 0; i < v.size(); ++i) {
            REQUIRE(v[i] >= 0.0);
            REQUIRE(v[i] <= 1.0);
        }
        
        // Check that we get different values (very high probability)
        bool has_different_values = false;
        for (size_t i = 1; i < v.size(); ++i) {
            if (std::abs(v[i] - v[0]) > 1e-10) {
                has_different_values = true;
                break;
            }
        }
        REQUIRE(has_different_values);
    }
    
    SECTION("linspace") {
        Vector v = Vector::linspace(0.0, 10.0, 11);
        REQUIRE(v.size() == 11);
        REQUIRE(v[0] == Catch::Approx(0.0));
        REQUIRE(v[10] == Catch::Approx(10.0));
        
        // Check spacing
        for (size_t i = 0; i < v.size(); ++i) {
            REQUIRE(v[i] == Catch::Approx(static_cast<double>(i)));
        }
        
        // Edge cases
        Vector single = Vector::linspace(5.0, 5.0, 1);
        REQUIRE(single.size() == 1);
        REQUIRE(single[0] == Catch::Approx(5.0));
        
        Vector empty = Vector::linspace(0.0, 1.0, 0);
        REQUIRE(empty.size() == 0);
    }
}

TEST_CASE("Vector Iterator Support", "[vector][iterator]") {
    Vector v{1.0, 2.0, 3.0, 4.0, 5.0};
    
    SECTION("Range-based for loop") {
        double sum = 0.0;
        for (double value : v) {
            sum += value;
        }
        REQUIRE(sum == Catch::Approx(15.0));
    }
    
    SECTION("Iterator modification") {
        for (double& value : v) {
            value *= 2.0;
        }
        
        REQUIRE(v[0] == Catch::Approx(2.0));
        REQUIRE(v[1] == Catch::Approx(4.0));
        REQUIRE(v[2] == Catch::Approx(6.0));
        REQUIRE(v[3] == Catch::Approx(8.0));
        REQUIRE(v[4] == Catch::Approx(10.0));
    }
    
    SECTION("Const iterator") {
        const Vector& const_v = v;
        double sum = 0.0;
        for (const double& value : const_v) {
            sum += value;
        }
        REQUIRE(sum == Catch::Approx(15.0));
    }
}

TEST_CASE("Vector Stream Output", "[vector][io]") {
    Vector v{1.234567, 2.345678, 3.456789};
    
    std::ostringstream oss;
    oss << v;
    
    std::string output = oss.str();
    
    // Should contain brackets and commas
    REQUIRE(output.find('[') != std::string::npos);
    REQUIRE(output.find(']') != std::string::npos);
    REQUIRE(output.find(',') != std::string::npos);
    
    // Should contain the values (approximately)
    REQUIRE(output.find("1.23") != std::string::npos);
    REQUIRE(output.find("2.35") != std::string::npos);
    REQUIRE(output.find("3.46") != std::string::npos);
}

TEST_CASE("Vector Performance and Memory", "[vector][performance]") {
    SECTION("Large vector creation and operations") {
        const size_t large_size = 10000;
        
        Vector v1 = Vector::random(large_size, -1.0, 1.0);
        Vector v2 = Vector::random(large_size, -1.0, 1.0);
        
        REQUIRE(v1.size() == large_size);
        REQUIRE(v2.size() == large_size);
        
        // Basic operations should work
        Vector sum = v1 + v2;
        REQUIRE(sum.size() == large_size);
        
        double dot = v1.dot(v2);
        REQUIRE(std::isfinite(dot));
        
        double mag = v1.magnitude();
        REQUIRE(mag > 0.0);
        REQUIRE(std::isfinite(mag));
    }
    
    SECTION("Move semantics efficiency") {
        Vector v1{1.0, 2.0, 3.0, 4.0, 5.0};
        Vector v2;
        
        // Moving should be more efficient than copying
        v2 = std::move(v1);
        
        REQUIRE(v2.size() == 5);
        REQUIRE(v1.size() == 0); // v1 should be empty after move
    }
}