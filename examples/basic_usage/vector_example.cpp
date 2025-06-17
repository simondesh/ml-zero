#include "mlzero/mlzero.hpp"
#include <iostream>

int main() {
    std::cout << "mlzero ML Framework - Vector Example\n";
    std::cout << "Version: " << mlzero::version() << "\n\n";
    
    // Create vectors using different methods
    std::cout << "=== Vector Creation ===\n";
    mlzero::Vector v1{1.0, 2.0, 3.0};
    mlzero::Vector v2 = mlzero::Vector::ones(3);
    mlzero::Vector v3 = mlzero::Vector::linspace(0.0, 2.0, 3);
    
    std::cout << "v1 = " << v1 << "\n";
    std::cout << "v2 = " << v2 << "\n";
    std::cout << "v3 = " << v3 << "\n\n";
    
    // Basic arithmetic operations
    std::cout << "=== Arithmetic Operations ===\n";
    mlzero::Vector sum = v1 + v2;
    mlzero::Vector diff = v1 - v2;
    mlzero::Vector scaled = v1 * 2.0;
    
    std::cout << "v1 + v2 = " << sum << "\n";
    std::cout << "v1 - v2 = " << diff << "\n";
    std::cout << "v1 * 2.0 = " << scaled << "\n\n";
    
    // Mathematical operations
    std::cout << "=== Mathematical Operations ===\n";
    double dot_product = v1.dot(v2);
    double magnitude = v1.magnitude();
    double distance = v1.distance(v2);
    
    std::cout << "v1 · v2 = " << dot_product << "\n";
    std::cout << "|v1| = " << magnitude << "\n";
    std::cout << "distance(v1, v2) = " << distance << "\n\n";
    
    // Normalization
    std::cout << "=== Normalization ===\n";
    mlzero::Vector normalized = v1.normalized();
    std::cout << "v1 normalized = " << normalized << "\n";
    std::cout << "magnitude of normalized = " << normalized.magnitude() << "\n\n";
    
    // Cross product (3D only)
    std::cout << "=== Cross Product ===\n";
    mlzero::Vector a{1.0, 0.0, 0.0};
    mlzero::Vector b{0.0, 1.0, 0.0};
    mlzero::Vector cross = a.cross(b);
    std::cout << "a = " << a << "\n";
    std::cout << "b = " << b << "\n";
    std::cout << "a × b = " << cross << "\n\n";
    
    // Statistical functions
    std::cout << "=== Statistical Functions ===\n";
    mlzero::Vector data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    std::cout << "data = " << data << "\n";
    std::cout << "sum = " << data.sum() << "\n";
    std::cout << "mean = " << data.mean() << "\n";
    std::cout << "min = " << data.min() << "\n";
    std::cout << "max = " << data.max() << "\n\n";
    
    // Dynamic operations
    std::cout << "=== Dynamic Operations ===\n";
    mlzero::Vector dynamic;
    std::cout << "Empty vector: " << dynamic << " (size: " << dynamic.size() << ")\n";
    
    for (int i = 1; i <= 5; ++i) {
        dynamic.push_back(i * i); // Add squares
    }
    std::cout << "After adding squares: " << dynamic << "\n";
    
    dynamic.pop_back();
    std::cout << "After pop_back: " << dynamic << "\n\n";
    
    // Slice operation
    std::cout << "=== Slice Operation ===\n";
    mlzero::Vector slice = data.slice(2, 7);
    std::cout << "data[2:7] = " << slice << "\n\n";
    
    // Random vector
    std::cout << "=== Random Vector ===\n";
    mlzero::Vector random = mlzero::Vector::random(5, -1.0, 1.0);
    std::cout << "Random vector [-1, 1]: " << random << "\n";
    std::cout << "Its magnitude: " << random.magnitude() << "\n\n";
    
    std::cout << "Vector example completed successfully!\n";
    return 0;
}