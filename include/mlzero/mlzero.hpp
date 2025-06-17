

#ifndef MLZERO_HPP
#define MLZERO_HPP
#include "mlzero/core/vector.hpp"

#define MLZERO_VERSION_MAJOR 1
#define MLZERO_VERSION_MINOR 0
#define MLZERO_VERSION_PATCH 0

namespace mlzero {
    constexpr const char* version() {
        return "1.0.0";
    };

    using Vector = core::Vector;

}
#endif // MLZERO_HPP



//                                 We want something like this 
// namespace zerodep {
//     namespace core {
//         class Vector;
//         class Matrix;
//     }
    
//     namespace algorithms {
//         class LinearRegression;
//         class KNN;
//     }
    
//     namespace preprocessing {
//         class StandardScaler;
//     }
    
//     namespace metrics {
//         double accuracy(const Vector& y_true, const Vector& y_pred);
//     }
// }