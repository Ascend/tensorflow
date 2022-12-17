//===- ComplexToStandard.h - Utils to convert from the complex dialect ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_COMPLEXTOSTANDARD_COMPLEXTOSTANDARD_H_
#define MLIR_CONVERSION_COMPLEXTOSTANDARD_COMPLEXTOSTANDARD_H_

#include <memory>

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class FuncOp;
class RewritePatternSet;
template <typename T>
class OperationPass;

/// Populate the given list with patterns that convert from Complex to Standard.
void populateComplexToStandardConversionPatterns(RewritePatternSet &patterns);

/// Create a pass to convert Complex operations to the Standard dialect.
std::unique_ptr<OperationPass<FuncOp>> createConvertComplexToStandardPass();

} // namespace mlir

#endif // MLIR_CONVERSION_COMPLEXTOSTANDARD_COMPLEXTOSTANDARD_H_
