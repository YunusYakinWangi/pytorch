#pragma once

#include <c10/macros/Export.h>

namespace c10 {

struct StorageImpl;

// Pluggable one-shot materialization hook for StorageImpl.
// Called on write-path access before returning a mutable pointer.
// The hook is cleared before invocation, so re-entrant access is safe.
using MaterializeFn = void (*)(StorageImpl*);

} // namespace c10
