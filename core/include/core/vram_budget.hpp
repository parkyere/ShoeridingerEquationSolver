#pragma once

// ses::GpuPrecision, kVramUnknown, choose_state_precision -- implemented in the
// ses.vram_budget module (core/src/vram_budget.ixx). Import shim for existing
// #include sites. The std include below MUST precede the import (MSVC redefines
// std if a GMF-std module is imported before a consumer textually includes std).
#include <cstdint>
import ses.vram_budget;
