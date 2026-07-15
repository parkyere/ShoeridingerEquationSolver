#pragma once

// ses::Rgb, phase_color, magnitude_color -- implemented in the ses.colormap
// module (core/src/colormap.ixx). Import shim for existing #include sites.
// The std includes below MUST precede the import: MSVC redefines std if a
// module whose global-module-fragment includes these is imported before a
// consumer textually includes them. Keeping them here forces textual-std-first.
#include <algorithm>
#include <cmath>
#include <numbers>
import ses.colormap;
