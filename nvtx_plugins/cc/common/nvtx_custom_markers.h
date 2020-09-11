#ifndef NVTX_CUSTOM_MARKERS_H_
#define NVTX_CUSTOM_MARKERS_H_

#include <map>
#include <string>

#include "nvtx3.hpp"

namespace nvtx_markers {

static const nvtx3::rgb COLOR_RED{255, 0, 0};
static const nvtx3::rgb COLOR_GREEN{0, 255, 0};
static const nvtx3::rgb COLOR_BLUE{0, 0, 255};
static const nvtx3::rgb COLOR_YELLOW{255, 255, 0};
static const nvtx3::rgb COLOR_MAGENTA{255, 0, 255};
static const nvtx3::rgb COLOR_CYAN{0, 255, 255};

struct NvtxCategory{
  char const* name;
  const uint32_t id;
};

struct _NvtxDefaultDomain{
  static constexpr char const* name{"nvtx-plugins-core"};
};

static const nvtxDomainHandle_t NvtxDefaultDomain =
  nvtx3::domain::get<_NvtxDefaultDomain>();

using named_category = nvtx3::named_category<_NvtxDefaultDomain>;

nvtxRangeId_t NVTX_API start_range(
  const std::string& range_name, const std::string& category_name);

void NVTX_API end_range(const nvtxRangeId_t range_id);

} // nvtx_markers

#endif //NVTX_CUSTOM_MARKERS_H_
