#ifndef NVTX_CUSTOM_MARKERS_H_
#define NVTX_CUSTOM_MARKERS_H_

#include <map>
#include <string>

#include "nvtx3.hpp"

//#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
//  TypeName(const TypeName&) = delete;         \
//  void operator=(const TypeName&) = delete

namespace nvtx_markers {

static const nvtx3::rgb COLOR_RED{255, 0, 0};
static const nvtx3::rgb COLOR_GREEN{0, 255, 0};
static const nvtx3::rgb COLOR_BLUE{0, 0, 255};
static const nvtx3::rgb COLOR_YELLOW{255, 255, 0};
static const nvtx3::rgb COLOR_MAGENTA{255, 0, 255};
static const nvtx3::rgb COLOR_CYAN{0, 255, 255};

struct NvtxRangeDescriptor {
  nvtxRangeId_t range_id;
  nvtxDomainHandle_t domain_handle;
};

struct NvtxCategory{
  char const* name;
  const uint32_t id;
};

struct _NvtxDefaultDomain{
  static constexpr char const* name{"nvtx-plugins"};
};

static const nvtxDomainHandle_t NvtxDefaultDomain =
  nvtx3::domain::get<_NvtxDefaultDomain>();

using named_category = nvtx3::named_category<_NvtxDefaultDomain>;

//class NvtxDomain {
// public:
//  explicit NvtxDomain(const char* name) : handle_(nvtxDomainCreateA(name)) {}
//  ~NvtxDomain() { nvtxDomainDestroy(handle_); }
//  operator nvtxDomainHandle_t() const { return handle_; }
//
// private:
//  DISALLOW_COPY_AND_ASSIGN(NvtxDomain);
//  nvtxDomainHandle_t handle_;
//};



//class DomainRegistry {
// public:
//  DomainRegistry()
//#ifdef NEED_NVTX_INIT
// : initialized(false)
//#endif
//  {}
//
//  ~DomainRegistry();
//
//  nvtxDomainHandle_t Register(const std::string &domain_name);
//
// private:
//  std::map<std::string, nvtxDomainHandle_t> domains;
//#ifdef NEED_NVTX_INIT
//  bool initialized;
//#endif
//};

//static DomainRegistry domain_registry;

//static const NvtxDomain NVTX_DEFAULT_DOMAIN{"nvtx-plugins"};

NvtxRangeDescriptor NVTX_API start_range(
  const std::string range_name, const std::string category_name);

void NVTX_API end_range(const nvtxRangeId_t range_id);

}

#endif //NVTX_CUSTOM_MARKERS_H_
