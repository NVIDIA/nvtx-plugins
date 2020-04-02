#ifndef NVTX_CUSTOM_MARKERS_H_
#define NVTX_CUSTOM_MARKERS_H_

#include <map>
#include <string>

#include "nvToolsExt.h"

static const uint32_t COLOR_RED     = 0xFFFF0000;
static const uint32_t COLOR_GREEN   = 0xFF00FF00;
static const uint32_t COLOR_BLUE    = 0xFF0000FF;
static const uint32_t COLOR_YELLOW  = 0xFFFFFF00;
static const uint32_t COLOR_MAGENTA = 0xFFFF00FF;
static const uint32_t COLOR_CYAN    = 0xFF00FFFF;

#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;         \
  void operator=(const TypeName&) = delete

namespace nvtx_markers {

struct NvtxRangeDescriptor {
  nvtxRangeId_t range_id;
  nvtxDomainHandle_t domain_handle;
};

class NvtxDomain {
 public:
  explicit NvtxDomain(const char* name) : handle_(nvtxDomainCreateA(name)) {}
  ~NvtxDomain() { nvtxDomainDestroy(handle_); }
  operator nvtxDomainHandle_t() const { return handle_; }

 private:
  DISALLOW_COPY_AND_ASSIGN(NvtxDomain);
  nvtxDomainHandle_t handle_;
};



class DomainRegistry {
 public:
  DomainRegistry()
#ifdef NEED_NVTX_INIT
 : initialized(false)
#endif
  {}

  ~DomainRegistry();

  nvtxDomainHandle_t Register(const std::string &domain_name);

 private:
  std::map<std::string, nvtxDomainHandle_t> domains;
#ifdef NEED_NVTX_INIT
  bool initialized;
#endif
};

static DomainRegistry domain_registry;

static const NvtxDomain NVTX_DEFAULT_DOMAIN{"nvtx-plugins"};

NVTX_DECLSPEC NvtxRangeDescriptor NVTX_API start_range(
  const std::string range_name, const std::string domain_name);

NVTX_DECLSPEC void NVTX_API end_range(
  const nvtxRangeId_t range_id, const nvtxDomainHandle_t domain_handle);

}

#endif //NVTX_CUSTOM_MARKERS_H_
