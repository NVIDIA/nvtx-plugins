#include "nvtx_custom_markers.h"

namespace nvtx_markers {

NVTX_DECLSPEC NvtxRangeDescriptor NVTX_API start_range(
  const std::string range_name, const std::string domain_name) {

  // get domain handle (create one if necessary)
  nvtxDomainHandle_t domain_handle = NVTX_DEFAULT_DOMAIN;

  if (!domain_name.empty()) {
    domain_handle = domain_registry.Register(domain_name);
  }

  nvtxEventAttributes_t eventAttrib = {};

  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;

  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = COLOR_RED;

  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = range_name.c_str();

  NvtxRangeDescriptor range_desc;
  range_desc.range_id = nvtxDomainRangeStartEx(domain_handle, &eventAttrib);
  range_desc.domain_handle = domain_handle;

  return range_desc;
}

NVTX_DECLSPEC void NVTX_API end_range(const nvtxRangeId_t range_id, const nvtxDomainHandle_t domain_handle) {
  nvtxDomainRangeEnd(domain_handle, range_id);
}

DomainRegistry::~DomainRegistry() {
  for (auto domain : domains) {
    nvtxDomainDestroy(domain.second);
  }
}

nvtxDomainHandle_t DomainRegistry::Register(const std::string &domain_name) {
#ifdef NEED_NVTX_INIT
  if (!initialized) {
    nvtxInitializationAttributes_t initAttribs = {};
    initAttribs.version = NVTX_VERSION;
    initAttribs.size = NVTX_INITIALIZATION_ATTRIB_STRUCT_SIZE;

    nvtxInitialize(&initAttribs);
    initialized = true;
  }
#endif

  auto it = domains.find(domain_name);
  if (it != domains.end()) {
    return it->second;
  }

  domains[domain_name] = nvtxDomainCreateA(domain_name.c_str());
  return domains[domain_name];
}

} // nvtx_markers
