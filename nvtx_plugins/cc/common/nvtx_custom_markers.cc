#include "nvtx_custom_markers.h"

namespace nvtx_markers {

NvtxRangeDescriptor NVTX_API start_range(
  const std::string range_name, const std::string category_name) {

  auto event_attributes = nvtx3::event_attributes{
    COLOR_RED,
    nvtx3::payload{0},
    range_name.c_str(),
    named_category{
      std::hash<std::string>{}(category_name),
      category_name.c_str()
    }
  };

  auto range_id = nvtxDomainRangeStartEx(
    NvtxDefaultDomain, event_attributes.get()
  );

  return NvtxRangeDescriptor {range_id, NvtxDefaultDomain};
}

void NVTX_API end_range(const nvtxRangeId_t range_id) {
  nvtxDomainRangeEnd(NvtxDefaultDomain, range_id);
}
//
//DomainRegistry::~DomainRegistry() {
//  for (auto domain : domains) {
//    nvtxDomainDestroy(domain.second);
//  }
//}
//
//nvtxDomainHandle_t DomainRegistry::Register(const std::string &domain_name) {
//#ifdef NEED_NVTX_INIT
//  if (!initialized) {
//    nvtxInitializationAttributes_t initAttribs = {};
//    initAttribs.version = NVTX_VERSION;
//    initAttribs.size = NVTX_INITIALIZATION_ATTRIB_STRUCT_SIZE;
//
//    nvtxInitialize(&initAttribs);
//    initialized = true;
//  }
//#endif
//
//  auto it = domains.find(domain_name);
//  if (it != domains.end()) {
//    return it->second;
//  }
//
//  domains[domain_name] = nvtxDomainCreateA(domain_name.c_str());
//  return domains[domain_name];
//}

} // nvtx_markers
