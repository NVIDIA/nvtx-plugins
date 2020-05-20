#include <map>
#include "nvtx_custom_markers.h"

namespace nvtx_markers {

namespace detail {

inline uint32_t hash_str_uint32(const std::string& str) {
    uint32_t hash = 0x811c9dc5;
    // uint32_t prime = 0x1000193;
    uint32_t prime = 0x6C0B13B5;

    for(unsigned int i = 0; i < str.size(); ++i) {
        uint8_t value = str[i];
        hash = hash ^ value;
        hash *= prime;
    }
    return hash;
}

named_category const get_or_create_named_category (const std::string& name){
  static std::map<std::string, const named_category> named_cat_map {};
  auto named_cat_pair = named_cat_map.find(name);

  if (named_cat_pair == named_cat_map.end()) {
    auto named_cat = named_category{
      detail::hash_str_uint32(name),
      name.c_str()
    };

    named_cat_map.insert(
      std::pair<std::string, const named_category>(name, named_cat)
    );
    return named_cat;
  }

  return named_cat_pair->second;
}

}

nvtxRangeId_t NVTX_API start_range(
  const std::string& range_name, const std::string& category_name) {
    auto event_attributes = nvtx3::event_attributes{
      COLOR_RED,
      nvtx3::payload{0},
      range_name.c_str(),
      detail::get_or_create_named_category(category_name)
    };

    auto range_id = nvtxDomainRangeStartEx(
      NvtxDefaultDomain, event_attributes.get()
    );

    return range_id;
}

void NVTX_API end_range(const nvtxRangeId_t range_id) {
  nvtxDomainRangeEnd(NvtxDefaultDomain, range_id);
}

} // nvtx_markers
