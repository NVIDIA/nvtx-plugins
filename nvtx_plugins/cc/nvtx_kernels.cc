/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <map>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

#include "nvToolsExt.h"


#define NVTX_DEFAULT_DOMAIN nullptr

using namespace tensorflow;

class DomainRegistry {
 public:
  DomainRegistry()
#ifdef NEED_NVTX_INIT
 : initialized(false)
#endif
  {
  }

  ~DomainRegistry() {
    for (auto domain : domains) {
      nvtxDomainDestroy(domain.second);
    }
  }

  nvtxDomainHandle_t Register(const string &domain_name) {
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

 private:
  std::map<string, nvtxDomainHandle_t> domains;
#ifdef NEED_NVTX_INIT
  bool initialized;
#endif
};

static DomainRegistry domain_registry;


template <typename T>
class NvtxStartOp : public OpKernel {
 public:
  explicit NvtxStartOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Ouput 0: Input => Output
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }

    // Inputs 1,2: message and domain_name
    const Tensor *message_t, *domain_t;
    OP_REQUIRES_OK(context, context->input("message", &message_t));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(message_t->shape()),
                errors::InvalidArgument("message must be scalar, but received ",
                                        message_t->shape().DebugString()));
    OP_REQUIRES_OK(context, context->input("domain_name", &domain_t));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(domain_t->shape()),
                errors::InvalidArgument("domain_name must be scalar, ",
                                        "but received ",
                                        domain_t->shape().DebugString()));
    const string message = message_t->flat<string>()(0);
    const string domain_name = domain_t->flat<string>()(0);

    // get domain handle (create one if necessary)
    nvtxDomainHandle_t domain_handle = NVTX_DEFAULT_DOMAIN;
    if (!domain_name.empty()) {
      domain_handle = domain_registry.Register(domain_name);
    }

    // create nvtx marker
    nvtxRangeId_t marker_id;
    if (domain_handle != NVTX_DEFAULT_DOMAIN) {
      nvtxEventAttributes_t attr = {};
      attr.version = NVTX_VERSION;
      // TODO(ahmadki): feature - ability to set the marker color
      attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
      attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
      attr.message.ascii = message.c_str();

      marker_id = nvtxDomainRangeStartEx(domain_handle, &attr);
    } else {
      marker_id = nvtxRangeStart(message.c_str());
    }

    // push marker_id and domain_handle to outputs 1 and 2
    Tensor *output_marker_id = nullptr, *output_domain_handle = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("marker_id",
                                            TensorShape({}),
                                            &output_marker_id)
                  );
    OP_REQUIRES_OK(context,
                   context->allocate_output("domain_handle",
                                            TensorShape({}),
                                            &output_domain_handle)
                  );
    output_marker_id->scalar<int64>()() = marker_id;
    output_domain_handle->scalar<int64>()() = (int64)domain_handle;
  }

  bool IsExpensive() override { return false; }
};

template <typename T>
class NvtxEndOp : public OpKernel {
 public:
  explicit NvtxEndOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Ouput 0: Input => Output
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }

    // Close NVTX range
    const Tensor *marker_t, *domain_t;
    OP_REQUIRES_OK(context, context->input("marker_id", &marker_t));
    OP_REQUIRES_OK(context, context->input("domain_handle", &domain_t));
    auto marker_id = marker_t->scalar<int64>()();
    auto domain_handle = reinterpret_cast<nvtxDomainHandle_t>(
      marker_t->scalar<int64>()());

    if (domain_handle != NVTX_DEFAULT_DOMAIN) {
      nvtxDomainRangeEnd(domain_handle, marker_id);
    } else {
      nvtxRangeEnd(marker_id);
    }

    Tensor *output_null_output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("null_output",
                                            TensorShape({}),
                                            &output_null_output)
                  );
  }

  bool IsExpensive() override { return false; }
};


#define REGISTER_GPU_KERNEL(type)                                 \
  REGISTER_KERNEL_BUILDER(Name("NvtxStart")                       \
                              .Device(DEVICE_GPU)                 \
                              .HostMemory("message")              \
                              .HostMemory("domain_name")          \
                              .HostMemory("marker_id")            \
                              .HostMemory("domain_handle")        \
                              .TypeConstraint<type>("T"),         \
                          NvtxStartOp<type>);                     \
  REGISTER_KERNEL_BUILDER(Name("NvtxEnd")                         \
                              .Device(DEVICE_GPU)                 \
                              .HostMemory("marker_id")            \
                              .HostMemory("domain_handle")        \
                              .HostMemory("grad_message")         \
                              .HostMemory("grad_domain_name")     \
                              .TypeConstraint<type>("T"),         \
                          NvtxEndOp<type>);

TF_CALL_NUMBER_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL
