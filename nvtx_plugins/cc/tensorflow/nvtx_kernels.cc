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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/public/version.h"

#include "../common/nvtx_custom_markers.h"


#define NVTX_DEFAULT_DOMAIN nullptr

using namespace tensorflow;


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

    // Inputs 1,2: message and category_name
    const Tensor *message_t, *category_t;
    OP_REQUIRES_OK(context, context->input("message", &message_t));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(message_t->shape()),
                errors::InvalidArgument("message must be scalar, but received ",
                                        message_t->shape().DebugString()));
    OP_REQUIRES_OK(context, context->input("category_name", &category_t));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(category_t->shape()),
                errors::InvalidArgument("category_name must be scalar, ",
                                        "but received ",
                                        category_t->shape().DebugString()));

#if TF_MAJOR_VERSION > 2 || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION >= 2)
    const string message = message_t->flat<tstring>()(0);
    const string category_name = category_t->flat<tstring>()(0);
#else
    const string message = message_t->flat<std::string>()(0);
    const string category_name = category_t->flat<std::string>()(0);
#endif

    // create nvtx marker
    const nvtx_markers::NvtxRangeDescriptor range_desc =
      nvtx_markers::start_range(message.c_str(), category_name.c_str());

    const nvtxRangeId_t marker_id = range_desc.range_id;
    const nvtxDomainHandle_t domain_handle = range_desc.domain_handle;


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
    const Tensor *marker_t, *category_t;

    OP_REQUIRES_OK(context, context->input("marker_id", &marker_t));
    OP_REQUIRES_OK(context, context->input("domain_handle", &category_t));

    auto marker_id = marker_t->scalar<int64>()();
    auto domain_handle = reinterpret_cast<nvtxDomainHandle_t>(
      marker_t->scalar<int64>()());

    nvtx_markers::end_range(marker_id);

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
                              .HostMemory("category_name")          \
                              .HostMemory("marker_id")            \
                              .HostMemory("domain_handle")        \
                              .TypeConstraint<type>("T"),         \
                          NvtxStartOp<type>);                     \
  REGISTER_KERNEL_BUILDER(Name("NvtxEnd")                         \
                              .Device(DEVICE_GPU)                 \
                              .HostMemory("marker_id")            \
                              .HostMemory("domain_handle")        \
                              .HostMemory("grad_message")         \
                              .HostMemory("grad_category_name")     \
                              .TypeConstraint<type>("T"),         \
                          NvtxEndOp<type>);

TF_CALL_NUMBER_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL
