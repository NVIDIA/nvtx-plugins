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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// TODO(ahmadki): marker_id and domain handle should be uint64, but int64
// might cause op placement issues.
REGISTER_OP("NvtxStart")
    .Input("inputs: T")
    .Input("null_input: float32")
    .Input("message: string")
    .Input("domain_name: string")
    .Output("output: T")
    .Output("marker_id: int64")
    .Output("domain_handle: int64")
    .Attr("T: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data != nullptr) {
        c->set_output_handle_shapes_and_types(0, *handle_data);
      }
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
An identity graph node with a side effect of opening an NVTX marker.


Arguments
    inputs: A `Tensor` object that will be passed to `output`.
    null_input: A `float32 Tensor` object used as a trick to force gradient
                calculation. The tesnor is not used inside the op.
    message: A `String` message associated with this op.
    domain_name: A `String` domain name associated with this op.

Output
    output: The input `Tensor` passed to the output.
    marker_id: An NVTX marker id that is passed to `NvtxEnd`.
    domain_handle: An NVTX domain handler that is passed to `NvtxEnd`.
)doc");

REGISTER_OP("NvtxEnd")
    .Input("inputs: T")
    .Input("marker_id: int64")
    .Input("domain_handle: int64")
    .Input("grad_message: string")
    .Input("grad_domain_name: string")
    .Output("output: T")
    .Output("null_output: float32")
    .Attr("T: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data != nullptr) {
        c->set_output_handle_shapes_and_types(0, *handle_data);
      }
      c->set_output(1, c->Scalar());
      return Status::OK();
    })
    .Doc(R"doc(
An identity graph node with a side effect of closing an NVTX marker.


Arguments
    inputs: A `Tensor` object that will be passed to `output`.
    marker_id: An NVTX marker id that is recived from `NvtxStart`.
    domain_handle: An NVTX domain handler that is recived from `NvtxStart`.
    grad_message: A `String` message associated with this op gradient.
    grad_domain_name: A `String` domain name associated with this op gradient.

Output
    output: The input `Tensor` passed to the output.
    null_output: A `float32 Tensor` object used as a trick to force gradient
                 calculation. The tesnor is not used inside the op.
)doc");
