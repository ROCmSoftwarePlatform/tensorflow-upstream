/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/gpu/miopen_conv_algorithm_picker.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_comparator.h"
#include "tensorflow/compiler/xla/service/gpu/convolution_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/mutex.h"

namespace xla {
namespace gpu {
namespace {

using absl::optional;
using se::DeviceMemoryBase;
using se::dnn::AlgorithmDesc;
  
class ScratchAllocator : public se::ScratchAllocator {
 public:
  ScratchAllocator(int device_ordinal, se::DeviceMemoryAllocator* memory_allocator)
      : device_ordinal_(device_ordinal), memory_allocator_(memory_allocator) {}

  int64 GetMemoryLimitInBytes(se::Stream* stream) override {
    return 1LL << 32;  // 4GB.  TODO(jlebar): Tune this?
  }
  int64 TotalAllocatedBytes() { return total_allocated_bytes_; }

  StatusOr<se::DeviceMemory<uint8>> AllocateBytes(se::Stream* stream,
                                                  int64 byte_size) override;

 private:
  const int device_ordinal_;
  se::DeviceMemoryAllocator* memory_allocator_;
  std::vector<se::OwningDeviceMemory> allocated_buffers_;
  int64 total_allocated_bytes_ = 0;
};

StatusOr<se::DeviceMemory<uint8>> ScratchAllocator::AllocateBytes(
    se::Stream* stream, int64 byte_size) {
  CHECK_GE(byte_size, 0) << "byte_size must be positive.";
  if (byte_size > GetMemoryLimitInBytes(stream)) {
    return se::port::Status(
        se::port::error::RESOURCE_EXHAUSTED,
        absl::StrFormat(
            "Allocating %d bytes exceeds the memory limit of %d bytes.",
            byte_size, GetMemoryLimitInBytes(stream)));
  }

  TF_ASSIGN_OR_RETURN(se::OwningDeviceMemory allocated_buffer,
                      memory_allocator_->Allocate(device_ordinal_, byte_size,
                                                  /*retry_on_failure=*/false));
  total_allocated_bytes_ += byte_size;

  se::DeviceMemoryBase buffer_addr = *allocated_buffer;

  allocated_buffers_.push_back(std::move(allocated_buffer));
  return se::DeviceMemory<uint8>(buffer_addr);
}

StatusOr<se::dnn::ConvolutionKind> GetDnnConvolutionKind(
    const HloCustomCallInstruction* conv) {
  TF_ASSIGN_OR_RETURN(CudnnConvKind kind, GetCudnnConvKind(conv));

  switch (kind) {
    case CudnnConvKind::kBackwardFilter:
      return se::dnn::ConvolutionKind::BACKWARD_FILTER;
    case CudnnConvKind::kBackwardInput:
      return se::dnn::ConvolutionKind::BACKWARD_DATA;
    case CudnnConvKind::kForward:
      return se::dnn::ConvolutionKind::FORWARD;
    default:
      break;
  }
  return InternalError("Unsupported convolution type : %s", conv->ToString());
}

StatusOr<se::dnn::DataType> GetDnnDataType(
    const HloCustomCallInstruction* conv) {
  PrimitiveType output_primitive_type =
      conv->shape().tuple_shapes(0).element_type();

  switch (output_primitive_type) {
    case F16:
      return se::dnn::ToDataType<Eigen::half>::value;
    case F32:
      return se::dnn::ToDataType<float>::value;
    case F64:
      return se::dnn::ToDataType<double>::value;
    default:
      break;
  }

  return InternalError("Unsupported convolution datatype : %s",
                       conv->ToString());
}

StatusOr<std::vector<AlgorithmDesc>> GetAlgorithms(
    const HloCustomCallInstruction* conv,
    absl::Span<se::DeviceMemoryBase> operand_buffers,
    se::DeviceMemoryBase result_buffer, se::StreamExecutor* stream_exec,
    se::Stream* stream) {
  std::vector<AlgorithmDesc> algorithms;

  TF_ASSIGN_OR_RETURN(se::dnn::ConvolutionKind kind,
                      GetDnnConvolutionKind(conv));

  TF_ASSIGN_OR_RETURN(se::dnn::DataType dtype, GetDnnDataType(conv));

  TF_ASSIGN_OR_RETURN(CudnnConvParams params,
                      GetCudnnConvParams(conv, operand_buffers, result_buffer));

  bool succ = stream_exec->GetMIOpenConvolveAlgorithms(
      kind, stream, dtype, params.input_descriptor, params.filter_descriptor,
      params.conv_desc, params.output_descriptor, &algorithms);
  DCHECK(succ);

  return algorithms;
}

string AlgorithmToString(const AlgorithmDesc& algo) {
  if (algo.tensor_ops_enabled()) {
    return absl::StrCat(algo.algo_id(), "+TC");
  }
  return absl::StrCat(algo.algo_id());
}

string NumBytesToString(int64 bytes) {
  return absl::StrCat(tensorflow::strings::HumanReadableNumBytes(bytes), " (",
                      bytes, "B)");
}

// Acquires a process-global lock on the device pointed to by the given
// StreamExecutor.
//
// This is used to prevent other XLA instances from trying to autotune on this
// device while we're using it.
tensorflow::mutex_lock LockGpu(const se::StreamExecutor* stream_exec) {
  static tensorflow::mutex mu(tensorflow::LINKER_INITIALIZED);
  // se::Platform*s are global singletons guaranteed to live forever.
  static auto* mutexes =
      new std::map<std::pair<const se::Platform*, /*device_ordinal*/ int64>,
                   tensorflow::mutex>();

  tensorflow::mutex_lock global_lock(mu);
  auto it = mutexes
                ->emplace(std::piecewise_construct,
                          std::make_tuple(stream_exec->platform(),
                                          stream_exec->device_ordinal()),
                          std::make_tuple())
                .first;
  return tensorflow::mutex_lock{it->second};
}

using ConvCacheKey =
    std::tuple<se::StreamExecutor*, std::string, std::string, Shape,
               std::vector<Shape>, std::string, std::string, int64>;

struct ConvCacheStats {
  int64 cache_hits = 0;
  int64 cache_misses = 0;

  void LogStats() {
    VLOG(2) << "Cache hits: " << cache_hits;
    VLOG(2) << "Cache misses: " << cache_misses;
  }
};

StatusOr<ConvCacheKey> AutotuneCacheKeyfromInstruction(
    const HloCustomCallInstruction* conv, se::StreamExecutor* se) {
  TF_ASSIGN_OR_RETURN(CudnnConvBackendConfig backend_config,
                      conv->backend_config<CudnnConvBackendConfig>());
  std::vector<Shape> operand_shapes;
  absl::c_transform(conv->operands(), std::back_inserter(operand_shapes),
                    [&](const HloInstruction* op) { return op->shape(); });

  return std::make_tuple(
      se, backend_config.SerializeAsString(), conv->custom_call_target(),
      conv->shape(), std::move(operand_shapes),
      conv->window().SerializeAsString(),
      conv->convolution_dimension_numbers().SerializeAsString(),
      conv->feature_group_count());
}

tensorflow::mutex autotune_cache_lock(tensorflow::LINKER_INITIALIZED);
auto& autotune_cache GUARDED_BY(autotune_cache_lock) =
    *new absl::flat_hash_map<ConvCacheKey, MiopenConvAlgorithmPicker::AutotuneResult>();
auto& autotune_cache_stats GUARDED_BY(autotune_cache_lock) =
    *new ConvCacheStats();
}  // anonymous namespace

StatusOr<MiopenConvAlgorithmPicker::AutotuneResult>
MiopenConvAlgorithmPicker::PickBestAlgorithm(
    const HloCustomCallInstruction* instr) {
  // Don't run this function concurrently on the same GPU.
  //
  // This is a bit of a hack and doesn't protect us against arbitrary concurrent
  // use of a GPU, but it's sufficient to let us compile two HLO modules
  // concurrently and then run them sequentially.
  //
  // Putting the lock in here rather than in PickBestAlgorithmNoCache lets us
  // avoid ever doing duplicate work.  If we have a cache miss, only one thread
  // will run PickBestAlgorithmImpl for a particular device.
  tensorflow::mutex_lock lock = LockGpu(stream_exec_);

  // We cache the autotuning results to avoid doing the duplicate work,
  // which can greatly improve both stability (deterministic numeric results
  // within a process for a given input) and performance (2x speedup on some
  // models).
  TF_ASSIGN_OR_RETURN(ConvCacheKey key,
                      AutotuneCacheKeyfromInstruction(instr, stream_exec_));
  {
    tensorflow::mutex_lock lock(autotune_cache_lock);
    auto it = autotune_cache.find(key);
    if (it != autotune_cache.end()) {
      autotune_cache_stats.cache_hits++;
      return it->second;
    }
    autotune_cache_stats.cache_misses++;
  }

  StatusOr<MiopenConvAlgorithmPicker::AutotuneResult> result_or = PickBestAlgorithmNoCache(instr);
  if (result_or.ok()) {
    tensorflow::mutex_lock lock(autotune_cache_lock);
    CHECK(autotune_cache.insert({key, result_or.ValueOrDie()}).second);
  }
  return result_or;
}

StatusOr<MiopenConvAlgorithmPicker::AutotuneResult>
MiopenConvAlgorithmPicker::PickBestAlgorithmNoCache(
    const HloCustomCallInstruction* instr) {
  XLA_SCOPED_LOGGING_TIMER(
      absl::StrCat("CudnnConvAlgorithmPicker::PickBestAlgorithmImpl for ",
                   instr->ToString()));

  const Shape& result_shape = instr->shape().tuple_shapes(0);

  // Make sure any previous activity on this executor is done. We don't want to
  // interfere with programs that are still running on the GPU.
  if (!stream_exec_->SynchronizeAllActivity()) {
    return InternalError("Failed to synchronize GPU for autotuning.");
  }

  // Create a stream for us to do our work on.
  se::Stream stream{stream_exec_};
  stream.Init();
  const auto device_ordinal = stream_exec_->device_ordinal();

  // allocator either points to this->allocator_ or, if that's null, to a
  // StreamExecutorMemoryAllocator for stream_exec_.
  se::DeviceMemoryAllocator* allocator;
  optional<se::StreamExecutorMemoryAllocator> se_allocator;
  if (allocator_ != nullptr) {
    allocator = allocator_;
  } else {
    se_allocator.emplace(stream_exec_);
    allocator = &*se_allocator;
  }

  const auto initialize_buffer = [&stream](
                                     DeviceMemoryBase buffer) {
    // Although we don't have evidence this matters, zero out the buffers
    // before autotuning.  It's conceivable that using uninitialized memory as
    // the inputs might affect performance if e.g. the inputs contain
    // denormals, and this is easy enough.
    stream.ThenMemZero(&buffer, buffer.size());
  };

  // Allocate space for the input, filter, and output of the convolution.  We
  // use a ScratchAllocator for this instead of calling allocator_ directly so
  // that our allocations don't leak.
  ScratchAllocator input_output_allocator(device_ordinal, allocator);
  std::vector<se::DeviceMemoryBase> operand_buffers;
  for (const auto* operand : instr->operands()) {
    TF_ASSIGN_OR_RETURN(auto buffer,
                        input_output_allocator.AllocateBytes(
                            &stream, ShapeUtil::ByteSizeOf(operand->shape())));
    initialize_buffer(buffer);
    operand_buffers.push_back(buffer);
  }
  TF_ASSIGN_OR_RETURN(
      auto result_buffer,
      input_output_allocator.AllocateBytes(
          &stream, ShapeUtil::ByteSizeOf(instr->shape().tuple_shapes(0))));
  initialize_buffer(result_buffer);

  TF_ASSIGN_OR_RETURN(std::vector<AlgorithmDesc> algorithms,
                      GetAlgorithms(instr, absl::MakeSpan(operand_buffers),
                                    result_buffer, stream_exec_, &stream));

  std::vector<MiopenConvAlgorithmPicker::AutotuneResult> profile_results;

  for (const AlgorithmDesc& alg : algorithms) {
    XLA_SCOPED_LOGGING_TIMER_LEVEL(
        absl::StrCat("CudnnConvAlgorithmPicker::PickBestAlgorithm algo ",
                     AlgorithmToString(alg)),
        2);

    ScratchAllocator scratch_allocator(device_ordinal, allocator);
    se::dnn::ProfileResult profile_result;
    VLOG(3) << "Trying algorithm " << AlgorithmToString(alg) << " for "
            << instr->ToString();

    // Use assignment instead of brace-list to make GCC 4.9 happy.
    RunConvOptions options;
    options.profile_result = &profile_result;
    options.algo_override = alg;
    Status launch_status =
        RunCudnnConv(instr, absl::MakeSpan(operand_buffers), result_buffer,
                     &scratch_allocator, &stream, options);

    if (!launch_status.ok()) {
      continue;
    }

    if (!profile_result.is_valid()) {
      continue;
    }

    profile_results.emplace_back();
    MiopenConvAlgorithmPicker::AutotuneResult& result = profile_results.back();
    result.algorithm = alg.algo_id();
    result.tensor_ops_enabled = alg.tensor_ops_enabled();

    int64 scratch_bytes_used = scratch_allocator.TotalAllocatedBytes();
    result.scratch_bytes = scratch_bytes_used;
    result.runtime = absl::Milliseconds(profile_result.elapsed_time_in_ms());
  }

  const auto& best_result = absl::c_min_element(
      profile_results,
      [&](const MiopenConvAlgorithmPicker::AutotuneResult& lhs, const MiopenConvAlgorithmPicker::AutotuneResult& rhs) {
        return lhs.runtime < rhs.runtime;
      });

  if (best_result != profile_results.end()) {
    return *best_result;
  }

  return InternalError(
      "All algorithms tried for convolution %s failed.  Falling back to "
      "default algorithm.",
      instr->ToString());
}

StatusOr<bool> MiopenConvAlgorithmPicker::RunOnInstruction(
    HloInstruction* instr) {
  CHECK(IsCustomCallToDnnConvolution(*instr));

  StatusOr<MiopenConvAlgorithmPicker::AutotuneResult> best_algo_or =
      PickBestAlgorithm(Cast<HloCustomCallInstruction>(instr));
  if (!best_algo_or.ok()) {
    LOG(ERROR) << best_algo_or.status();
    return false;
  }

  auto best_algo = std::move(best_algo_or).ValueOrDie();
  VLOG(1) << "Setting cudnn conv to use algorithm " << best_algo.algorithm
          << " and " << NumBytesToString(best_algo.scratch_bytes)
          << " of scratch memory: " << instr->ToString()
          << " tensor_ops_enabled: " << best_algo.tensor_ops_enabled;

  // Replace instr with a new CustomCall which has the correct algorithm, and
  // whose output shape has the appropriate amount of scratch memory.
  HloComputation* computation = instr->parent();
  Shape new_call_shape = ShapeUtil::MakeTupleShape(
      {instr->shape().tuple_shapes(0),
       ShapeUtil::MakeShape(U8, {best_algo.scratch_bytes})});

  TF_ASSIGN_OR_RETURN(CudnnConvBackendConfig backend_config,
                      instr->backend_config<CudnnConvBackendConfig>());
  backend_config.set_algorithm(best_algo.algorithm);
  backend_config.set_tensor_ops_enabled(best_algo.tensor_ops_enabled);

  HloInstruction* new_call = computation->AddInstruction(
      instr->CloneWithNewOperands(new_call_shape, instr->operands()));

  VLOG(1) << "Replacing convolution " << instr->ToString() << " with "
          << new_call->ToString();

  TF_RETURN_IF_ERROR(new_call->set_backend_config(backend_config));

  // Repackage new_call so it has the same shape as the original call, namely
  // (conv_result, u8[0]).
  HloInstruction* new_tuple =
      computation->AddInstruction(HloInstruction::CreateTuple(
          {computation->AddInstruction(HloInstruction::CreateGetTupleElement(
               new_call_shape.tuple_shapes(0), new_call, 0)),
           computation->AddInstruction(HloInstruction::CreateConstant(
               LiteralUtil::CreateR1<uint8>({})))}));

  TF_RETURN_IF_ERROR(instr->parent()->ReplaceInstruction(instr, new_tuple));
  return true;
}

StatusOr<bool> MiopenConvAlgorithmPicker::RunOnComputation(
    HloComputation* computation) {
  std::vector<HloInstruction*> convs;
  for (auto* instr : computation->instructions()) {
    if (IsCustomCallToDnnConvolution(*instr)) {
      convs.push_back(instr);
    }
  }

  bool changed = false;
  for (auto* instr : convs) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnInstruction(instr));
    changed |= result;
  }
  return changed;
}

StatusOr<bool> MiopenConvAlgorithmPicker::Run(HloModule* module) {
  XLA_SCOPED_LOGGING_TIMER("CudnnConvAlgorithmPicker");

  if (module->config().debug_options().xla_gpu_disable_autotune()) {
    VLOG(2) << "Convolution auto-tuning disabled, CudnnConvAlgorithmPicker "
               "returning early.";
    return false;
  }

  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }

  {
    tensorflow::mutex_lock lock(autotune_cache_lock);
    autotune_cache_stats.LogStats();
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
