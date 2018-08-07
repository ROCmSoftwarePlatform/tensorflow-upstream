/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/device_tracer.h"

#if TENSORFLOW_USE_ROCM

#include <stdlib.h>
#include <memory>
#include <iostream>

#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/tracing.h"
#include "hip/hip_runtime.h"
#include "hip/roctracer.h"
#include "hsa.h"

#define DT_LOG(s) do {} while (0)
#define DT_LOG1(s) do { std::cout << s << std::endl << std::flush; } while (0)
#define DT_LOG2(s) DT_LOG(s)
#define DT_LOG3(s) DT_LOG(s)

#define ROCTRACER_CALL(call)                                                                       \
  do {                                                                                             \
    int err = call;                                                                                \
    if (err != 0) {                                                                                \
      std::cerr << roctracer_error_string() << std::endl << std::flush;                            \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

namespace tensorflow {
namespace devicetracer {

// TODO(pbar) Move this to platform specific header file?
// Static thread local variable for POD types.
#define TF_STATIC_THREAD_LOCAL_POD(_Type_, _var_)                  \
  static thread_local _Type_ s_obj_##_var_;                            \
  namespace {                                                      \
  class ThreadLocal_##_var_ {                                      \
   public:                                                         \
    ThreadLocal_##_var_() {}                                       \
    void Init() {}                                                 \
    inline _Type_ *pointer() const { return &s_obj_##_var_; }      \
    inline _Type_ *safe_pointer() const { return &s_obj_##_var_; } \
    _Type_ &get() const { return s_obj_##_var_; }                  \
    bool is_native_tls() const { return true; }                    \
                                                                   \
   private:                                                        \
    TF_DISALLOW_COPY_AND_ASSIGN(ThreadLocal_##_var_);              \
  } _var_;                                                         \
  }  // namespace

// Thread-local state recording the most recent annotation (if any).
// When non-null, this points to a string in the active annotation
// of the current thread.  The annotation is guaranteed to remain live
// for the duration of the API callback.
TF_STATIC_THREAD_LOCAL_POD(const char *, tls_current_annotation);

class DeviceTracerBase;
class TraceMgr {
  public:
  TraceMgr() {}
  virtual ~TraceMgr() {}
  virtual bool Start(DeviceTracerBase*) = 0;
  virtual bool Stop() = 0;
};

class DeviceTracerIf : public DeviceTracer,
                       public port::Tracing::Engine {};

class DeviceTracerBase : public DeviceTracerIf {
 public:
  DeviceTracerBase(TraceMgr* mgr);
  ~DeviceTracerBase() override;

  // DeviceTracer interface:
  Status Start() override;
  Status Stop() override;
  Status Collect(StepStatsCollector *collector) override;
  void AddApiRecord(uint32_t cid, const hip_cb_data_t* data);
  void AddActivityRecord(const roctracer_async_record_t* record);

  // port::Tracing::Engine interface:
  bool IsEnabled() const override {
    // We only register the Engine while tracing is enabled.
    return true;
  }
  Annotation *PushAnnotation(StringPiece name) override {
    DT_LOG("PushAnnotation " << name);
    struct Impl : public port::Tracing::Engine::Annotation {
      string annotation;
      explicit Impl(StringPiece n) : annotation(n.ToString()) {
        // Remember the most recent ScopedAnnotation for each thread.
        tls_current_annotation.get() = annotation.c_str();
      }
      ~Impl() override { tls_current_annotation.get() = nullptr; }
    };
    return new Impl(name);
  }
  Tracer *StartTracing(StringPiece label, bool is_expensive) override {
    // We don't do anything with 'TraceMe' regions yet.
    return nullptr;
  }

  // Records the mapping between correlation ID and kernel name.
  static void AddCorrelationId(uint32 correlation_id, const string &name);
  static std::recursive_mutex trace_mu_;

 private:
  // Internal struct to record kernel launches.
  struct KernelRecord {
    int64_t start_timestamp;
    int64_t end_timestamp;
    int device_id;
    uint64 stream_id;
    uint64 correlation_id;
    uint8 activityDomain;
    uint8 activityKind;
  };
  // Internal struct to record memcpy operations.
  struct MemcpyRecord {
    int64_t start_timestamp;
    int64_t end_timestamp;
    int device_id;
    uint64 stream_id;
    uint64 correlation_id;
    uint8 activityDomain;
    uint8 activityKind;
    uint64 bytes;
  };

  typedef int64_t timestamp_t;
  typedef double freq_t;

  timestamp_t GetTimestamp() {
    timestamp_t timestamp_hz = 0;
    hsa_status_t status = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &timestamp_hz);
    if (status != HSA_STATUS_SUCCESS) {
      std::cerr << "hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY) failed" << std::endl;
      abort();
    }
    const freq_t timestamp_factor = (freq_t)1000000000 / (freq_t)timestamp_hz;
    timestamp_t timestamp = 0;
    status = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &timestamp);
    if (status != HSA_STATUS_SUCCESS) {
      std::cerr << "hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP) failed" << std::endl;
      abort();
    }
    const freq_t timestamp_ns = (freq_t)timestamp * timestamp_factor;
    return (timestamp_t)timestamp_ns;
  }

  // Returns the current system time in microseconds.
  inline int64 NowInUsec() { return Env::Default()->NowMicros(); }

  static constexpr size_t kMaxRecords = 1024 * 1024;
  static std::map<uint32, string>* correlations_;
  std::vector<KernelRecord> kernel_records_;
  std::vector<MemcpyRecord> memcpy_records_;

  mutex mu_;
  bool enabled_ GUARDED_BY(mu_);
  int64_t start_walltime_us_ GUARDED_BY(mu_);
  int64_t end_walltime_us_ GUARDED_BY(mu_);
  int64_t start_timestamp_ GUARDED_BY(mu_);
  int64_t end_timestamp_ GUARDED_BY(mu_);
  static uint64_t step_;

  TraceMgr* trace_mgr_;

  TF_DISALLOW_COPY_AND_ASSIGN(DeviceTracerBase);
};

std::recursive_mutex DeviceTracerBase::trace_mu_;
std::map<uint32, string>* DeviceTracerBase::correlations_ = NULL;
uint64_t DeviceTracerBase::step_ = 0;

DeviceTracerBase::DeviceTracerBase(TraceMgr* mgr) {
  DT_LOG3("DeviceTracer created. " << this);
  enabled_ = false;
  trace_mgr_ = mgr;
  start_timestamp_ = 0;
  end_timestamp_ = 0;
}

DeviceTracerBase::~DeviceTracerBase() {
  // freed memory.
  Stop().IgnoreError();
  DT_LOG3("DeviceTracer destroed. " << this);
}

Status DeviceTracerBase::Start() {
  mutex_lock l(mu_);
  if (enabled_) {
    return errors::FailedPrecondition("DeviceTracer is already enabled.");
  }

  if (correlations_ == NULL) correlations_ = new std::map<uint32, string>;

  start_walltime_us_ = NowInUsec();
  start_timestamp_ = GetTimestamp();
  ++step_;
  DT_LOG2("DeviceTracer::Start(" << step_ << ") wt " << start_walltime_us_ << ", ts " << start_timestamp_ << " " << this);

  while(trace_mgr_->Start(this) != true) {}
  // Register as a TraceEngine to receive ScopedAnnotations.
  port::Tracing::RegisterEngine(this);
  enabled_ = true;

  return Status::OK();
}

Status DeviceTracerBase::Stop() {
  mutex_lock l(mu_);
  if (!enabled_) {
    return Status::OK();
  }

  end_walltime_us_ = NowInUsec();
  end_timestamp_ = GetTimestamp();
  DT_LOG2("DeviceTracer::Stop(" << step_ << ") wt " << end_walltime_us_ << ", ts " << end_timestamp_ << " " << this);

  port::Tracing::RegisterEngine(nullptr);
  trace_mgr_->Stop();
  enabled_ = false;

  return Status::OK();
}

void DeviceTracerBase::AddCorrelationId(uint32 correlation_id,
                                        const string &name) {
  DT_LOG3("CORRID: " << correlation_id << " : " << name);
  std::lock_guard<std::recursive_mutex> l(trace_mu_);
  //if (correlations_.size() >= kMaxRecords) return;
  //correlations_.emplace(correlation_id, name);
  (*correlations_)[correlation_id] = name;
}

Status DeviceTracerBase::Collect(StepStatsCollector *collector) {
  mutex_lock l(mu_);
  if (enabled_) {
    return errors::FailedPrecondition("DeviceTracer is still enabled.");
  }
  DT_LOG2("DeviceTracer::Collect(" << step_ << ") wt " << start_walltime_us_ << " : " << end_walltime_us_ << ", ts " << start_timestamp_ << " : " << end_timestamp_ << " " << this);

  // TODO(pbar) Handle device IDs and prefix properly.
  const string prefix = "";

  std::lock_guard<std::recursive_mutex> l2(trace_mu_);
  for (const auto &rec : kernel_records_) {
    const string stream_device =
      strings::StrCat(prefix, "/device:GPU:", rec.device_id, "/stream:");
    const char* cmd_kind_string = roctracer_id_string(rec.activityDomain, rec.activityKind);
    auto it = correlations_->find(rec.correlation_id);
    const string name = (it != correlations_->cend()) ? it->second : (string("unknown:") + cmd_kind_string);
    NodeExecStats *ns = new NodeExecStats;
    ns->set_all_start_micros(start_walltime_us_ +
                             ((rec.start_timestamp - start_timestamp_) / 1000));
    ns->set_op_start_rel_micros(0);
    auto elapsed_us =
        std::max<int64>((rec.end_timestamp - rec.start_timestamp) / 1000, 1);
    ns->set_op_end_rel_micros(elapsed_us);
    ns->set_all_end_rel_micros(elapsed_us);
    ns->set_node_name(name);
    // TODO(pbar) Generate details based on the kernel activity record.
    // ns->set_timeline_label(details);
    auto nscopy = new NodeExecStats;
    *nscopy = *ns;
    collector->Save(strings::StrCat(stream_device, "all"), ns);
    collector->Save(strings::StrCat(stream_device, rec.stream_id), nscopy);
    DT_LOG2("KERNEL REC: " <<
      name <<
      ", " << strings::StrCat(stream_device, rec.stream_id) <<
      ", corrid " << rec.correlation_id <<
      ", start " << (start_walltime_us_ + ((rec.start_timestamp - start_timestamp_) / 1000)) << "us" <<
      ", elapsed " << elapsed_us << "us"
    );
    if (it == correlations_->cend()) {
      std::cerr << "Unknown async kernel record" << std::endl << std::flush;
      abort();
    }
  }
  for (const auto &rec : memcpy_records_) {
    const string stream_device =
      strings::StrCat(prefix, "/device:GPU:", rec.device_id, "/stream:");
    const string memcpy_device =
      strings::StrCat(prefix, "/device:GPU:", rec.device_id, "/memcpy");
    const char* cmd_kind_string = roctracer_id_string(rec.activityDomain, rec.activityKind);
    auto it = correlations_->find(rec.correlation_id);
    const string name = (it != correlations_->cend()) ? it->second : "unknown";
    NodeExecStats *ns = new NodeExecStats;
    ns->set_all_start_micros(start_walltime_us_ +
                             ((rec.start_timestamp - start_timestamp_) / 1000));
    ns->set_op_start_rel_micros(0);
    auto elapsed_us =
        std::max<int64>((rec.end_timestamp - rec.start_timestamp) / 1000, 1);
    ns->set_op_end_rel_micros(elapsed_us);
    ns->set_all_end_rel_micros(elapsed_us);
    const string details = strings::Printf(
        "%s %llu bytes", cmd_kind_string, rec.bytes);
    ns->set_node_name(
        strings::StrCat(name, ":", cmd_kind_string));
    ns->set_timeline_label(details);
    auto nscopy = new NodeExecStats;
    *nscopy = *ns;
    collector->Save(memcpy_device, ns);
    collector->Save(strings::StrCat(stream_device, rec.stream_id), nscopy);
    DT_LOG2("MEMCPY REC: " <<
      strings::StrCat(name, ":", cmd_kind_string) <<
      ", " << details <<
      ", " << memcpy_device <<
      ", " << strings::StrCat(stream_device, rec.stream_id) <<
      ", corrid " << rec.correlation_id <<
      ", start " << start_walltime_us_ + ((rec.start_timestamp - start_timestamp_) / 1000) << "us" <<
      ", elapsed " << elapsed_us << "us"
    );
    if (it == correlations_->cend()) {
      std::cerr << "Unknown async memcpy record" << std::endl << std::flush;
      abort();
    }
  }
  return Status::OK();
}

void DeviceTracerBase::AddApiRecord(uint32_t cid, const hip_cb_data_t* data) {
  const char* name = roctracer_id_string(ROCTRACER_DOMAIN_HIP_API, cid);
  (void)name;

  // API callbacks are invoked synchronously on the thread making the
  // API call. If this pointer is non-null then the ScopedAnnotation
  // must be valid.
  const char* tls_annotation = tls_current_annotation.get();

  DT_LOG("API Callback for " << name << " '" << tls_annotation << "'");

  switch (cid) {
    case HIP_API_ID_hipModuleLaunchKernel:
    case HIP_API_ID_hipHccModuleLaunchKernel:
      {
#if 0
        const string annotation = hipKernelNameRef(data->args.hipModuleLaunchKernel.f);
#else
        const string annotation = 
          tls_annotation ? string(tls_annotation) : string(hipKernelNameRef(data->args.hipModuleLaunchKernel.f)) + ":kernel";
#endif
        AddCorrelationId(data->correlation_id, annotation);
      }
      break;
    case HIP_API_ID_hipMemcpyToSymbolAsync:
    case HIP_API_ID_hipMemcpyFromSymbolAsync:
    case HIP_API_ID_hipMemcpy2DToArray:
    case HIP_API_ID_hipMemcpyAsync:
    case HIP_API_ID_hipMemcpyFromSymbol:
    case HIP_API_ID_hipMemcpy3D:
    case HIP_API_ID_hipMemcpyAtoH:
    case HIP_API_ID_hipMemcpyHtoD:
    case HIP_API_ID_hipMemcpyHtoA:
    case HIP_API_ID_hipMemcpy2D:
    case HIP_API_ID_hipMemcpyPeerAsync:
    case HIP_API_ID_hipMemcpyDtoH:
    case HIP_API_ID_hipMemcpyHtoDAsync:
    case HIP_API_ID_hipMemcpyDtoD:
    case HIP_API_ID_hipMemcpyFromArray:
    case HIP_API_ID_hipMemcpy2DAsync:
    case HIP_API_ID_hipMemcpy:
    case HIP_API_ID_hipMemcpyToArray:
    case HIP_API_ID_hipMemcpyToSymbol:
    case HIP_API_ID_hipMemcpyPeer:
    case HIP_API_ID_hipMemcpyDtoDAsync:
    case HIP_API_ID_hipMemcpyDtoHAsync:
    case HIP_API_ID_hipMemcpyParam2D:
      {
        const string annotation = tls_annotation ? tls_annotation : "unknown:unknown";
        AddCorrelationId(data->correlation_id, annotation);
      }
      break;
    default:
      DT_LOG("Unhandled API Callback for " << name);
  }
}

void DeviceTracerBase::AddActivityRecord(const roctracer_async_record_t* record) {
  std::lock_guard<std::recursive_mutex> l(trace_mu_);

  const char* name = roctracer_id_string(record->domain, record->activity_kind);
  (void)name;
//  DT_LOG("Activity " << name << ", op(" << record->op_id << ")");

  switch (record->op_id) {
    // Kernel activity
    case 1:
      if (kernel_records_.size() >= kMaxRecords) return;
      kernel_records_.push_back(KernelRecord{
        (int64)(record->begin_ns),
        (int64)(record->end_ns),
        record->device_id,
        record->stream_id,
        record->correlation_id,
        (uint8)(record->domain),
        (uint8)(record->activity_kind)
      });
      break;
    // Memcpy activity
    case 2:
      if (memcpy_records_.size() >= kMaxRecords) return;
      memcpy_records_.push_back(MemcpyRecord{
        (int64)(record->begin_ns),
        (int64)(record->end_ns),
        record->device_id,
        record->stream_id,
        record->correlation_id,
        (uint8)(record->domain),
        (uint8)(record->activity_kind),
        record->bytes
      });
      break;
    default:
      DT_LOG("Unhandled Activity Callback for " << name << ", op(" << record->op_id << ")");
  }
}

class DeviceTracerRocm : public TraceMgr {
  public:
  DeviceTracerRocm() {
    device_ = NULL;
    // Check tracer domains consitency
    ROCTRACER_CALL(roctracer_validate_domains());
    // Creating tracer pool
    roctracer_properties_t properties{};
    properties.buffer_size = 12;
    properties.buffer_callback_fun = activity_callback;
    properties.buffer_callback_arg = this;
    if (roctracer_default_pool() == NULL)
      ROCTRACER_CALL(roctracer_open_pool(&properties));
  }

  ~DeviceTracerRocm() override {
    DT_LOG("~DeviceTracerRocm()");
  }

  bool Start(DeviceTracerBase* device) override {
    mutex_lock l(mu_);
    if (device_ != NULL) {
      DT_LOG1("DeviceTracerRocm::Start BUSY");
      abort();
      return false;
    }
    DT_LOG("DeviceTracerRocm::Start");
    device_ = device;

    // Enable HIP API callbacks
    ROCTRACER_CALL(roctracer_enable_api_callback(ROCTRACER_DOMAIN_ANY, HIP_API_ID_ANY, hip_api_callback, this));
    // Enable HIP activity tracing
    ROCTRACER_CALL(roctracer_enable_api_activity(ROCTRACER_DOMAIN_ANY, HIP_API_ID_ANY));
    DT_LOG("DeviceTracerRocm::Start DONE");
    return true;
  }

  bool Stop() override {
    mutex_lock l(mu_);
    if (device_ == NULL) {
       DT_LOG("DeviceTracerRocm::Stop already disabled");
       abort();
    }
    DT_LOG("DeviceTracerRocm::Stop");
    device_ = NULL;
#if 0
    // Disable HIP API callbacks
    ROCTRACER_CALL(roctracer_disable_api_callback(ROCTRACER_DOMAIN_ANY, HIP_API_ID_ANY));
    // Disable HIP activity tracing
    ROCTRACER_CALL(roctracer_disable_api_activity(ROCTRACER_DOMAIN_ANY, HIP_API_ID_ANY));
#endif
    // Flush buffered atcivity
    ROCTRACER_CALL(roctracer_flush_api_activity());
    DT_LOG("DeviceTracerRocm::Stop DONE");
    return true;
  }

  private:
  // HIP API callback function
  static void hip_api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg)
  {
    std::lock_guard<std::recursive_mutex> l(DeviceTracerBase::trace_mu_);
    const hip_cb_data_t* data = reinterpret_cast<const hip_cb_data_t*>(callback_data);
    const char* name = roctracer_id_string(ROCTRACER_DOMAIN_HIP_API, cid);
    DT_LOG2("API: " << name << "(" << data->correlation_id << ") phase(" << data->phase << ")");
    if (data->phase == ROCTRACER_API_PHASE_ENTER) {
      DeviceTracerRocm* tracer = reinterpret_cast<DeviceTracerRocm*>(arg);
      DeviceTracerBase* device = tracer->device_;
//      DT_LOG("API: " << name << "(" << data->correlation_id << ")");
      DeviceTracerBase::AddCorrelationId(data->correlation_id, string(name));
      if (device != NULL) device->AddApiRecord(cid, data);
    }
  }

  // Activity tracing callback
  static void activity_callback(const char* begin, const char* end, void* arg) {
    DeviceTracerRocm* tracer = reinterpret_cast<DeviceTracerRocm*>(arg);
    DeviceTracerBase* device = tracer->device_;
    if (device != NULL) {
      const roctracer_record_t* record = reinterpret_cast<const roctracer_record_t*>(begin);
      const roctracer_record_t* end_record = reinterpret_cast<const roctracer_record_t*>(end);
      while (record < end_record) {
//        const char* name = roctracer_id_string(record->domain, record->activity_kind);
//        DT_LOG("Activity Callback " << name << " " << record->correlation_id << ", op(" << record->op_id << "), tracer " << tracer << ", device " << device);
        if (record->op_id != 0) {
          const roctracer_async_record_t* async_record = reinterpret_cast<const roctracer_async_record_t*>(record);
          device->AddActivityRecord(async_record);
        }
        ROCTRACER_CALL(roctracer_next_record(record, &record));
      }
    }
  }

  mutex mu_;
  DeviceTracerBase* device_;
};

}  // namespace devicetracer

std::unique_ptr<DeviceTracer> CreateDeviceTracer() {
  static auto* tracer_mgr = new devicetracer::DeviceTracerRocm();
  std::unique_ptr<DeviceTracer> tracer(new devicetracer::DeviceTracerBase(tracer_mgr));
  return tracer;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_ROCM
