#include "generic_backend.hpp"

#include "mercan_graph.h"
#include "mercan_tensor.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct generic_tensor_catalog {
    std::unordered_map<std::string, ggml_tensor *> tensors;
    std::string last_error;
};

struct generic_graph_state {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * embedding = nullptr;
    ggml_tensor * logits = nullptr;
    ggml_tensor * hidden = nullptr;
};

static ggml_tensor * from_handle(mercan_tensor_handle_v1 h) {
    return reinterpret_cast<ggml_tensor *>(static_cast<uintptr_t>(h));
}
static mercan_tensor_handle_v1 to_handle(ggml_tensor * t) {
    return static_cast<mercan_tensor_handle_v1>(reinterpret_cast<uintptr_t>(t));
}

static generic_tensor_catalog * catalog(mercan_tensor_resolver_v1 * r) {
    return r ? static_cast<generic_tensor_catalog *>(r->userdata) : nullptr;
}
static int tensor_declare(mercan_tensor_resolver_v1 * r, const char * name, uint32_t flags) {
    auto * c = catalog(r);
    if (!c || !name || !*name) return -1;
    if ((flags & MERCAN_TENSOR_REQUIRED_V1) && c->tensors.find(name) == c->tensors.end()) {
        c->last_error = std::string("required tensor is missing: ") + name;
        return -1;
    }
    return 0;
}
static mercan_tensor_handle_v1 tensor_by_name(mercan_tensor_resolver_v1 * r, const char * name) {
    auto * c = catalog(r);
    if (!c || !name) return MERCAN_TENSOR_NONE_V1;
    auto it = c->tensors.find(name);
    return it == c->tensors.end() ? MERCAN_TENSOR_NONE_V1 : to_handle(it->second);
}
static mercan_tensor_handle_v1 tensor_require(mercan_tensor_resolver_v1 * r, const char * name) {
    auto h = tensor_by_name(r, name);
    if (h == MERCAN_TENSOR_NONE_V1) {
        if (auto * c = catalog(r)) c->last_error = std::string("required tensor lookup failed: ") + (name ? name : "<null>");
    }
    return h;
}
static int tensor_has(mercan_tensor_resolver_v1 * r, const char * name) { return tensor_by_name(r, name) != MERCAN_TENSOR_NONE_V1; }
static const char * tensor_error(mercan_tensor_resolver_v1 * r) { auto * c=catalog(r); return c ? c->last_error.c_str() : "invalid resolver"; }
static size_t tensor_decl_count(mercan_tensor_resolver_v1 *) { return 0; }
static const char * tensor_decl_name(mercan_tensor_resolver_v1 *, size_t) { return nullptr; }
static uint32_t tensor_decl_flags(mercan_tensor_resolver_v1 *, size_t) { return 0; }
static const mercan_tensor_api_v1 TENSOR_API = {
    MERCAN_TENSOR_ABI_VERSION, sizeof(mercan_tensor_api_v1), tensor_declare, tensor_by_name, tensor_require,
    tensor_has, tensor_error, tensor_decl_count, tensor_decl_name, tensor_decl_flags
};

static generic_graph_state * gs(mercan_graph_builder_v1 * b) { return b ? static_cast<generic_graph_state *>(b->userdata) : nullptr; }
static int64_t g_dim(mercan_graph_builder_v1 *, mercan_tensor_handle_v1 h, uint32_t a) { auto *t=from_handle(h); return t && a<GGML_MAX_DIMS ? t->ne[a] : 0; }
static size_t g_stride(mercan_graph_builder_v1 *, mercan_tensor_handle_v1 h, uint32_t a) { auto *t=from_handle(h); return t && a<GGML_MAX_DIMS ? t->nb[a] : 0; }
static mercan_tensor_handle_v1 g_get_rows(mercan_graph_builder_v1 *b, mercan_tensor_handle_v1 a, mercan_tensor_handle_v1 i) { auto*s=gs(b); auto*x=from_handle(a); auto*y=from_handle(i); return s&&s->ctx&&x&&y?to_handle(ggml_get_rows(s->ctx,x,y)):0; }
static mercan_tensor_handle_v1 g_cast(mercan_graph_builder_v1*b, mercan_tensor_handle_v1 a) { auto*s=gs(b); auto*x=from_handle(a); return s&&x?to_handle(ggml_cast(s->ctx,x,GGML_TYPE_F32)):0; }
static mercan_tensor_handle_v1 g_swiglu(mercan_graph_builder_v1*b, mercan_tensor_handle_v1 a, mercan_tensor_handle_v1 c) { auto*s=gs(b); auto*x=from_handle(a); auto*y=from_handle(c); return s&&x&&y?to_handle(ggml_swiglu_split(s->ctx,x,y)):0; }
static mercan_tensor_handle_v1 g_view2(mercan_graph_builder_v1*b, mercan_tensor_handle_v1 a,int64_t n0,int64_t n1,size_t nb1,size_t off) { auto*s=gs(b); auto*x=from_handle(a); return s&&x?to_handle(ggml_view_2d(s->ctx,x,n0,n1,nb1,off)):0; }
static mercan_tensor_handle_v1 g_mul(mercan_graph_builder_v1*b, mercan_tensor_handle_v1 a, mercan_tensor_handle_v1 c) { auto*s=gs(b); auto*x=from_handle(a); auto*y=from_handle(c); return s&&x&&y?to_handle(ggml_mul(s->ctx,x,y)):0; }
static mercan_tensor_handle_v1 g_add(mercan_graph_builder_v1*b, mercan_tensor_handle_v1 a, mercan_tensor_handle_v1 c) { auto*s=gs(b); auto*x=from_handle(a); auto*y=from_handle(c); return s&&x&&y?to_handle(ggml_add(s->ctx,x,y)):0; }
static mercan_tensor_handle_v1 g_concat(mercan_graph_builder_v1*b, mercan_tensor_handle_v1 a, mercan_tensor_handle_v1 c,int32_t d) { auto*s=gs(b); auto*x=from_handle(a); auto*y=from_handle(c); return s&&x&&y?to_handle(ggml_concat(s->ctx,x,y,d)):0; }
static mercan_tensor_handle_v1 g_matmul(mercan_graph_builder_v1*b, mercan_tensor_handle_v1 a, mercan_tensor_handle_v1 c) { auto*s=gs(b); auto*x=from_handle(a); auto*y=from_handle(c); return s&&x&&y?to_handle(ggml_mul_mat(s->ctx,x,y)):0; }
static mercan_tensor_handle_v1 g_rms(mercan_graph_builder_v1*b, mercan_tensor_handle_v1 a,float eps) { auto*s=gs(b); auto*x=from_handle(a); return s&&x?to_handle(ggml_rms_norm(s->ctx,x,eps)):0; }
static mercan_tensor_handle_v1 g_rope(mercan_graph_builder_v1*b,mercan_tensor_handle_v1 a,mercan_tensor_handle_v1 pos,mercan_tensor_handle_v1 freq,int32_t nd,mercan_rope_mode_v1 mode,int32_t ctx,float base,float scale,float ext,float attn,float bf,float bs) { auto*s=gs(b); auto*x=from_handle(a); auto*p=from_handle(pos); auto*f=from_handle(freq); int m=mode==MERCAN_ROPE_MODE_NORMAL_V1?GGML_ROPE_TYPE_NORMAL:mode==MERCAN_ROPE_MODE_NEOX_V1?GGML_ROPE_TYPE_NEOX:-1; return s&&x&&p&&m>=0?to_handle(ggml_rope_ext(s->ctx,x,p,f,nd,m,ctx,base,scale,ext,attn,bf,bs)):0; }
static mercan_tensor_handle_v1 g_attention(mercan_graph_builder_v1*,mercan_tensor_handle_v1,mercan_tensor_handle_v1,mercan_tensor_handle_v1,mercan_tensor_handle_v1,mercan_tensor_handle_v1,mercan_tensor_handle_v1,float,int32_t) { return MERCAN_TENSOR_NONE_V1; }
static int g_output(mercan_graph_builder_v1*b, mercan_graph_output_kind_v1 k, mercan_tensor_handle_v1 h) { auto*s=gs(b); auto*t=from_handle(h); if(!s||!t)return -1; if(k==MERCAN_GRAPH_OUTPUT_EMBEDDING_V1)s->embedding=t; else if(k==MERCAN_GRAPH_OUTPUT_LOGITS_V1)s->logits=t; else if(k==MERCAN_GRAPH_OUTPUT_HIDDEN_V1)s->hidden=t; else return -1; ggml_set_output(t); return 0; }
static int g_finalize(mercan_graph_builder_v1*b, mercan_tensor_handle_v1 h) { auto*s=gs(b); auto*t=from_handle(h); if(!s||!s->graph||!t)return -1; ggml_build_forward_expand(s->graph,t); return 0; }
static const mercan_graph_api_v1 GRAPH_API = {
    MERCAN_GRAPH_ABI_VERSION, sizeof(mercan_graph_api_v1), g_dim, g_stride, g_get_rows, g_cast, g_swiglu,
    g_view2, g_mul, g_add, g_concat, g_matmul, g_rms, g_rope, g_attention, g_output, g_finalize
};

static int key_i64(const gguf_context * g, const char * key, int fallback) {
    int64_t id=gguf_find_key(g,key); if(id<0)return fallback;
    switch(gguf_get_kv_type(g,id)) {
        case GGUF_TYPE_UINT32:return (int)gguf_get_val_u32(g,id); case GGUF_TYPE_INT32:return gguf_get_val_i32(g,id);
        case GGUF_TYPE_UINT64:return (int)gguf_get_val_u64(g,id); case GGUF_TYPE_INT64:return (int)gguf_get_val_i64(g,id); default:return fallback;
    }
}

static int metadata_has(const mercan_metadata_v1 *m,const char*k){auto*g=m?static_cast<const gguf_context*>(m->userdata):nullptr;return g&&k&&gguf_find_key(g,k)>=0;}
static const char * metadata_str(const mercan_metadata_v1*m,const char*k){auto*g=m?static_cast<const gguf_context*>(m->userdata):nullptr;if(!g||!k)return nullptr;auto id=gguf_find_key(g,k);return id>=0&&gguf_get_kv_type(g,id)==GGUF_TYPE_STRING?gguf_get_val_str(g,id):nullptr;}
static int64_t metadata_i64(const mercan_metadata_v1*m,const char*k,int64_t f){auto*g=m?static_cast<const gguf_context*>(m->userdata):nullptr;return g?key_i64(g,k,(int)f):f;}
static double metadata_f64(const mercan_metadata_v1*m,const char*k,double f){auto*g=m?static_cast<const gguf_context*>(m->userdata):nullptr;if(!g||!k)return f;auto id=gguf_find_key(g,k);if(id<0)return f;auto t=gguf_get_kv_type(g,id);if(t==GGUF_TYPE_FLOAT32)return gguf_get_val_f32(g,id);if(t==GGUF_TYPE_FLOAT64)return gguf_get_val_f64(g,id);return (double)key_i64(g,k,(int)f);}

} // namespace

struct mercan_generic_model {
    const mercan_architecture_v1 * architecture = nullptr;
    gguf_context * gguf = nullptr;
    ggml_context * weights_ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_t cpu_fallback = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    generic_tensor_catalog catalog;
    mercan_tensor_resolver_v1 resolver{};
    mercan_metadata_v1 metadata{};
    int32_t vocab_size = 0;
    mercan_token bos = -1, eos = -1, pad = -1;
};

struct mercan_generic_context {
    mercan_generic_model * model = nullptr;
    mercan_context_params params{};
    ggml_backend_sched_t sched = nullptr;
    ggml_context * graph_ctx = nullptr;
    std::vector<float> logits;
    uint32_t n_past = 0;
};

mercan_generic_model * mercan_generic_model_load(const char * path, const mercan_architecture_v1 * architecture,
                                                  mercan_model_params params, std::string & error) {
    auto m=std::make_unique<mercan_generic_model>(); m->architecture=architecture;
    gguf_init_params ip{}; ip.no_alloc=true; ip.ctx=&m->weights_ctx;
    m->gguf=gguf_init_from_file(path,ip);
    if(!m->gguf||!m->weights_ctx){error="generic backend failed to read GGUF tensors"; return nullptr;}

    ggml_backend_dev_t dev=nullptr;
    if(params.n_gpu_layers!=0) dev=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if(!dev) dev=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if(!dev){error="generic backend found no ggml backend device"; return nullptr;}
    m->backend=ggml_backend_dev_init(dev,nullptr);
    if(!m->backend){error="generic backend device initialization failed"; return nullptr;}
    if(ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        ggml_backend_dev_t cpu_dev=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if(!cpu_dev){error="generic GPU backend requires a CPU fallback device"; return nullptr;}
        m->cpu_fallback=ggml_backend_dev_init(cpu_dev,nullptr);
        if(!m->cpu_fallback){error="generic CPU fallback initialization failed"; return nullptr;}
    }
    m->weights_buffer=ggml_backend_alloc_ctx_tensors(m->weights_ctx,m->backend);
    if(!m->weights_buffer){error="generic backend failed to allocate model weights"; return nullptr;}
    ggml_backend_buffer_set_usage(m->weights_buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    FILE *f=ggml_fopen(path,"rb"); if(!f){error="generic backend could not reopen model data"; return nullptr;}
    const size_t data_off=gguf_get_data_offset(m->gguf);
    std::vector<unsigned char> tmp;
    const int64_t nt=gguf_get_n_tensors(m->gguf);
    for(int64_t i=0;i<nt;++i){
        const char *name=gguf_get_tensor_name(m->gguf,i); ggml_tensor *t=ggml_get_tensor(m->weights_ctx,name);
        if(!t){std::fclose(f);error=std::string("generic tensor metadata missing: ")+name;return nullptr;}
        const size_t sz=gguf_get_tensor_size(m->gguf,i); tmp.resize(sz);
        if(std::fseek(f,(long)(data_off+gguf_get_tensor_offset(m->gguf,i)),SEEK_SET)!=0 || std::fread(tmp.data(),1,sz,f)!=sz){std::fclose(f);error=std::string("generic tensor read failed: ")+name;return nullptr;}
        ggml_backend_tensor_set(t,tmp.data(),0,sz); m->catalog.tensors.emplace(name,t);
    }
    std::fclose(f); ggml_backend_synchronize(m->backend);

    m->resolver={MERCAN_TENSOR_ABI_VERSION,sizeof(mercan_tensor_resolver_v1),&m->catalog,&TENSOR_API};
    m->metadata={MERCAN_ARCH_ABI_VERSION,sizeof(mercan_metadata_v1),m->gguf,metadata_has,metadata_str,metadata_i64,metadata_f64};
    ggml_tensor *te=ggml_get_tensor(m->weights_ctx,"token_embd.weight");
    if(!te||te->ne[1]<=0){error="generic backend requires token_embd.weight";return nullptr;}
    m->vocab_size=(int32_t)te->ne[1];
    m->bos=key_i64(m->gguf,"mercan.tokenizer.bos_token_id",-1);
    m->eos=key_i64(m->gguf,"mercan.tokenizer.eos_token_id",-1);
    m->pad=key_i64(m->gguf,"mercan.tokenizer.pad_token_id",-1);
    return m.release();
}

void mercan_generic_model_free(mercan_generic_model *m){if(!m)return;if(m->weights_buffer)ggml_backend_buffer_free(m->weights_buffer);if(m->cpu_fallback)ggml_backend_free(m->cpu_fallback);if(m->backend)ggml_backend_free(m->backend);if(m->gguf)gguf_free(m->gguf);if(m->weights_ctx)ggml_free(m->weights_ctx);delete m;}

mercan_generic_context * mercan_generic_context_create(mercan_generic_model*m,mercan_context_params p,std::string&error){
    if(!m||!m->backend){error="invalid generic model";return nullptr;}
    auto*c=new mercan_generic_context();c->model=m;c->params=p;
    ggml_backend_t bs[2]={m->backend,m->cpu_fallback};
    const int nb=m->cpu_fallback?2:1;
    c->sched=ggml_backend_sched_new(bs,nullptr,nb,GGML_DEFAULT_GRAPH_SIZE,false,true);
    if(!c->sched){delete c;error="generic scheduler creation failed";return nullptr;}return c;
}
void mercan_generic_context_free(mercan_generic_context*c){if(!c)return;if(c->sched){ggml_backend_sched_reset(c->sched);ggml_backend_sched_free(c->sched);}if(c->graph_ctx)ggml_free(c->graph_ctx);delete c;}

int32_t mercan_generic_decode(mercan_generic_context*c,const mercan_token*tokens,int32_t n,std::string&error){
    if(!c||!tokens||n<=0){error="invalid generic decode arguments";return -1;}
    if(c->graph_ctx){ggml_backend_sched_reset(c->sched);ggml_free(c->graph_ctx);c->graph_ctx=nullptr;}
    ggml_init_params gp{};gp.mem_size=32u*1024u*1024u;gp.mem_buffer=nullptr;gp.no_alloc=true;c->graph_ctx=ggml_init(gp);if(!c->graph_ctx){error="generic graph metadata allocation failed";return -1;}
    generic_graph_state st{};st.ctx=c->graph_ctx;st.graph=ggml_new_graph_custom(c->graph_ctx,GGML_DEFAULT_GRAPH_SIZE,false);if(!st.graph){error="generic graph creation failed";return -1;}

    // Graph leaves must already own backend buffers. Scheduler allocation only covers tensors
    // reachable as allocatable graph compute nodes; unused optional inputs may be skipped.
    ggml_init_params ip{}; ip.mem_size=1024u*1024u; ip.mem_buffer=nullptr; ip.no_alloc=true;
    ggml_context * input_ctx=ggml_init(ip);
    if(!input_ctx){error="generic input metadata allocation failed";return -1;}
    ggml_tensor *inp=ggml_new_tensor_1d(input_ctx,GGML_TYPE_I32,n);ggml_set_input(inp);
    ggml_tensor *pos=ggml_new_tensor_1d(input_ctx,GGML_TYPE_I32,n);ggml_set_input(pos);
    ggml_backend_buffer_t input_buffer=ggml_backend_alloc_ctx_tensors(input_ctx,c->model->backend);
    if(!input_buffer){ggml_free(input_ctx);error="generic input backend allocation failed";return -1;}
    auto cleanup_inputs=[&](){ggml_backend_buffer_free(input_buffer);ggml_free(input_ctx);};
    ggml_backend_tensor_set(inp,tokens,0,sizeof(mercan_token)*(size_t)n);
    std::vector<int32_t> positions((size_t)n);for(int32_t i=0;i<n;++i)positions[(size_t)i]=(int32_t)c->n_past+i;
    ggml_backend_tensor_set(pos,positions.data(),0,sizeof(int32_t)*(size_t)n);

    mercan_graph_builder_v1 b{MERCAN_GRAPH_ABI_VERSION,sizeof(mercan_graph_builder_v1),&st,&GRAPH_API,&c->model->resolver,nullptr};
    mercan_arch_graph_invocation_v1 inv{};inv.abi_version=MERCAN_ARCH_GRAPH_INVOCATION_ABI_VERSION;inv.struct_size=sizeof(inv);inv.metadata=&c->model->metadata;inv.builder=&b;inv.input_tokens=to_handle(inp);inv.input_positions=to_handle(pos);inv.output_ids=MERCAN_TENSOR_NONE_V1;inv.n_tokens=n;inv.n_outputs=n;
    char e[512]={};if(mercan_arch_build_graph_v1(c->model->architecture,&inv,e,sizeof(e))!=0){error=std::string("external graph build failed: ")+(e[0]?e:"unknown error");cleanup_inputs();return -1;}
    if(!st.logits||!st.graph){error="external architecture did not publish logits/finalize graph";cleanup_inputs();return -1;}
    if(st.logits->type!=GGML_TYPE_F32){error="generic backend currently requires F32 logits";cleanup_inputs();return -1;}
    if(!ggml_backend_sched_alloc_graph(c->sched,st.graph)){error="generic scheduler graph allocation failed";cleanup_inputs();return -1;}
    enum ggml_status rc=ggml_backend_sched_graph_compute(c->sched,st.graph);if(rc!=GGML_STATUS_SUCCESS){error=std::string("generic graph compute failed: ")+ggml_status_to_string(rc);cleanup_inputs();return -1;}
    ggml_backend_sched_synchronize(c->sched);
    const int64_t vocab=st.logits->ne[0];if(vocab<=0){error="generic logits have invalid shape";return -1;}
    c->logits.resize((size_t)vocab);const size_t off=(size_t)(std::max<int64_t>(1,st.logits->ne[1])-1)*st.logits->nb[1];ggml_backend_tensor_get(st.logits,c->logits.data(),off,sizeof(float)*(size_t)vocab);c->n_past+=(uint32_t)n;cleanup_inputs();return 0;
}
const float * mercan_generic_logits(mercan_generic_context*c){return c&& !c->logits.empty()?c->logits.data():nullptr;}
int32_t mercan_generic_vocab_size(const mercan_generic_model*m){return m?m->vocab_size:0;}
uint32_t mercan_generic_context_size(const mercan_generic_context*c){return c?c->params.n_ctx:0;}
mercan_token mercan_generic_bos_token(const mercan_generic_model*m){return m?m->bos:-1;}
mercan_token mercan_generic_eos_token(const mercan_generic_model*m){return m?m->eos:-1;}
mercan_token mercan_generic_pad_token(const mercan_generic_model*m){return m?m->pad:-1;}
