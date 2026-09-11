#include "mercan.h"
#include "mercan_plugin.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

static bool make_model(const char * path) {
    ggml_init_params p{}; p.mem_size=2*1024*1024; p.no_alloc=false;
    ggml_context * c=ggml_init(p); if (!c) return false;
    ggml_tensor * emb=ggml_new_tensor_2d(c,GGML_TYPE_F32,4,256); ggml_set_name(emb,"token_embd.weight");
    ggml_tensor * out=ggml_new_tensor_2d(c,GGML_TYPE_F32,4,256); ggml_set_name(out,"output.weight");
    float * e=(float*)emb->data; float * o=(float*)out->data;
    std::fill(e,e+4*256,0.0f); std::fill(o,o+4*256,0.0f);
    for(int t=0;t<256;++t) e[t*4]=1.0f;
    o[33*4]=10.0f; // every token predicts '!'
    gguf_context * g=gguf_init_empty();
    gguf_set_val_str(g,"mercan.format","mercan"); gguf_set_val_i64(g,"mercan.format_version",1); gguf_set_val_i64(g,"mercan.runtime_abi",1);
    gguf_set_val_str(g,"general.architecture","anka"); gguf_set_val_str(g,"mercan.tokenizer.type","anka-byte");
    gguf_set_val_i64(g,"mercan.tokenizer.eos_token_id",255);
    gguf_add_tensor(g,emb); gguf_add_tensor(g,out); const bool ok=gguf_write_to_file(g,path,false);
    gguf_free(g); ggml_free(c); return ok;
}

int main(int argc,char **argv){
    auto fail=[](const char *m){std::cerr<<"FAIL: "<<m<<"\n";return 10;};
    if(argc!=3 && argc!=4) return fail("usage");
    if(!make_model(argv[2])) return fail("model write");
    const int prc=mercan_plugin_load_v1(argv[1]); if(prc!=0){std::cerr<<mercan_plugin_last_error_v1()<<"\n";return fail("plugin load");}
    mercan_backend_init();
    auto mp=mercan_model_default_params(); mp.n_gpu_layers=(argc==4 && std::strcmp(argv[3], "gpu")==0) ? -1 : 0;
    mercan_model *m=mercan_model_load(argv[2],mp); if(!m){std::cerr<<mercan_last_error()<<"\n";return fail("model load");}
    if(std::strcmp(mercan_model_architecture(m),"anka")!=0) return fail("architecture");
    if(std::strcmp(mercan_model_tokenizer(m),"anka-byte")!=0) return fail("tokenizer");
    if(mercan_vocab_size(m)!=256) return fail("vocab");
    mercan_token toks[8]={}; int32_t n=mercan_tokenize(m,"Hi",2,false,false,toks,8);
    if(n!=2||toks[0]!=72||toks[1]!=105) return fail("encode");
    auto cp=mercan_context_default_params(); cp.n_ctx=64; cp.n_threads=2; cp.n_threads_batch=2;
    mercan_context *ctx=mercan_context_create(m,cp); if(!ctx){std::cerr<<mercan_last_error()<<"\n";return fail("context");}
    if(mercan_decode(ctx,toks,n)!=0){std::cerr<<mercan_last_error()<<"\n";return fail("decode");}
    const float *logits=mercan_logits(ctx); if(!logits) return fail("logits");
    int best=(int)(std::max_element(logits,logits+256)-logits); if(best!=33) return fail("prediction");
    char ch=0; if(mercan_token_to_piece(m,33,&ch,1,false)!=1||ch!='!') return fail("decode piece");
    mercan_context_free(ctx); mercan_model_free(m); mercan_backend_free(); std::remove(argv[2]);
    std::cout<<"GENERIC_PLUGIN_BACKEND_OK predicted="<<best<<" backend="<<(mp.n_gpu_layers!=0?"gpu":"cpu")<<"\n"; return 0;
}
