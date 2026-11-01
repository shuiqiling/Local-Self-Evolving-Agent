#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "repack.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
static std::mt19937 rng(20261031);
static void one(ggml_backend_t backend,FILE * out,int K,int R,int E,int topk,int slots,int batch,int threads,bool strided) {
    ggml_backend_cpu_set_n_threads(backend,threads);
    ggml_init_params params={4*1024*1024,nullptr,true};
    ggml_context * weights=ggml_init(params),*ctx=ggml_init(params);
    auto * a=ggml_new_tensor_3d(weights,GGML_TYPE_Q4_K,K,R,E);
    auto * ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,topk,batch);
    auto * storage=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,slots*batch*(strided?2:1));
    auto * b=ggml_view_3d(ctx,storage,K,slots,batch,size_t(K)*4*(strided?2:1),size_t(K)*4*slots*(strided?2:1),0);
    auto * result=ggml_mul_mat_id(ctx,a,b,ids);auto * graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,result);
    auto wb=ggml_backend_alloc_ctx_tensors_from_buft(weights,ggml_backend_cpu_repack_buffer_type());
    auto buf=ggml_backend_alloc_ctx_tensors(ctx,backend);
    if(!wb||!buf||!a->extra){fprintf(stderr,"repack allocation failed\n");exit(2);}
    std::vector<float> raw(size_t(K)*R*E),input(size_t(K)*slots*batch*(strided?2:1));
    for(auto &v:raw)v=float(int(rng()%20001)-10000)/8192.f;
    for(auto &v:input)v=float(int(rng()%20001)-10000)/4096.f;
    std::vector<uint8_t> q(ggml_nbytes(a));ggml_quantize_chunk(GGML_TYPE_Q4_K,raw.data(),q.data(),0,R*E,K,nullptr);
    ggml_backend_tensor_set(a,q.data(),0,q.size());ggml_backend_tensor_set(storage,input.data(),0,input.size()*4);
    std::vector<int32_t> route(topk*batch);
    for(int t=0;t<batch;++t)for(int e=0;e<topk;++e)route[t*topk+e]=(t*topk+e)%E;
    ggml_backend_tensor_set(ids,route.data(),0,route.size()*4);
    if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS){fprintf(stderr,"compute failed\n");exit(3);}
    std::vector<float> values(size_t(R)*topk*batch);ggml_backend_tensor_get(result,values.data(),0,values.size()*4);
    for(float v:values)if(!std::isfinite(v)){fprintf(stderr,"nonfinite\n");exit(4);}
    uint64_t n=values.size();fwrite(&n,sizeof(n),1,out);fwrite(values.data(),4,values.size(),out);
    ggml_backend_buffer_free(buf);ggml_backend_buffer_free(wb);ggml_free(ctx);ggml_free(weights);
}
int main(int argc,char **argv) {
    if(argc!=2)return 1;FILE * out=fopen(argv[1],"wb");if(!out)return 1;
    auto backend=ggml_backend_cpu_init();if(!backend)return 1;
    int cases=0;
    for(int K:{256,768,2560})for(int R:{8,24,128})for(int batch:{1,3,4,5,7,8,17,33})for(int threads:{1,6,16}) {
        one(backend,out,K,R,4,2,1,batch,threads,false);++cases;
    }
    for(int slots:{1,2,4})for(int threads:{1,6,16})for(bool strided:{false,true}) {
        one(backend,out,1024,264,8,4,slots,37,threads,strided);++cases;
    }
    ggml_backend_free(backend);fclose(out);printf("{\"cases\":%d,\"finite\":true,\"repack_selected\":true}\n",cases);
}
