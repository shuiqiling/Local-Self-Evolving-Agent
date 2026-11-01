#define main sq_previous_test_main
#include "test_actual.cpp"
#undef main
#include "pack-q8k.inc"
static double timed(Mat gemv,Mat gemm,int n,int nc,const void * w,const block_q8_K * const rows[4],bool batched) {
    std::vector<block_q8_Kx4> packed(n/256);std::vector<float> out(4*nc);
    auto start=std::chrono::steady_clock::now();
    for(int rep=0;rep<12;++rep) {
        if(batched){sq_pack_q8k_4x8(rows,packed.data(),n/256);gemm(n,out.data(),nc,w,packed.data(),4,nc);}
        else for(int r=0;r<4;++r)gemv(n,out.data()+r*nc,nc,w,rows[r],1,nc);
    }
    return std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/12;
}
int main(int argc,char**argv) {
    if(argc!=2)return 1;auto lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!lib){fprintf(stderr,"%s\n",dlerror());return 1;}
    Mat gv=(Mat)dlsym(lib,"ggml_gemv_q4_K_8x8_q8_K"),gm=(Mat)dlsym(lib,"ggml_gemm_q4_K_8x8_q8_K");
    if(!gv||!gm)return 1;int count=0;size_t values=0,differing_values=0;float max_error=0.f,max_scaled_error=0.f;
    for(int n:{256,512,1024,2560,4096,9728})for(int nc:{8,16,24,64})for(int pattern=0;pattern<5;++pattern) {
        std::vector<block_q4_Kx8>w(size_t(n/256)*(nc/8));fill_weights(w,pattern);
        std::vector<block_q8_K>acts[4];const block_q8_K * rows[4];
        for(int r=0;r<4;++r){acts[r].resize(n/256);fill_q8(acts[r],pattern);rows[r]=acts[r].data();}
        std::vector<block_q8_Kx4>packed(n/256);sq_pack_q8k_4x8(rows,packed.data(),n/256);
        std::vector<float>x(16+4*nc+16,guard),y=x;
        for(int r=0;r<4;++r)gv(n,x.data()+16+r*nc,nc,w.data(),rows[r],1,nc);
        gm(n,y.data()+16,nc,w.data(),packed.data(),4,nc);
        float matrix_scale=0.f,matrix_error=0.f;
        for(size_t z=16;z<16+size_t(4*nc);++z)matrix_scale=std::max(matrix_scale,std::fabs(x[z]));
        for(size_t z=0;z<x.size();++z) {
            float error=std::fabs(x[z]-y[z]);matrix_error=std::max(matrix_error,error);
            if(memcmp(&x[z],&y[z],4))++differing_values;
            const bool active=z>=16&&z<16+size_t(4*nc);
            if(!std::isfinite(y[z])||(!active&&memcmp(&x[z],&y[z],4))||error>1e-4f+2e-6f*matrix_scale){
                fprintf(stderr,"GEMV/GEMM mismatch n=%d nc=%d pattern=%d z=%zu old=%.9g new=%.9g\n",n,nc,pattern,z,x[z],y[z]);return 2;
            }
        }
        max_error=std::max(max_error,matrix_error);max_scaled_error=std::max(max_scaled_error,matrix_error/std::max(1.f,matrix_scale));
        ++count;values+=4*nc;
    }
    printf("{\"kind\":\"pack_correctness\",\"cases\":%d,\"values\":%zu,\"bitwise_equal\":%s,\"differing_values\":%zu,\"max_error\":%.9g,\"max_scaled_error\":%.9g,\"canaries\":true,\"tolerance\":\"1e-4+2e-6*matrix_max_abs\"}\n",count,values,differing_values?"false":"true",differing_values,max_error,max_scaled_error);
    for(auto shape:std::vector<std::pair<int,int>>{{2560,768},{768,2560},{2560,2560}}) {
        int n=shape.first,nc=shape.second;std::vector<block_q4_Kx8>w(size_t(n/256)*(nc/8));fill_weights(w,0);
        std::vector<block_q8_K>acts[4];const block_q8_K *rows[4];
        for(int r=0;r<4;++r){acts[r].resize(n/256);fill_q8(acts[r],0);rows[r]=acts[r].data();}
        std::vector<double>x,y;
        for(int r=0;r<6;++r){if(r%2){y.push_back(timed(gv,gm,n,nc,w.data(),rows,true));x.push_back(timed(gv,gm,n,nc,w.data(),rows,false));}
            else{x.push_back(timed(gv,gm,n,nc,w.data(),rows,false));y.push_back(timed(gv,gm,n,nc,w.data(),rows,true));}}
        std::sort(x.begin(),x.end());std::sort(y.begin(),y.end());
        printf("{\"kind\":\"moe_micro\",\"n\":%d,\"nc\":%d,\"old_ns\":%.1f,\"new_ns\":%.1f,\"speedup\":%.4f}\n",n,nc,x[3],y[3],x[3]/y[3]);
    }
}
