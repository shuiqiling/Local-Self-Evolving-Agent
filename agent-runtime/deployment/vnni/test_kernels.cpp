// Compare actual shared-library kernels, including complete repacked blocks.
// No model weights are modified. Reference is the production binary snapshot.
#include "repack.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <vector>
using Mat=void(*)(int,float*,size_t,const void*,const void*,int,int);
using Quant=void(*)(const float*,void*,int64_t);
static std::mt19937 random_bits(20261029);
static size_t cases=0,values=0;
static constexpr float guard=-987654.0f;

static void fill_weights(std::vector<block_q4_Kx8>&w,int pattern) {
    for(auto &b:w) {
        for(int i=0;i<8;++i) {b.d[i]=pattern?0x3c00:0x2000;b.dmin[i]=pattern?0x3800:0x1800;}
        for(auto &v:b.scales)v=pattern==1?0:pattern?255:uint8_t(random_bits());
        for(auto &v:b.qs)v=pattern==1?0:pattern==2||pattern==3?255:uint8_t(random_bits());
    }
}
static int8_t edge(int pattern,size_t i) {
    if(pattern==1)return 0;
    if(pattern==2)return -128;
    if(pattern==3)return 127;
    return i%2?127:-128;
}
static void fill_q8(std::vector<block_q8_K>&a,int pattern) {
    for(auto &b:a) {
        b.d=pattern?0.01f:0.03125f;
        for(size_t i=0;i<256;++i)b.qs[i]=pattern?edge(pattern,i):int8_t(random_bits());
        for(int j=0;j<16;++j) {int sum=0;for(int k=0;k<16;++k)sum+=b.qs[j*16+k];b.bsums[j]=sum;}
    }
}
static void fill_q8x4(std::vector<block_q8_Kx4>&a,int k,int rows,int pattern,Quant quant) {
    if(!pattern) {
        std::vector<float> raw(size_t(k)*4);
        for(int group=0;group<rows/4;++group) {
            for(auto &v:raw)v=float(int32_t(random_bits()%20001)-10000)/8192.f;
            quant(raw.data(),a.data()+size_t(group)*(k/256),k);
        }
    } else {
        const int sum=pattern==1?0:pattern==2?-2048:pattern==3?2032:-8;
        for(auto &b:a) {
            for(auto &d:b.d)d=0.01f;
            for(size_t i=0;i<1024;++i)b.qs[i]=edge(pattern,i);
            for(auto &s:b.bsums)s=sum;
        }
    }
}
static void compare(const char *kind,int k,int nc,int nr,int stride,int pattern,Mat old,Mat candidate,
                    const void*w,const void*a) {
    const size_t size=32+size_t(stride)*nr;
    std::vector<float>x(size,guard),y(size,guard);
    old(k,x.data()+16,stride,w,a,nr,nc);candidate(k,y.data()+16,stride,w,a,nr,nc);
    for(size_t i=0;i<size;++i) {
        const bool active=i>=16&&i<16+size_t(stride)*nr&&(i-16)%stride<size_t(nc);
        if(std::memcmp(&x[i],&y[i],sizeof(float)) || !std::isfinite(x[i]) || (!active&&x[i]!=guard)) {
            fprintf(stderr,"MISMATCH %s K=%d N=%d M=%d stride=%d pattern=%d i=%zu old=%.9g new=%.9g active=%d\n",kind,k,nc,nr,stride,pattern,i,x[i],y[i],active);
            std::exit(2);
        }
    }
    ++cases;values+=size_t(nc)*nr;
}
static double time_one(int k,int nc,int nr,Mat f,const void*w,const void*a,std::vector<float>&out) {
    auto start=std::chrono::steady_clock::now();f(k,out.data(),nc,w,a,nr,nc);
    return double(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count());
}
static void micro(const char*kind,int k,int nc,int nr,Mat old,Mat candidate,const void*w,const void*a) {
    std::vector<float> out(size_t(nc)*nr);std::vector<double>x,y;
    old(k,out.data(),nc,w,a,nr,nc);candidate(k,out.data(),nc,w,a,nr,nc);
    for(int rep=0;rep<6;++rep) {
        if(rep%2) {y.push_back(time_one(k,nc,nr,candidate,w,a,out));x.push_back(time_one(k,nc,nr,old,w,a,out));}
        else {x.push_back(time_one(k,nc,nr,old,w,a,out));y.push_back(time_one(k,nc,nr,candidate,w,a,out));}
    }
    std::sort(x.begin(),x.end());std::sort(y.begin(),y.end());double before=(x[2]+x[3])/2,after=(y[2]+y[3])/2;
    printf("{\"kind\":\"micro\",\"kernel\":\"%s\",\"k\":%d,\"n\":%d,\"m\":%d,\"before_ns\":%.0f,\"after_ns\":%.0f,\"speedup\":%.6f}\n",kind,k,nc,nr,before,after,before/after);fflush(stdout);
}
int main(int argc,char**argv) {
    if(argc!=3)return 1;
    void *a=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL),*b=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);
    if(!a||!b){fprintf(stderr,"dlopen: %s\n",dlerror());return 1;}
    Mat av=(Mat)dlsym(a,"ggml_gemv_q4_K_8x8_q8_K"),bv=(Mat)dlsym(b,"ggml_gemv_q4_K_8x8_q8_K");
    Mat am=(Mat)dlsym(a,"ggml_gemm_q4_K_8x8_q8_K"),bm=(Mat)dlsym(b,"ggml_gemm_q4_K_8x8_q8_K");
    Quant quant=(Quant)dlsym(a,"ggml_quantize_mat_q8_K_4x8");
    if(!av||!bv||!am||!bm||!quant||av==bv||am==bm)return 1;
    for(int k:{256,512,1024,2560,9728})for(int nc:{8,16,24,40})for(int pattern=0;pattern<5;++pattern) {
        std::vector<block_q4_Kx8>w(size_t(k/256)*(nc/8));fill_weights(w,pattern);
        std::vector<block_q8_K>act(k/256);fill_q8(act,pattern);
        compare("gemv",k,nc,1,nc+8,pattern,av,bv,w.data(),act.data());
        for(int nr:{4,8,12,16,20,32}) {
            std::vector<block_q8_Kx4>mat(size_t(k/256)*(nr/4));fill_q8x4(mat,k,nr,pattern,quant);
            compare("gemm",k,nc,nr,nc+8,pattern,am,bm,w.data(),mat.data());
        }
    }
    for(auto shape:std::vector<std::pair<int,int>>{{2560,2560},{2560,9728},{9728,2560},{2560,1024}}) {
        int k=shape.first,nc=shape.second;
        std::vector<block_q4_Kx8>w(size_t(k/256)*(nc/8));fill_weights(w,0);
        std::vector<block_q8_K>act(k/256);fill_q8(act,0);
        std::vector<block_q8_Kx4>mat(size_t(k/256)*4);fill_q8x4(mat,k,16,0,quant);
        compare("gemv-real",k,nc,1,nc,0,av,bv,w.data(),act.data());
        compare("gemm-real",k,nc,16,nc,0,am,bm,w.data(),mat.data());
        micro("gemv",k,nc,1,av,bv,w.data(),act.data());micro("gemm",k,nc,16,am,bm,w.data(),mat.data());
    }
    printf("{\"kind\":\"correctness\",\"passed\":true,\"cases\":%zu,\"output_values\":%zu,\"bitwise_identical\":true,\"canaries_intact\":true}\n",cases,values);
}
