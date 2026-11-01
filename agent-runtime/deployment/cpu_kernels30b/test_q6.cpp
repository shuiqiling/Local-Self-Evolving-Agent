#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <vector>
using Dot=void(*)(int,float*,size_t,const void*,size_t,const void*,size_t,int);
static std::mt19937 rng(10312026);
static void fill(std::vector<block_q6_K>&w,std::vector<block_q8_K>&a,int pattern) {
    for(auto &b:w) {
        b.d=pattern ? 0x3c00 : 0x2800+(rng()%0x400);
        for(auto &q:b.ql)q=pattern==1?0:pattern==2?255:uint8_t(rng());
        for(auto &q:b.qh)q=pattern==1?0:pattern==2?255:uint8_t(rng());
        for(auto &s:b.scales)s=pattern==3?-128:pattern==4?127:int8_t(rng());
    }
    for(auto &b:a) {
        b.d=pattern?0.00390625f:float(rng()%500+1)/65536;
        for(int k=0;k<256;++k)b.qs[k]=pattern==1?0:pattern==2||pattern==3?-128:pattern==4?127:pattern==5?(k%2?-128:127):int8_t(rng());
        for(int k=0;k<16;++k){int sum=0;for(int j=0;j<16;++j)sum+=b.qs[k*16+j];b.bsums[k]=sum;}
    }
}
static double bench(Dot f,int n,std::vector<block_q6_K>&w,std::vector<block_q8_K>&a,int rows) {
    float out[128];auto t=std::chrono::steady_clock::now();
    for(int rep=0;rep<80;++rep)for(int row=0;row<rows;++row)f(n,out+row,0,w.data()+row*(n/256),0,a.data(),0,1);
    return std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-t).count()/(80*rows);
}
int main(int argc,char**argv) {
    if(argc!=3)return 1;
    auto l=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL),r=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);
    if(!l||!r){fprintf(stderr,"%s\n",dlerror());return 1;}
    auto old=(Dot)dlsym(l,"ggml_vec_dot_q6_K_q8_K"),next=(Dot)dlsym(r,"ggml_vec_dot_q6_K_q8_K");
    if(!old||!next||old==next)return 1;
    int cases=0;
    for(int n:{256,512,768,1024,2048,2560,4096,8192,9728})for(int p=0;p<38;++p){
        std::vector<block_q6_K>w(n/256);std::vector<block_q8_K>a(n/256);fill(w,a,p<6?p:0);
        float x[3]={123.f,0.f,456.f},y[3]={123.f,0.f,456.f};
        old(n,x+1,0,w.data(),0,a.data(),0,1);next(n,y+1,0,w.data(),0,a.data(),0,1);
        if(memcmp(x,y,sizeof(x))||!std::isfinite(y[1])){fprintf(stderr,"q6 mismatch n=%d p=%d old=%.9g new=%.9g\n",n,p,x[1],y[1]);return 2;}++cases;
    }
    printf("{\"kind\":\"q6_correctness\",\"cases\":%d,\"bitwise_equal\":true,\"canaries\":true}\n",cases);
    for(int n:{1024,2560,4096,8192}) {
        int rows=64;std::vector<block_q6_K>w(rows*n/256);std::vector<block_q8_K>a(n/256);fill(w,a,0);
        std::vector<double>x,y;
        for(int rep=0;rep<8;++rep){if(rep%2){y.push_back(bench(next,n,w,a,rows));x.push_back(bench(old,n,w,a,rows));}
            else{x.push_back(bench(old,n,w,a,rows));y.push_back(bench(next,n,w,a,rows));}}
        std::sort(x.begin(),x.end());std::sort(y.begin(),y.end());
        printf("{\"kind\":\"q6_micro\",\"n\":%d,\"old_ns\":%.1f,\"new_ns\":%.1f,\"speedup\":%.5f}\n",n,x[4],y[4],x[4]/y[4]);
    }
}
