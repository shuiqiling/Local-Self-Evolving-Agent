#include "tiled-kernel.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <random>
#include <vector>
using Run = void(*)(int,const tiled_tile_src0*,const tiled_tile_src1*,int,int,int,int,float*,int);
using Repack = void(*)(int,tiled_tile_src0*,tiled_tile_src1*,int,int);
struct Lib {void * h; Run run; Repack repack;};
static std::mt19937 rng(20261031);
static constexpr int stride = 272;
static constexpr float guard = -9876.5f;
static int cases = 0;
static size_t values = 0;
static int8_t activation(int pattern, int k) {
    if (pattern == 1) return 0;
    if (pattern == 2) return -128;
    if (pattern == 3) return 127;
    if (pattern == 4) return k%2 ? -128 : 127;
    return (int8_t) rng();
}
static void fill(int fmt,int nk,int rows,int pattern,tiled_tile_src0 & a,tiled_tile_src1 & b) {
    std::memset(&a,0,sizeof(a)); std::memset(&b,0,sizeof(b));
    const int biases[] = {0,128,128,128,32,4,0};
    const int maxes[] = {31,127,62,62,32,4,3};
    const int sb = fmt < 3 ? 32 : 16;
    const int nb = 256/sb, bias = biases[fmt], mx = maxes[fmt];
    for (int row=0; row<rows; ++row) for (int slab=0; slab<nk; ++slab) {
        a.d[slab*16+row] = pattern ? 0.03125f : float(rng()%500+1)/32768;
        a.dmin[slab*16+row] = pattern ? 0.015625f : float(rng()%500)/65536;
        b.d[slab*16+row] = pattern ? 0.0078125f : float(rng()%500+1)/16384;
        for (int s=0;s<nb;++s) {
            const int z=row*nb*nk+slab*nb+s;
            a.scales[z] = bias ? int(rng()%256)-128 : int(rng()%64);
            a.mins[z] = rng()%64;
        }
        for (int k=0;k<256;++k) {
            const int z=row*nk*256+slab*256+k;
            const int w=pattern==5 ? (k%2 ? (bias ? -mx : 0) : mx-1)
                                  : (bias ? int(rng()%(2*mx))-mx : int(rng()%(mx+1)));
            a.q[z] = uint8_t(bias+w);
            b.q[z] = activation(pattern,k);
        }
        for (int u=0;u<16;++u) {
            int sum=0; for(int k=0;k<16;++k) sum += b.q[row*nk*256+slab*256+u*16+k];
            b.bsums[(slab*16+u)*(nk==1?256:16)+row] = sum;
        }
    }
}
static void check(Lib old,Lib next,int fmt,int nk,int pattern) {
    const int rows = nk==1 ? 64 : 16;
    auto a=std::make_unique<tiled_tile_src0>(); auto b=std::make_unique<tiled_tile_src1>();
    fill(fmt,nk,rows,pattern,*a,*b);
    auto a1=std::make_unique<tiled_tile_src0>(*a); auto b1=std::make_unique<tiled_tile_src1>(*b);
    auto a2=std::make_unique<tiled_tile_src0>(*a); auto b2=std::make_unique<tiled_tile_src1>(*b);
    old.repack(fmt,a1.get(),b1.get(),rows,nk); next.repack(fmt,a2.get(),b2.get(),rows,nk);
    for (int pos=0;pos<rows;pos+=16) {
        std::vector<float> x(16+size_t(stride)*rows+16,guard),y=x;
        const int j=rows-16-pos;
        for (int slab=0;slab<nk;++slab) {
            old.run(fmt,a1.get(),b1.get(),pos,j,nk,slab,x.data()+16,stride);
            next.run(fmt,a2.get(),b2.get(),pos,j,nk,slab,y.data()+16,stride);
        }
        for (size_t z=0;z<x.size();++z) {
            const bool active = z>=16 && z<16+size_t(stride)*rows &&
                (z-16)/stride>=size_t(pos) && (z-16)/stride<size_t(pos+16) &&
                (z-16)%stride>=size_t(j) && (z-16)%stride<size_t(j+16);
            if (std::memcmp(&x[z],&y[z],4) || !std::isfinite(y[z]) || (!active && y[z]!=guard)) {
                std::fprintf(stderr,"mismatch fmt=%d nk=%d pattern=%d pos=%d z=%zu old=%.9g new=%.9g\n",fmt,nk,pattern,pos,z,x[z],y[z]);
                std::exit(2);
            }
        }
        // An independent scalar integer oracle for Q4/Q5, Q6, Q3 and Q2.
        if (fmt==0 || fmt==4 || fmt==5 || fmt==6) {
            const int sb=fmt==0 ? 32 : 16, nb=256/sb, bias=fmt==4 ? 32 : fmt==5 ? 4 : 0;
            const bool has_min=fmt==0 || fmt==6;
            for (int row=pos;row<pos+16;++row) for (int col=j;col<j+16;++col) {
                float want=guard;
                for(int slab=0;slab<nk;++slab) {
                    int sum=0,mins=0;
                    for(int s=0;s<nb;++s) {
                        int raw=0,bsum=0;
                        for(int k=0;k<sb;++k) {
                            int act=b->q[col*nk*256+slab*256+s*sb+k];
                            raw+=(int(a->q[row*nk*256+slab*256+s*sb+k])-bias)*act; bsum+=act;
                        }
                        int z=row*nb*nk+slab*nb+s;sum+=a->scales[z]*raw;
                        if(has_min) mins+=a->mins[z]*bsum;
                    }
                    float part=a->d[slab*16+row]*float(sum);
                    if(has_min) part=std::fma(-a->dmin[slab*16+row],float(mins),part);
                    want=std::fma(part,b->d[slab*16+col],want);
                }
                float got=y[16+row*stride+col];
                if(std::memcmp(&want,&got,4)) {std::fprintf(stderr,"oracle mismatch fmt=%d nk=%d want=%.9g got=%.9g\n",fmt,nk,want,got);std::exit(3);}
            }
        }
        ++cases;values+=256;
    }
}
static double bench(Lib lib,int fmt,tiled_tile_src0 * a,tiled_tile_src1 * b,int nk) {
    std::vector<float> out(stride*16);
    auto start=std::chrono::steady_clock::now();
    for(int rep=0;rep<400;++rep) {
        std::fill(out.begin(),out.end(),0.f);
        for(int slab=0;slab<nk;++slab) lib.run(fmt,a,b,0,0,nk,slab,out.data(),stride);
    }
    return std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/400;
}
int main(int argc,char ** argv) {
    if(argc!=3)return 1;
    Lib libs[2];
    for(int i=0;i<2;++i) {
        libs[i].h=dlopen(argv[i+1],RTLD_NOW|RTLD_LOCAL);
        if(!libs[i].h){std::fprintf(stderr,"%s\n",dlerror());return 1;}
        libs[i].run=(Run)dlsym(libs[i].h,"sq_run");libs[i].repack=(Repack)dlsym(libs[i].h,"sq_repack");
        if(!libs[i].run||!libs[i].repack)return 1;
    }
    for(int f=0;f<7;++f)for(int nk:{1,2,4,8,16})for(int p=0;p<6;++p)check(libs[0],libs[1],f,nk,p);
    std::printf("{\"kind\":\"correctness\",\"cases\":%d,\"values\":%zu,\"bitwise_equal\":true,\"integer_oracle\":true,\"canaries\":true}\n",cases,values);
    for(int fmt:{0,4})for(int nk:{1,4,16}) {
        auto a=std::make_unique<tiled_tile_src0>();auto b=std::make_unique<tiled_tile_src1>();fill(fmt,nk,16,0,*a,*b);
        auto a2=std::make_unique<tiled_tile_src0>(*a);auto b2=std::make_unique<tiled_tile_src1>(*b);
        libs[0].repack(fmt,a.get(),b.get(),16,nk);libs[1].repack(fmt,a2.get(),b2.get(),16,nk);
        std::vector<double>x,y;
        for(int r=0;r<6;++r){if(r%2){y.push_back(bench(libs[1],fmt,a2.get(),b2.get(),nk));x.push_back(bench(libs[0],fmt,a.get(),b.get(),nk));}
            else{x.push_back(bench(libs[0],fmt,a.get(),b.get(),nk));y.push_back(bench(libs[1],fmt,a2.get(),b2.get(),nk));}}
        std::sort(x.begin(),x.end());std::sort(y.begin(),y.end());
        std::printf("{\"kind\":\"micro\",\"format\":%d,\"nk\":%d,\"old_ns\":%.1f,\"new_ns\":%.1f,\"speedup\":%.4f}\n",fmt,nk,x[3],y[3],x[3]/y[3]);
    }
}
