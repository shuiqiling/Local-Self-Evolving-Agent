// User-space diagnostic only. Never preload this into the production service.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

using Dot = void (*)(int,float*,size_t,const void*,size_t,const void*,size_t,int);
using Mat = void (*)(int,float*,size_t,const void*,const void*,int,int);
static const char * names[] = {"q4_dot", "q6_dot", "q4_gemv", "q4_gemm"};
static void * functions[4];
static const char * output;
struct Counter { uint64_t calls=0, samples=0, ns=0; };
struct Counters {
    Counter values[4];
    ~Counters() {
        if (!output) return;
        const int fd = open(output,O_WRONLY|O_CREAT|O_APPEND,0600);
        if (fd<0) return;
        for (int i=0;i<4;++i) {
            auto &v=values[i]; if (!v.calls) continue;
            char line[320];
            const int n=snprintf(line,sizeof(line),"{\"tid\":%ld,\"function\":\"%s\",\"calls\":%llu,\"samples\":%llu,\"thread_cpu_ns\":%llu}\n",
                syscall(SYS_gettid),names[i],(unsigned long long)v.calls,
                (unsigned long long)v.samples,(unsigned long long)v.ns);
            if(n>0) { const auto ignored=write(fd,line,n);(void)ignored; }
        }
        close(fd);
    }
};
static thread_local Counters counters;
static uint64_t cpu_ns() { timespec t;clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return uint64_t(t.tv_sec)*1000000000ULL+t.tv_nsec; }
__attribute__((constructor)) static void initialize() {
    output=getenv("LEA_PROFILE_OUTPUT");
    const char *path=getenv("LEA_PROFILE_CPU_LIB");
    if(!path) { fprintf(stderr,"Missing diagnostic CPU library path\n");abort(); }
    void *lib=dlopen(path,RTLD_NOW|RTLD_LOCAL);
    if(!lib) { fprintf(stderr,"Profile dlopen: %s\n",dlerror());abort(); }
    const char *symbols[]={"ggml_vec_dot_q4_K_q8_K","ggml_vec_dot_q6_K_q8_K",
                           "ggml_gemv_q4_K_8x8_q8_K","ggml_gemm_q4_K_8x8_q8_K"};
    for(int i=0;i<4;++i) { functions[i]=dlsym(lib,symbols[i]);if(!functions[i])abort(); }
}
#define DOT_WRAPPER(NAME,INDEX) \
extern "C" void NAME(int n,float*s,size_t bs,const void*x,size_t bx,const void*y,size_t by,int nrc) { \
    auto &v=counters.values[INDEX];auto f=reinterpret_cast<Dot>(functions[INDEX]);++v.calls; \
    if((v.calls&127)!=1) {f(n,s,bs,x,bx,y,by,nrc);return;} \
    const uint64_t start=cpu_ns();f(n,s,bs,x,bx,y,by,nrc);v.ns+=cpu_ns()-start;++v.samples; }
#define MAT_WRAPPER(NAME,INDEX) \
extern "C" void NAME(int n,float*s,size_t bs,const void*x,const void*y,int nr,int nc) { \
    auto &v=counters.values[INDEX];auto f=reinterpret_cast<Mat>(functions[INDEX]);++v.calls; \
    const uint64_t start=cpu_ns();f(n,s,bs,x,y,nr,nc);v.ns+=cpu_ns()-start;++v.samples; }
DOT_WRAPPER(ggml_vec_dot_q4_K_q8_K,0)
DOT_WRAPPER(ggml_vec_dot_q6_K_q8_K,1)
MAT_WRAPPER(ggml_gemv_q4_K_8x8_q8_K,2)
MAT_WRAPPER(ggml_gemm_q4_K_8x8_q8_K,3)
