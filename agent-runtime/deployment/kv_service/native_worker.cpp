#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "json.hpp"
#include "json-schema-to-grammar.h"
#include <filesystem>
#include <memory>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <array>

using json=nlohmann::json;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point p){return std::chrono::duration<double,std::milli>(Clock::now()-p).count();}
static void need(bool ok,const std::string& s){if(!ok)throw std::runtime_error(s);}

using Selection=std::array<std::array<std::vector<int>,8>,36>;
struct Audit {
    long decode_calls=0,decoded_tokens=0,fa_callbacks=0;int scan=0,fa_layer=0;json shape;
    bool gating=false,masking=false;int fd=-1;double gate_ms=0,readback_ms=0,mask_ms=0;
    std::array<std::array<std::vector<float>,8>,36> scores;
    Selection selected;
    ggml_tensor replacement{};ggml_tensor* original=nullptr;std::vector<ggml_fp16_t> mask;
};
static void transfer(int fd,void* data,size_t n,bool send_data){
    char* p=(char*)data;
    while(n){ssize_t k=send_data?::send(fd,p,n,MSG_NOSIGNAL) : ::recv(fd,p,n,0);need(k>0,"gate socket I/O");p+=k;n-=k;}
}
static bool observe(ggml_tensor* t,bool ask,void* user){
    auto& a=*static_cast<Audit*>(user);std::string name=ggml_get_name(t);
    if(a.gating && name.rfind("attn_norm-",0)==0){
        if(ask)return true;
        int layer=std::stoi(name.substr(10));need(layer>=0&&layer<36,"gate layer");
        need(t->type==GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[0]==2560,"gate hidden layout");
        auto timer=Clock::now();std::vector<float> x(ggml_nelements(t));ggml_backend_tensor_get(t,x.data(),0,ggml_nbytes(t));
        a.readback_ms+=ms(timer);timer=Clock::now();int32_t header[3]={layer,(int32_t)t->ne[1],2560};
        transfer(a.fd,header,sizeof header,true);transfer(a.fd,x.data(),x.size()*4,true);
        std::vector<float> scores(8*t->ne[1]);transfer(a.fd,scores.data(),scores.size()*4,false);
        for(int h=0;h<8;++h)a.scores[layer][h].insert(a.scores[layer][h].end(),scores.begin()+h*t->ne[1],scores.begin()+(h+1)*t->ne[1]);
        a.gate_ms+=ms(timer);return true;
    }
    if(t->op==GGML_OP_FLASH_ATTN_EXT){
        if(ask){
            ++a.fa_callbacks;auto* k=t->src[1];int layer=a.fa_layer++;
            need(layer<36,"unexpected extra flash operation");
            if(k && k->ne[1]>=a.scan){a.scan=(int)k->ne[1];a.shape={{"name",name},{"k",{k->ne[0],k->ne[1],k->ne[2],k->ne[3]}}};}
            if(a.masking){
                // CPU-only reference: expose the same retained set without
                // moving rows, to check the packed representation separately.
                auto timer=Clock::now();auto* base=t->src[3];need(base && base->type==GGML_TYPE_F16 && base->ne[2]==1 && base->ne[3]==1,"base mask layout");
                need(t->src[0]->ne[2]==32 && k->ne[2]==8,"GQA layout");
                size_t plane=base->ne[0]*base->ne[1];std::vector<ggml_fp16_t> original(plane);
                ggml_backend_tensor_get(base,original.data(),0,plane*2);a.mask.resize(plane*32);
                for(int hq=0;hq<32;++hq){
                    std::copy(original.begin(),original.end(),a.mask.begin()+hq*plane);
                    std::vector<bool> valid(3072,false);for(int p:a.selected[layer][hq/4])valid[p]=true;
                    for(int q=0;q<base->ne[1];++q)for(int p=0;p<3072;++p)if(!valid[p])a.mask[hq*plane+q*base->ne[0]+p]=ggml_fp32_to_fp16(-INFINITY);
                }
                a.replacement=*base;a.replacement.ne[2]=32;a.replacement.nb[3]=plane*32*2;a.replacement.data=a.mask.data();
                a.original=base;t->src[3]=&a.replacement;a.mask_ms+=ms(timer);return true;
            }
        }else if(a.masking && a.original){t->src[3]=a.original;a.original=nullptr;return true;}
    }
    return !ask;
}
struct Engine {
    llama_model* model=nullptr;llama_context* ctx=nullptr;const llama_vocab* vocab=nullptr;Audit a;
    ~Engine(){if(ctx)llama_free(ctx);if(model)llama_model_free(model);llama_backend_free();}
    void open(const std::string& root,int context=8192,int sequences=2){
        auto release=root+"/runtime/q4-vnni-20261029-r1";
        ggml_backend_load_all_from_path(release.c_str());llama_backend_init();
        auto mp=llama_model_default_params();mp.n_gpu_layers=0;
        model=llama_model_load_from_file((root+"/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf").c_str(),mp);need(model,"model load");
        vocab=llama_model_get_vocab(model);auto cp=llama_context_default_params();
        cp.n_ctx=context;cp.n_seq_max=sequences;cp.kv_unified=true;cp.n_batch=512;cp.n_ubatch=512;
        cp.n_threads=6;cp.n_threads_batch=16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k=GGML_TYPE_F16;cp.type_v=GGML_TYPE_F16;cp.no_perf=false;
        cp.cb_eval=observe;cp.cb_eval_user_data=&a;
        ctx=llama_init_from_model(model,cp);need(ctx,"context load");
        auto* adapter=llama_adapter_lora_init(model,(root+"/adapters/concise-v2/concise-v2-f16.gguf").c_str());need(adapter,"adapter load");
        float scale=1;need(llama_set_adapters_lora(ctx,&adapter,1,&scale)==0,"adapter set");
    }
    std::vector<llama_token> tok(const std::string& text){
        int n=llama_tokenize(vocab,text.c_str(),text.size(),nullptr,0,false,true);need(n<0,"token size");
        std::vector<llama_token> out(-n);n=llama_tokenize(vocab,text.c_str(),text.size(),out.data(),out.size(),false,true);need(n>=0,"tokenize");out.resize(n);return out;
    }
    void decode(const std::vector<llama_token>& tokens,int pos,int seq){
        need(!tokens.empty(),"empty decode");
        for(size_t off=0;off<tokens.size();off+=512){
            int n=std::min<size_t>(512,tokens.size()-off);auto b=llama_batch_init(n,0,1);b.n_tokens=n;
            for(int i=0;i<n;++i){b.token[i]=tokens[off+i];b.pos[i]=pos+off+i;b.n_seq_id[i]=1;b.seq_id[i][0]=seq;b.logits[i]=(off+i+1==tokens.size());}
            a.fa_layer=0;++a.decode_calls;a.decoded_tokens+=n;int result=llama_decode(ctx,b);llama_batch_free(b);need(result==0,"decode failed "+std::to_string(result));
        }
        llama_synchronize(ctx);
    }
    std::vector<uint8_t> save(int seq){
        size_t n=llama_state_seq_get_size(ctx,seq);need(n>16,"state size");std::vector<uint8_t> b(n);
        need(llama_state_seq_get_data(ctx,b.data(),b.size(),seq)==n,"state export");return b;
    }
    void load(const std::vector<uint8_t>& b,int seq){
        llama_memory_clear(llama_get_memory(ctx),false);
        need(llama_state_seq_set_data(ctx,b.data(),b.size(),seq)==b.size(),"state import");
    }
    std::string piece(llama_token t){char b[1024];int n=llama_token_to_piece(vocab,t,b,sizeof b,0,true);need(n>=0,"piece");return {b,(size_t)n};}
    json answer(const std::vector<llama_token>& delta,int pos,int seq,int limit){
        a.scan=0;auto start=Clock::now();long before=a.decoded_tokens;decode(delta,pos,seq);
        json row;row["delta_tokens"]=delta.size();row["first_step_scan"]=a.scan;row["attention_shape"]=a.shape;
        auto* f=llama_get_logits_ith(ctx,-1);int nv=llama_vocab_n_tokens(vocab);
        std::vector<float> first(f,f+nv);row["first_logits"]=first;
        std::vector<llama_token> ids;std::string text;double ttft=0;
        for(int i=0;i<limit;++i){
            f=llama_get_logits_ith(ctx,-1);llama_token t=std::max_element(f,f+nv)-f;
            if(i==0)ttft=ms(start);
            if(llama_vocab_is_eog(vocab,t))break;
            ids.push_back(t);text+=piece(t);decode({t},pos+delta.size()+i,seq);
        }
        row["tokens"]=ids;row["text"]=text;row["ttft_ms"]=ttft;row["answer_ms"]=ms(start);
        row["actually_decoded_tokens"]=a.decoded_tokens-before;row["final_scan"]=a.scan;return row;
    }
};
template<class T>static T read(const std::vector<uint8_t>& b,size_t& p){need(p+sizeof(T)<=b.size(),"state parse bounds");T v;std::memcpy(&v,b.data()+p,sizeof v);p+=sizeof v;return v;}
struct State {uint32_t count;std::vector<int> positions;size_t data_offset;uint32_t layers;};
static State inspect(const std::vector<uint8_t>& b){
    // Pinned 86ea01d, ordinary Qwen3: no MRoPE/PLE cell extension, one unified stream.
    size_t p=0;need(read<uint32_t>(b,p)==0xaf143cd8,"state magic");read<int32_t>(b,p);
    need(read<uint32_t>(b,p)==1,"one stream required");State s;s.count=read<uint32_t>(b,p);
    for(uint32_t i=0;i<s.count;++i){s.positions.push_back(read<int32_t>(b,p));need(read<uint32_t>(b,p)==1,"one owner required");read<int32_t>(b,p);}
    s.data_offset=p;need(read<uint32_t>(b,p)==0,"non-transposed V required");s.layers=read<uint32_t>(b,p);need(s.layers==36,"Qwen3 layer count");
    for(int kind=0;kind<2;++kind)for(uint32_t layer=0;layer<s.layers;++layer){
        need(read<int32_t>(b,p)==GGML_TYPE_F16,"unchanged F16 KV required");auto width=read<uint64_t>(b,p);need(width>0&&width<65536,"KV row width");p+=width*s.count;need(p<=b.size(),"KV data bounds");}
    need(p==b.size(),"state format mismatch");return s;
}
static bool same_kv_and_positions(const std::vector<uint8_t>& x,const std::vector<uint8_t>& y){
    auto a=inspect(x),b=inspect(y);return a.positions==b.positions && x.size()-a.data_offset==y.size()-b.data_offset &&
        std::equal(x.begin()+a.data_offset,x.end(),y.begin()+b.data_offset);
}
static void write_report(const std::string& path,const json& report){std::ofstream f(path);f<<report.dump(2)<<"\n";}

static double logits_diff(const json& a,const json& b){
    auto x=a["first_logits"].get<std::vector<float>>(),y=b["first_logits"].get<std::vector<float>>();double d=0;
    for(size_t i=0;i<x.size();++i)d=std::max(d,(double)std::abs(x[i]-y[i]));return d;
}
template<class T>static void append(std::vector<uint8_t>& b,T x){size_t n=b.size();b.resize(n+sizeof x);std::memcpy(b.data()+n,&x,sizeof x);}
static std::vector<uint8_t> pack_heads(const std::vector<uint8_t>& initial,const Selection& selected,const Selection& positions){
    auto info=inspect(initial);const int count=info.count;need(count>614 && count<=3072,"packing count");
    const int keep=selected[0][0].size();need(keep>192 && keep<=3072,"head budget");
    std::vector<uint8_t> out;out.reserve(1024+keep*147468);
    append<uint32_t>(out,0xaf143cd8);append<int32_t>(out,0);append<uint32_t>(out,1);append<uint32_t>(out,keep);
    // Native allocator metadata uses a representative chronological map.
    // TRUE positions for every layer/head remain in the separate Selection map.
    // This restricted probe only appends at monotonically increasing absolute positions, so all
    // old cells are causal for every new query. NO position transforms, partial
    // historical rollback, slot sharing, or production serialization supported.
    need(info.positions==positions[0][0],"representative metadata/ledger mismatch");
    for(int index:selected[0][0]){append<int32_t>(out,positions[0][0][index]);append<uint32_t>(out,1);append<int32_t>(out,0);}
    size_t p=info.data_offset;auto vt=read<uint32_t>(initial,p);auto nl=read<uint32_t>(initial,p);append(out,vt);append(out,nl);
    for(int kind=0;kind<2;++kind)for(int layer=0;layer<36;++layer){
        auto type=read<int32_t>(initial,p);auto width=read<uint64_t>(initial,p);need(type==GGML_TYPE_F16 && width==2048,"F16 8x128 cache rows");
        append(out,type);append(out,width);size_t base=out.size();out.resize(base+keep*width);
        for(int h=0;h<8;++h){need(selected[layer][h].size()==size_t(keep),"uniform head budget");
            for(int j=0;j<keep;++j){int pos=selected[layer][h][j];need(pos>=0&&pos<count,"absolute head position");
                std::memcpy(out.data()+base+j*width+h*256,initial.data()+p+pos*width+h*256,256);}}
        p+=count*width;
    }
    need(p==initial.size(),"packing consumed state");need(inspect(out).positions.back()==positions[0][0].back(),"next absolute position invariant");return out;
}



struct Snapshot {
    std::vector<uint8_t> kv;
    Selection positions;
    decltype(Audit{}.scores) scores;
    int next=0,cuts=0;
};
struct Online {
    Engine e;Selection positions;int next=0,cuts=0;
    std::unique_ptr<Snapshot> pending;
    json cut_log=json::array();
    int live()const{return positions[0][0].size();}
    void clear(){
        need(!pending,"transaction pending");llama_memory_clear(llama_get_memory(e.ctx),false);
        positions=Selection{};e.a.scores=decltype(e.a.scores){};next=cuts=0;cut_log=json::array();
    }
    void begin(){
        need(!pending,"transaction pending");pending=std::make_unique<Snapshot>();
        if(live())pending->kv=e.save(0);
        pending->positions=positions;pending->scores=e.a.scores;pending->next=next;pending->cuts=cuts;
    }
    void rollback(){
        if(!pending)return;
        if(pending->kv.empty())llama_memory_clear(llama_get_memory(e.ctx),false);else e.load(pending->kv,0);
        positions=std::move(pending->positions);e.a.scores=std::move(pending->scores);
        next=pending->next;cuts=pending->cuts;pending.reset();e.a.gating=false;cut_log=json::array();
    }
    json status()const{
        return {{"physical_tokens",live()},{"next_absolute_position",next},{"total_cuts",cuts},
            {"window_budget",3072},{"retain_budget",614},{"protected_prefix",64},{"protected_suffix",128},
            {"transaction_pending",bool(pending)}};
    }
    void compress(){
        if(live()<=614)return;
        int count=live();long calls=e.a.decode_calls,tokens=e.a.decoded_tokens,attn=e.a.fa_callbacks;
        auto all=Clock::now();auto t=Clock::now();auto before=e.save(0);double backup_ms=ms(t);
        need(inspect(before).positions==positions[0][0],"representative ledger mismatch");
        t=Clock::now();e.a.selected=Selection{};
        for(int l=0;l<36;++l)for(int h=0;h<8;++h){
            need(positions[l][h].size()==size_t(count)&&e.a.scores[l][h].size()==size_t(count),"unaligned scores");
            auto& scores=e.a.scores[l][h];std::vector<int> middle(count-192);std::iota(middle.begin(),middle.end(),64);
            for(float s:scores)need(std::isfinite(s),"non-finite score");
            std::stable_sort(middle.begin(),middle.end(),[&](int a,int b){return scores[a]>scores[b];});middle.resize(422);
            auto& selected=e.a.selected[l][h];for(int i=0;i<64;++i)selected.push_back(i);
            selected.insert(selected.end(),middle.begin(),middle.end());for(int i=count-128;i<count;++i)selected.push_back(i);
            std::sort(selected.begin(),selected.end());
        }
        double selection_ms=ms(t);t=Clock::now();auto packed=pack_heads(before,e.a.selected,positions);double pack_ms=ms(t);
        Selection new_positions;auto new_scores=e.a.scores;
        for(int l=0;l<36;++l)for(int h=0;h<8;++h){
            new_scores[l][h].clear();for(int i:e.a.selected[l][h]){
                new_positions[l][h].push_back(positions[l][h][i]);new_scores[l][h].push_back(e.a.scores[l][h][i]);
            }
        }
        t=Clock::now();e.load(packed,0);double import_ms=ms(t),total_ms=ms(all);
        need(e.a.decode_calls==calls&&e.a.decoded_tokens==tokens&&e.a.fa_callbacks==attn,"model executed during cut");
        need(same_kv_and_positions(packed,e.save(0)),"packed KV import changed bytes");
        positions=std::move(new_positions);e.a.scores=std::move(new_scores);++cuts;
        cut_log.push_back({{"cut",cuts},{"before",count},{"after",614},{"next_absolute_position",next},
            {"backup_ms",backup_ms},{"selection_ms",selection_ms},{"pack_ms",pack_ms},{"import_ms",import_ms},
            {"seconds",total_ms/1000},{"old_tokens_recomputed",0},{"decode_calls",0},{"attention_callbacks",0},
            {"retained_kv_bitwise_equal",true},{"before_bytes",before.size()},{"after_bytes",packed.size()}});
    }
    void append(const std::vector<llama_token>& input){
        for(size_t offset=0;offset<input.size();){
            if(live()==3072)compress();
            int count=std::min<int>({512,3072-live(),int(input.size()-offset)});
            need(next+count<=int(llama_model_n_ctx_train(e.model)),"absolute position limit reached; start a new session");
            std::vector<llama_token> part(input.begin()+offset,input.begin()+offset+count);
            long decoded=e.a.decoded_tokens;e.a.gating=true;e.decode(part,next,0);e.a.gating=false;
            need(e.a.decoded_tokens-decoded==count,"unexpected old-token prefill");
            for(int l=0;l<36;++l)for(int h=0;h<8;++h){
                for(int i=0;i<count;++i)positions[l][h].push_back(next+i);
                need(positions[l][h].size()==e.a.scores[l][h].size(),"score alignment after append");
            }
            next+=count;offset+=count;
        }
    }
    json infer(const json& input){
        need(!pending,"transaction pending");begin();cut_log=json::array();
        auto start=Clock::now();int before=live();long old_calls=e.a.decode_calls;
        auto prompt=e.tok(input.at("prompt").get<std::string>());
        need(!prompt.empty()&&prompt.size()<=8192,"input delta must contain 1-8192 tokens");
        append(prompt);double prefill_ms=ms(start);
        const auto schema=common_json::parse(input.at("schema_json").get<std::string>());
        auto grammar=json_schema_to_grammar(schema,true);
        auto* sampler=llama_sampler_chain_init(llama_sampler_chain_default_params());
        need(sampler,"sampler create");
        auto* grammar_sampler=llama_sampler_init_grammar(e.vocab,grammar.c_str(),"root");
        if(!grammar_sampler){llama_sampler_free(sampler);throw std::runtime_error("grammar rejected");}
        llama_sampler_chain_add(sampler,grammar_sampler);llama_sampler_chain_add(sampler,llama_sampler_init_greedy());
        std::string content;int generated=0;bool stopped=false;double first_ms=0;
        try{
            const int limit=input.value("limit",384);need(limit>0&&limit<=640,"output limit");
            for(int i=0;i<limit;++i){
                // The next token is sampled from the last forward result. If
                // full, prune before decoding that NEW token; never re-decode
                // a historical token merely to refresh logits after packing.
                auto token=llama_sampler_sample(sampler,e.ctx,-1);
                if(i==0)first_ms=ms(start);
                if(llama_vocab_is_eog(e.vocab,token)){stopped=true;break;}
                content+=e.piece(token);++generated;append({token});
            }
            llama_sampler_free(sampler);sampler=nullptr;
            if(stopped)append(e.tok("<|im_end|>\n"));
        }catch(...){if(sampler)llama_sampler_free(sampler);throw;}
        auto result=status();result.update({{"content",content},{"finish_reason",stopped?"stop":"length"},
            {"input_tokens",prompt.size()},{"output_tokens",generated},{"cache_tokens_before",before},
            {"prefill_ms",prefill_ms},{"first_token_ms",first_ms},{"total_ms",ms(start)},
            {"decode_calls",e.a.decode_calls-old_calls},{"cuts",cut_log}});
        return result;
    }
    void export_state(const std::string& kv,const std::string& meta){
        need(live()>0,"empty cache cannot be checkpointed");auto bytes=e.save(0);
        std::ofstream f(kv,std::ios::binary);f.write((char*)bytes.data(),bytes.size());f.close();need(f.good(),"KV write failed");
        json header=status();header["transaction_pending"]=false;header["positions"]=positions;header["scores"]=e.a.scores;
        write_report(meta,header);
    }
    void load_state(const std::string& kv,const std::string& meta){
        need(!pending,"transaction pending");json header;std::ifstream in(meta);in>>header;
        auto p=header.at("positions").get<Selection>();auto s=header.at("scores").get<decltype(e.a.scores)>();
        int count=header.at("physical_tokens");int pos=header.at("next_absolute_position");
        need(count>0&&count<=3072&&pos>0,"checkpoint bounds");
        for(int l=0;l<36;++l)for(int h=0;h<8;++h){
            need(p[l][h].size()==size_t(count)&&s[l][h].size()==size_t(count),"checkpoint ledger size");
            need(std::is_sorted(p[l][h].begin(),p[l][h].end()),"checkpoint order");
            need(std::adjacent_find(p[l][h].begin(),p[l][h].end())==p[l][h].end(),"duplicate checkpoint position");
            need(p[l][h].front()>=0&&p[l][h].back()<pos,"checkpoint positions");
            for(float value:s[l][h])need(std::isfinite(value),"invalid checkpoint score");
        }
        std::ifstream file(kv,std::ios::binary);std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),{});
        need(inspect(data).positions==p[0][0],"checkpoint representative mismatch");e.load(data,0);
        need(same_kv_and_positions(data,e.save(0)),"checkpoint restore mismatch");
        positions=std::move(p);e.a.scores=std::move(s);next=pos;cuts=header.at("total_cuts");cut_log=json::array();
    }
};
int main(int argc,char** argv){
    try{
        need(argc==3,"expected model root and gate socket");Online online;online.e.open(argv[1]);
        online.e.a.fd=socket(AF_UNIX,SOCK_STREAM,0);sockaddr_un addr{};addr.sun_family=AF_UNIX;
        need(strlen(argv[2])<sizeof addr.sun_path,"gate socket path");strcpy(addr.sun_path,argv[2]);
        need(connect(online.e.a.fd,(sockaddr*)&addr,sizeof addr)==0,"gate connect");
        std::cout<<json({{"ok",true},{"ready",true}}).dump()<<std::endl;
        std::string line;
        while(std::getline(std::cin,line)){
            try{
                auto request=json::parse(line);auto op=request.at("op").get<std::string>();json result;
                if(op=="inspect")result=online.status();
                else if(op=="reset"){online.clear();result=online.status();}
                else if(op=="infer")result=online.infer(request);
                else if(op=="compress"){online.begin();online.cut_log=json::array();online.compress();result=online.status();result["cuts"]=online.cut_log;}
                else if(op=="rollback"){online.rollback();result=online.status();}
                else if(op=="commit"){need(bool(online.pending),"no pending transaction");online.pending.reset();result=online.status();}
                else if(op=="export"){online.export_state(request.at("kv_path"),request.at("meta_path"));result=online.status();}
                else if(op=="load"){online.load_state(request.at("kv_path"),request.at("meta_path"));result=online.status();}
                else if(op=="shutdown")break;
                else throw std::runtime_error("unknown operation");
                std::cout<<json({{"ok",true},{"result",result}}).dump()<<std::endl;
            }catch(const std::exception& ex){std::cout<<json({{"ok",false},{"error",ex.what()}}).dump()<<std::endl;}
        }
        int32_t stop[3]={-1,0,0};transfer(online.e.a.fd,stop,sizeof stop,true);close(online.e.a.fd);online.e.a.fd=-1;
    }catch(const std::exception& ex){std::cerr<<ex.what()<<std::endl;return 1;}
}
