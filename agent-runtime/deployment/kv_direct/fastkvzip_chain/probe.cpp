#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "json.hpp"
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
    auto info=inspect(initial);need(info.count==3072,"packing initial count");
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
            for(int j=0;j<keep;++j){int pos=selected[layer][h][j];need(pos>=0&&pos<3072,"absolute head position");
                std::memcpy(out.data()+base+j*width+h*256,initial.data()+p+pos*width+h*256,256);}}
        p+=3072*width;
    }
    need(p==initial.size(),"packing consumed state");need(inspect(out).positions.back()==positions[0][0].back(),"next absolute position invariant");return out;
}


static bool has(const std::string& s,const std::string& word){return s.find(word)!=std::string::npos;}
static std::string normalized(std::string s){s.erase(std::remove_if(s.begin(),s.end(),[](char c){return c==' '||c=='\n'||c=='\r';}),s.end());return s;}
static json oracle(int round){
    return {{"pressure",round>=5?"47kPa":round>=1?"35kPa":"24kPa"},
        {"code","JN-2706"},{"status",round>=7?"尚未完成":round>=3?"已完成":"尚未完成"},
        {"file","不需要重写"},{"next",round>=8?"先校准阀门，再复核报告":round>=2?"校验传感器":"复核报告"}};
}
static bool fact_pass(const std::string& id,std::string text,const json& truth){
    text=normalized(text);
    if(id=="pressure")return has(text,truth[id].get<std::string>());
    if(id=="code")return has(text,"JN-2706");
    if(id=="status"){
        bool negative=has(text,"未完成")||has(text,"没完成");
        return truth[id]=="尚未完成"?negative:(has(text,"完成")&&!negative);
    }
    if(id=="file")return has(text,"不需要")||has(text,"无需")||has(text,"不用");
    if(truth[id]=="先校准阀门，再复核报告")return has(text,"校准阀门")&&has(text,"复核报告")&&text.find("校准阀门")<text.find("复核报告");
    return has(text,truth[id].get<std::string>());
}
static const std::vector<std::pair<std::string,std::string>> questions={
    {"pressure","BL-903当前生效压力是多少？只回答数值和单位。"},
    {"code","BL-903的工艺代码是什么？只回答代码。"},
    {"status","BL-903当前这一轮复测完成了吗？简短回答。"},
    {"file","checkpoint.txt需要重新写入吗？简短回答。"},
    {"next","BL-903下一步应该做什么？简短回答。"}};
static std::vector<llama_token> question_tokens(Engine& e,const std::string& q){
    return e.tok("<|im_start|>user\n"+q+"<|im_end|>\n<|im_start|>assistant\n");
}
static std::vector<std::pair<int,std::string>> updates(int round){
    if(round==0)return {{256,"主任务BL-903当前生效压力为24kPa。"},
        {768,"主任务BL-903工艺代码确认为JN-2706，此代码不是旁支批次代码。"},
        {1300,"工具结果：checkpoint.txt已成功写入并核验，不要重复执行该写入。"},
        {1900,"主任务BL-903本轮复测尚未完成。"},
        {2300,"主任务BL-903下一步是复核报告。"}};
    if(round==1)return {{300,"正式更正：BL-903生效压力现在为35kPa，旧24kPa和31kPa均作废。"}};
    if(round==2)return {{900,"用户变更BL-903下一步：先校验传感器，原下一步复核报告暂缓。"}};
    if(round==3)return {{1200,"BL-903工具复测结果：第一轮复测已完成，结果已核验。"}};
    if(round==4)return {{1400,"旁支GX-204的复测尚未完成，此状态只属于GX-204。"}};
    if(round==5)return {{300,"正式更正：BL-903生效压力改为47kPa，35kPa作废。"},
        {1500,"讨论草案：计划将BL-903改为52kPa；该方案尚未批准，当前不生效。"}};
    if(round==6)return {{1000,"工具错误：archive.json写入失败，没有生成archive.json；不能登记为完成。"}};
    if(round==7)return {{1400,"BL-903第二轮复测已经启动，第二轮尚未完成；此前完成的是第一轮。"}};
    if(round==8)return {{1600,"BL-903下一步变更为先校准阀门，再复核报告；不能先签字。"}};
    return {{900,"旁支BL-904生效压力88kPa，工艺代码ZQ-3145，复测完成；此记录只属于BL-904。"}};
}
static std::vector<llama_token> build_chunk(Engine& e,int round,int count,int absolute_start,json& anchors){
    std::vector<llama_token> out;
    auto add=[&](const std::string& text){auto t=e.tok(text);out.insert(out.end(),t.begin(),t.end());};
    if(round==0){
        add("<|im_start|>system\n按记录回答。新更正覆盖旧值；计划不等于完成；失败不等于成功；不知道就说不知道。<|im_end|>\n");
        need(out.size()<=64,"system prompt exceeds protected prefix");
    }
    add("<|im_start|>user\n主任务BL-903新增归档。以下旁支记录各自独立。\n");
    const auto pad=e.tok(" x");need(!pad.empty(),"pad token");int noise=0;
    auto fill=[&](int target){
        need(int(out.size())<=target,"event overlap");
        while(int(out.size())<target){
            int number=1000+round*100+noise++;
            auto line=e.tok("\n旁支Z"+std::to_string(number)+"：工艺码AB-"+std::to_string(number+71)+"，压力"+std::to_string(10+number%81)+"kPa，登记已完成，下一步签收；仅属于该旁支。\n");
            if(out.size()+line.size()<=size_t(target))out.insert(out.end(),line.begin(),line.end());
            else while(int(out.size())<target)out.push_back(pad[0]);
        }
    };
    for(auto [offset,text]:updates(round)){
        fill(offset);int start=absolute_start+out.size();add("\n"+text+"\n");
        anchors.push_back({{"absolute_start",start},{"absolute_end",absolute_start+int(out.size())},{"text",text}});
    }
    auto footer=e.tok("\n本段归档结束，普通归档标记不改变主任务事实。<|im_end|>\n<|im_start|>assistant\n已记录。<|im_end|>\n");
    fill(count-footer.size());out.insert(out.end(),footer.begin(),footer.end());
    need(out.size()==size_t(count),"chunk token budget mismatch");return out;
}
static void ledger_file(const std::string& file,int next_pos,const Selection& positions,const decltype(Audit{}.scores)& scores){
    std::ofstream f(file,std::ios::binary);const char magic[8]={'L','E','A','K','V','1','P','1'};f.write(magic,8);
    uint32_t l=36,h=8,n=614;uint64_t p=next_pos;
    f.write((char*)&l,4);f.write((char*)&h,4);f.write((char*)&n,4);f.write((char*)&p,8);
    for(int i=0;i<36;++i)for(int j=0;j<8;++j){
        need(positions[i][j].size()==614&&scores[i][j].size()==614,"ledger size");
        f.write((char*)positions[i][j].data(),614*4);f.write((char*)scores[i][j].data(),614*4);
    }
    need(f.good(),"ledger write failed");
}

int main(int argc,char** argv){
    std::string root=argv[1],path=argv[2],socketpath=argv[3];
    std::string work=path.substr(0,path.find_last_of('/'));
    json report={{"completed",false},{"scope","isolated ten-handover append-only CPU chain; probes rolled back; not production"}};
    try{
        if(argc>4 && std::string(argv[4])=="control"){
            json input;std::ifstream source(argv[5]);source>>input;
            Engine control;control.open(root,32768,1);
            auto history=input["tokens"].get<std::vector<llama_token>>();
            auto start=Clock::now();control.decode(history,0,0);report["prefill_ms"]=ms(start);auto state=control.save(0);
            report["history_tokens"]=history.size();report["source_window"]=input["source_window"];report["oracle"]=input["oracle"];
            report["scope"]="full uncompressed transcript control at first semantic failure";
            for(const auto& [id,q]:questions){
                control.load(state,0);auto a=control.answer(question_tokens(control,q),history.size(),0,32);a.erase("first_logits");
                bool pass=fact_pass(id,a["text"],input["oracle"]);
                report["questions"].push_back({{"id",id},{"answer",a},{"target_pass",pass}});
            }
            report["completed"]=true;write_report(path,report);return 0;
        }
        Engine e;e.open(root);
        e.a.fd=socket(AF_UNIX,SOCK_STREAM,0);need(e.a.fd>=0,"socket create");sockaddr_un addr{};addr.sun_family=AF_UNIX;
        need(socketpath.size()<sizeof addr.sun_path,"socket path");std::strcpy(addr.sun_path,socketpath.c_str());
        need(connect(e.a.fd,(sockaddr*)&addr,sizeof addr)==0,"gate connect");
        report["configuration"]={{"model","Qwen3-4B-Instruct-2507 Q4_K_M"},{"adapter","concise-v2 scale1"},
            {"windows",10},{"physical_before_cut",3072},{"retained_per_head",614},{"protected_prefix",64},{"protected_suffix",128},
            {"context_allocation",8192},{"threads",6},{"batch_threads",16},{"gate_threads",1},
            {"probe_policy","restore KV and gate scores after every probe; never replay probe answers into main chain"}};
        report["windows"]=json::array();report["first_semantic_failure_window"]=nullptr;
        std::vector<llama_token> transcript;Selection positions;std::vector<uint8_t> previous;
        int next_pos=0;
        auto combined=question_tokens(e,"BL-903当前生效压力、工艺代码、本轮复测状态、checkpoint.txt是否需要重写、下一步分别是什么？简短回答。");
        for(int round=0;round<10;++round){
            json row={{"window",round+1},{"oracle",oracle(round)}};
            if(round>0){need(same_kv_and_positions(e.save(0),previous),"next window did not inherit prior packed KV");row["inherited_previous_kv_bitwise"]=true;}
            int old=e.a.scores[0][0].size();need(old==(round?614:0),"live score count before append");
            row["source_anchors"]=json::array();auto chunk=build_chunk(e,round,3072-old,next_pos,row["source_anchors"]);
            row["new_input_tokens"]=chunk.size();row["new_absolute_start"]=next_pos;
            transcript.insert(transcript.end(),chunk.begin(),chunk.end());
            long before_tokens=e.a.decoded_tokens;double gate_before=e.a.gate_ms;
            e.a.gating=true;auto start=Clock::now();e.decode(chunk,next_pos,0);e.a.gating=false;
            row["append_ms"]=ms(start);row["append_gate_roundtrip_ms"]=e.a.gate_ms-gate_before;
            row["append_decoded_tokens"]=e.a.decoded_tokens-before_tokens;
            need(row["append_decoded_tokens"]==row["new_input_tokens"],"old tokens recomputed during append");
            for(int l=0;l<36;++l)for(int h=0;h<8;++h){
                for(size_t i=0;i<chunk.size();++i)positions[l][h].push_back(next_pos+i);
                need(positions[l][h].size()==3072 && e.a.scores[l][h].size()==3072,"physical ledger alignment");
            }
            next_pos+=chunk.size();row["next_absolute_position"]=next_pos;
            long calls=e.a.decode_calls,tokens=e.a.decoded_tokens,attention=e.a.fa_callbacks;
            start=Clock::now();auto before=e.save(0);double backup_ms=ms(start);
            need(inspect(before).positions==positions[0][0],"native cell metadata does not match representative ledger");
            start=Clock::now();e.a.selected=Selection{};
            for(int l=0;l<36;++l)for(int h=0;h<8;++h){
                auto& scores=e.a.scores[l][h];std::vector<int> middle(2880);std::iota(middle.begin(),middle.end(),64);
                for(float f:scores)need(std::isfinite(f),"invalid gate score");
                std::stable_sort(middle.begin(),middle.end(),[&](int a,int b){return scores[a]>scores[b];});middle.resize(422);
                auto& v=e.a.selected[l][h];for(int i=0;i<64;++i)v.push_back(i);v.insert(v.end(),middle.begin(),middle.end());
                for(int i=2944;i<3072;++i)v.push_back(i);std::sort(v.begin(),v.end());
            }
            double selection_ms=ms(start);start=Clock::now();auto packed=pack_heads(before,e.a.selected,positions);double packing_ms=ms(start);
            start=Clock::now();e.load(packed,0);double import_ms=ms(start);
            row["handover"]={{"backup_ms",backup_ms},{"selection_ms",selection_ms},{"pack_ms",packing_ms},{"import_ms",import_ms},
                {"total_ms",backup_ms+selection_ms+packing_ms+import_ms},{"before_state_bytes",before.size()},{"after_state_bytes",packed.size()},
                {"decode_calls",e.a.decode_calls-calls},{"decoded_tokens",e.a.decoded_tokens-tokens},{"attention_callbacks",e.a.fa_callbacks-attention}};
            need(e.a.decode_calls==calls && e.a.decoded_tokens==tokens && e.a.fa_callbacks==attention,"model execution during handover");
            need(same_kv_and_positions(packed,e.save(0)),"retained KV import mismatch");row["retained_kv_bitwise"]=true;
            auto new_scores=e.a.scores;Selection new_positions;
            for(int l=0;l<36;++l)for(int h=0;h<8;++h){
                new_scores[l][h].clear();
                for(int i:e.a.selected[l][h]){new_scores[l][h].push_back(e.a.scores[l][h][i]);new_positions[l][h].push_back(positions[l][h][i]);}
                need(new_positions[l][h].size()==614,"retained positions count");
                for(int i=0;i<64;++i)need(new_positions[l][h][i]==i,"protected original prefix changed");
                need(new_positions[l][h].back()==next_pos-1,"latest absolute position lost");
            }
            std::string ledger="ledger-"+std::to_string(round+1)+".bin";
            ledger_file(work+"/"+ledger,next_pos,new_positions,new_scores);row["ledger_file"]=ledger;
            // Compare first-step attention semantics without committing probes.
            e.load(before,0);e.a.masking=true;auto masked=e.answer(combined,next_pos,0,1);e.a.masking=false;
            e.load(packed,0);e.a.scores=new_scores;e.a.gating=true;
            auto answer=e.answer(combined,next_pos,0,96);e.a.gating=false;
            double difference=logits_diff(masked,answer);need(difference<=1e-5,"masked/packed first logits differ");
            row["packing_first_logits_max_diff"]=difference;answer.erase("first_logits");row["combined_answer"]=answer;
            bool all_pass=true;
            for(const auto& [id,q]:questions){
                e.load(packed,0);e.a.scores=new_scores;e.a.gating=true;
                auto a=e.answer(question_tokens(e,q),next_pos,0,32);e.a.gating=false;a.erase("first_logits");
                bool pass=fact_pass(id,a["text"],row["oracle"]);all_pass=all_pass&&pass;
                row["questions"].push_back({{"id",id},{"question",q},{"answer",a},{"target_pass",pass}});
            }
            row["all_target_facts_pass"]=all_pass;
            if(!all_pass && report["first_semantic_failure_window"].is_null()){
                report["first_semantic_failure_window"]=round+1;
                json control={{"source_window",round+1},{"oracle",row["oracle"]},{"tokens",transcript}};
                write_report(work+"/control-input.json",control);
            }
            e.load(packed,0);e.a.scores=new_scores;positions=std::move(new_positions);previous=std::move(packed);
            need(same_kv_and_positions(e.save(0),previous),"probe rollback did not restore chain");
            row["probe_rollback_verified"]=true;report["windows"].push_back(row);write_report(path,report);
        }
        write_report(work+"/transcript-tokens.json",transcript);
        int32_t end[3]={-1,0,0};transfer(e.a.fd,end,sizeof end,true);close(e.a.fd);e.a.fd=-1;
        report["completed"]=true;write_report(path,report);std::cout<<"completed\n";
    }catch(const std::exception& e){report["error"]=e.what();write_report(path,report);std::cerr<<e.what()<<std::endl;return 1;}
}
