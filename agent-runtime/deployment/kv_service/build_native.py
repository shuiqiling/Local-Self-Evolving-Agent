"""Build the online worker from the audited append-only KV primitives."""
from pathlib import Path

here=Path(__file__).resolve().parent
source=(here.parent/'kv_direct/fastkvzip_chain/probe.cpp').read_text(encoding='utf-8')
source=source[:source.index('static bool has(')]
source=source.replace('#include "json.hpp"','#include "json.hpp"\n#include "json-schema-to-grammar.h"\n#include <filesystem>\n#include <memory>')
start=source.index('static std::vector<uint8_t> pack_heads(')
head=source[:start]
pack=source[start:]
changes={
 'need(info.count==3072,"packing initial count");':'const int count=info.count;need(count>614 && count<=3072,"packing count");',
 'pos<3072':'pos<count',
 'p+=3072*width;':'p+=count*width;',
}
for old,new in changes.items():
 assert pack.count(old)==1,old
 pack=pack.replace(old,new)
# There is no diagnostic mask in the online worker. Keep the known callback,
# but never enable its mask flag; per-head physical packing is unchanged.
worker=r'''
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
'''
(here/'native_worker.cpp').write_text(head+pack+worker,encoding='utf-8')
print(here/'native_worker.cpp')
