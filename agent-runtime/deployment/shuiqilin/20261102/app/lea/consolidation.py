"""Evidence-first Memory/Skill/Weight routing for the project's Wake/Sleep plan.

Numbers are conservative prototype policy defaults, not measured optimal values.
An isolated successful task never qualifies for parameter consolidation.
"""
from dataclasses import asdict,dataclass
from datetime import datetime,timezone,timedelta
import hashlib
import json
import math
from pathlib import Path
import time

from .store import atomic_json


@dataclass(frozen=True)
class ConsolidationPolicy:
    min_support:int=5
    min_scopes:int=3
    min_success_rate:float=.9
    min_confidence_lower:float=.7
    max_counterexample_rate:float=.1
    min_idle_seconds:int=60
    min_available_gib:float=24
    max_normalized_load:float=.35
    max_new_rows:int=24
    replay_fraction:float=.5
    max_updates_per_day:int=1
    def __post_init__(self):
        if self.min_support<2 or self.min_scopes<2:raise ValueError('Consolidation requires repeated cross-scope evidence')
        if not 0<self.replay_fraction<1:raise ValueError('Replay fraction must be between0and1')
        if self.max_new_rows<self.min_support:raise ValueError('Batch must cover the selected episodes')


def lower_bound(successes,total):
    if not total:return 0.
    z=1.96;p=successes/total
    return (p+z*z/(2*total)-z*math.sqrt(p*(1-p)/total+z*z/(4*total*total)))/(1+z*z/total)


def assess(observations,policy=None,probe=None):
    """Source IDs and checks are validated by record_episode before persistence."""
    policy=policy or ConsolidationPolicy();unique={}
    for row in observations:
        old=unique.get(row['fingerprint'])
        is_counter=row['critical_violation'] or row.get('failed_outcome',False) or (row['independent_checked'] and not row['independently_verified'])
        if old is None or is_counter:unique[row['fingerprint']]=row
    rows=list(unique.values());support=[r for r in rows if r['independently_verified'] and r['policy_followed']
        and not r['critical_violation'] and not r.get('failed_outcome',False)]
    counter=[r for r in rows if r['critical_violation'] or r.get('failed_outcome',False) or (r['independent_checked'] and not r['independently_verified'])]
    scopes=sorted({r['scope'] for r in support if r['scope']!='unclassified'})
    n=len(support)+len(counter);rate=len(support)/n if n else 0.;counter_rate=len(counter)/n if n else 0.
    bound=lower_bound(len(support),n)
    reasons=[]
    if len(support)<policy.min_support:reasons.append('insufficient_independent_repetitions')
    if len(scopes)<policy.min_scopes:reasons.append('insufficient_cross_scope_validation')
    if rate<policy.min_success_rate:reasons.append('success_trend_not_stable')
    if bound<policy.min_confidence_lower:reasons.append('confidence_lower_bound_too_low')
    if counter_rate>policy.max_counterexample_rate:reasons.append('unresolved_counterexamples')
    # The paired probe measures whether retrieval actually helps on frozen
    # decisions. If baseline is already correct, there is no demonstrated need
    # to spend parameter capacity on this workflow.
    probe_valid=bool(probe and probe.get('provenance_verified'))
    gain=(probe.get('with_memory_passed',0)-probe.get('baseline_passed',0)) if probe_valid else None
    if not probe_valid:reasons.append('parametric_value_not_measured')
    elif gain<=0:reasons.append('no_demonstrated_decision_gain')
    status='accept' if not reasons else 'delay'
    return dict(pattern_id='developer.memory_guided_workflow',route='weight',status=status,
        recommendation='任务要求复用开发经验时，核对并读取相关流程索引；依据当前目标选择查询、实现、测试或登记，保留失败条件与适用范围。',
        scope='Local pure-Python JSON-tool development only; not arbitrary code or robotics',
        support_count=len(support),counterexample_count=len(counter),observation_count=len(rows),
        support_episode_ids=[r['id'] for r in support],counterexample_episode_ids=[r['id'] for r in counter],
        scopes=scopes,success_rate=rate,confidence_lower_95=bound,counterexample_rate=counter_rate,
        confidence_note='Wilson lower bound assumes independent trials. Here it is a conservative gate heuristic, not a measured general accuracy guarantee.',
        expected_decision_gain=gain,reasons=reasons,
        exceptions=[{'episode':r['id'],'details':r.get('exception','')} for r in counter],
        deterministic_parts_route='skill',policy=asdict(policy))


class Consolidation:
    def __init__(self,dev,policy=None):
        self.dev=dev;self.policy=policy or ConsolidationPolicy()
        with dev.db() as db:db.executescript('''
            CREATE TABLE IF NOT EXISTS episode_observations(
              id TEXT PRIMARY KEY,fingerprint TEXT NOT NULL,body TEXT NOT NULL,created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS consolidation_runs(
              id TEXT PRIMARY KEY,kind TEXT,status TEXT,body TEXT,created TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS consolidation_consumed(episode TEXT PRIMARY KEY,learning_job TEXT NOT NULL);
        ''')
    def record_episode(self,identifier,spec,error=None):
        path=self.dev.home/'episodes'/identifier/'episode.json'
        if not path.exists():return None
        raw=path.read_bytes();episode=json.loads(raw);state=episode['state'];artifact=state.get('artifact')
        trace=episode.get('trace',[]);tests=state.get('test_report') or {};independent=tests.get('independent')
        independently_verified=bool(artifact and state.get('published') and tests.get('passed') and independent and independent.get('passed'))
        if artifact:
            if hashlib.sha256(Path(artifact['source']).read_bytes()).hexdigest()!=artifact['code_sha256']:
                raise ValueError('Episode artifact integrity mismatch')
        if independently_verified and tests.get('code_sha256')!=artifact['code_sha256']:
            raise ValueError('Tests do not certify the current artifact revision')
        for entry in trace:
            if entry.get('event_ref'):
                event=self.dev.store.resolve(state['task_id'],entry['event_ref'])
                if event['body'].get('value')!=entry.get('result'):raise ValueError('Trace disagrees with immutable tool result')
        if independently_verified:
            event=self.dev.store.resolve(state['task_id'],episode['memory_ref'])
            if event['body'].get('artifact_sha256')!=artifact['code_sha256']:raise ValueError('Verified memory source mismatch')
        # A publish attempt rejected by the controller is a useful negative
        # example. Failed attempts remain available to the miner, not erased.
        violations=[r for r in trace if r.get('decision',{}).get('parsed',{}).get('tool')=='dev.publish'
                    and r.get('result',{}).get('status')=='error']
        read_steps=[r['step'] for r in trace if r.get('decision',{}).get('parsed',{}).get('tool')=='memory.read'
                    and r.get('result',{}).get('sha')]
        submit_steps=[r['step'] for r in trace if r.get('decision',{}).get('parsed',{}).get('tool')=='dev.submit']
        followed=bool(read_steps and submit_steps and min(read_steps)<max(submit_steps) and not violations)
        # Repeating the same request and hidden-test contract does not create
        # more support merely by changing job IDs or model random seeds.
        fingerprint=hashlib.sha256(json.dumps({'request':spec['request'],
            'acceptance':spec.get('acceptance_cases',[])},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        value=dict(id=identifier,fingerprint=fingerprint,scope=spec.get('domain','unclassified'),
            runtime_version=state.get('runtime_version','selfdev-context-v1'),
            independent_checked=independent is not None,independently_verified=independently_verified,
            policy_followed=followed,critical_violation=bool(violations),
            failed_outcome=bool(error or (tests and not tests.get('passed'))),
            exception=error or ('publish_attempt_without_passed_tests' if violations else ''),
            source_path=str(path),source_sha256=hashlib.sha256(raw).hexdigest(),
            task_id=state['task_id'],memory_ref=episode.get('memory_ref'),
            model_seconds=sum(r.get('decision',{}).get('seconds',0) for r in trace),steps=len(trace),
            failed_test_attempts=sum(r.get('decision',{}).get('parsed',{}).get('tool')=='dev.test'
                                     and not r.get('result',{}).get('passed',False) for r in trace),
            routes={'episode':'memory','tested_implementation':'skill' if state.get('published') else 'memory',
                    'repeated_decision_policy':'weight_candidate'},
            causal_claim=False)
        with self.dev.db() as db:
            db.execute('INSERT OR IGNORE INTO episode_observations VALUES(?,?,?,?)',
                (identifier,fingerprint,json.dumps(value,ensure_ascii=False),datetime.now(timezone.utc).isoformat()))
        return value
    def observations(self):
        with self.dev.db() as db:rows=[json.loads(r[0]) for r in db.execute('SELECT body FROM episode_observations ORDER BY created')]
        for row in rows:
            path=Path(row['source_path'])
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['source_sha256']:
                raise ValueError('Experience source changed; mining refused')
        return rows
    def probe_evidence(self):
        path=self.dev.home/'consolidation/paired-probe.json'
        if not path.exists():return None
        evidence=json.loads(path.read_text());source=Path(evidence['report'])
        if not source.resolve().is_relative_to((self.dev.home/'consolidation').resolve()):raise ValueError('Probe escaped evidence directory')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=evidence['sha256']:raise ValueError('Probe integrity mismatch')
        report=json.loads(source.read_text())
        if not report.get('completed') or report.get('purpose')!='memory_conditioned_pair':return None
        if report.get('model_sha256')!=self.dev.latest()['gguf_sha256']:return None
        if report.get('runtime_version')!=self.dev.runtime_version:return None
        return {**report,'provenance_verified':True}
    def mine(self):
        all_rows=self.observations()
        rows=[r for r in all_rows if r.get('runtime_version','selfdev-context-v1')==self.dev.runtime_version]
        report=assess(rows,self.policy,self.probe_evidence())
        report['runtime_version']=self.dev.runtime_version
        report['other_runtime_observations']=[{'id':r['id'],'runtime_version':r.get('runtime_version','selfdev-context-v1'),
                                             'failed_outcome':r.get('failed_outcome',False),'exception':r.get('exception','')}
                                            for r in all_rows if r not in rows]
        report['routing_note']='Facts and raw episodes stay in memory; executable implementations are registered skills. Only repeated, measured policy gains are weight candidates.'
        report['miner']='structural developer-trace miner v1; does not claim general semantic clustering'
        atomic_json(self.dev.home/'consolidation/latest-mining.json',report)
        return report
    def sleep_ready(self,now_value=None):
        now_value=time.time() if now_value is None else now_value
        with self.dev.db() as db:
            busy=db.execute("SELECT COUNT(*) FROM jobs WHERE kind IN ('develop','preview') AND status IN ('queued','running')").fetchone()[0]
            row=db.execute("SELECT MAX(created) FROM jobs WHERE kind IN ('develop','preview')").fetchone()
        with self.dev.store.db() as db:
            busy+=db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('running','queued')").fetchone()[0]
            event=db.execute('SELECT MAX(created) FROM events').fetchone()[0]
        timestamps=[v for v in [row[0],event] if v]
        idle=now_value-max((datetime.fromisoformat(v).timestamp() for v in timestamps),default=now_value)
        available=0.;load=None
        try:
            import os
            values={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
            available=int(values['MemAvailable'].split()[0])/1024**2
            load=os.getloadavg()[0]/max(1,len(os.sched_getaffinity(0)))
        except (OSError,AttributeError):pass
        reasons=[]
        if busy:reasons.append('foreground_jobs_present')
        if idle<self.policy.min_idle_seconds:reasons.append('idle_window_not_reached')
        if available<self.policy.min_available_gib:reasons.append('insufficient_free_memory')
        if load is None or load>self.policy.max_normalized_load:reasons.append('system_not_low_load')
        return {'ready':not reasons,'reasons':reasons,'idle_seconds':max(0,idle),'available_gib':available,'normalized_load':load}
    def maybe_queue(self):
        report=self.mine()
        with self.dev.db() as db:consumed={r[0] for r in db.execute('SELECT episode FROM consolidation_consumed')}
        fresh=[identifier for identifier in report['support_episode_ids'] if identifier not in consumed]
        if len(fresh)<self.policy.min_support:return {'queued':False,'reason':'new_evidence_threshold','gate':report}
        structural=[r for r in report['reasons'] if r!='parametric_value_not_measured']
        if not structural and 'parametric_value_not_measured' in report['reasons']:
            sleep=self.sleep_ready()
            if sleep['ready']:
                model_sha=self.dev.latest()['gguf_sha256']
                with self.dev.db() as db:busy=db.execute("SELECT COUNT(*) FROM jobs WHERE kind='probe' AND json_extract(spec,'$.model_sha256')=?",(model_sha,)).fetchone()[0]
                if not busy:
                    job=self.dev.create({'pattern':report['pattern_id'],'gate':report,'model_sha256':model_sha},kind='probe')
                    return {'queued':True,'kind':'probe','job_id':job['id'],'gate':report}
        if report['status']!='accept':return {'queued':False,'gate':report}
        sleep=self.sleep_ready()
        if not sleep['ready']:return {'queued':False,'gate':report,'sleep':sleep}
        local=datetime.now(timezone(timedelta(hours=8)))
        start=local.replace(hour=0,minute=0,second=0,microsecond=0).astimezone(timezone.utc)
        end=start+timedelta(days=1)
        with self.dev.db() as db:
            count=db.execute("SELECT COUNT(*) FROM jobs WHERE kind='learn' AND created>=? AND created<?",(start.isoformat(),end.isoformat())).fetchone()[0]
            consumed={r[0] for r in db.execute('SELECT episode FROM consolidation_consumed')}
        pending=[identifier for identifier in report['support_episode_ids'] if identifier not in consumed]
        if count>=self.policy.max_updates_per_day or len(pending)<self.policy.min_support:
            return {'queued':False,'reason':'budget_or_new_evidence_threshold','gate':report}
        by_id={row['id']:row for row in self.observations()};selected=[];scopes=set()
        for identifier in reversed(pending):
            scope=by_id[identifier]['scope']
            if scope not in scopes:selected.append(identifier);scopes.add(scope)
            if len(scopes)>=self.policy.min_scopes:break
        if len(scopes)<self.policy.min_scopes:return {'queued':False,'reason':'new_batch_lacks_scope_diversity','gate':report}
        selected+=(list(reversed([identifier for identifier in pending if identifier not in selected])))[:self.policy.min_support-len(selected)]
        job=self.dev.create({'episodes':selected,'initial_adapter':self.dev.latest(),'gate':report},kind='learn')
        with self.dev.db() as db:
            for identifier in selected:db.execute('INSERT INTO consolidation_consumed VALUES(?,?)',(identifier,job['id']))
        return {'queued':True,'job_id':job['id'],'episodes':selected,'gate':report}
