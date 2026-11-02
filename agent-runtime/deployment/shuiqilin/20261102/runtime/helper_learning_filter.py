"""Keep originals; spend extraction/training only on complete, teachable evidence."""
import helper_off_contract as C

VERSION='useful-evidence-v1'

def enabled(config):
    return config.get('memory_teacher',{}).get('learning_filter')==VERSION

def archive_reason(episode,config):
    if not enabled(config):return None
    if episode.get('state_aggregate'):
        return 'telemetry_statistics_remain_in_memory_not_weight_targets'
    payload=episode.get('input')
    if payload is None:return None  # Existing missing-source review handling.
    if len(C.required_current_results(payload))>3:
        return 'complete_operation_split_required_no_receipt_truncation'
    if sum(C.prior(row) for row in C.records(payload).values())>3:
        return 'complete_operation_split_required_no_condition_truncation'
    if episode.get('behavior'):return None
    # Match the actual supervised receipt boundary, not strings inside assistant claims.
    by=C.records(payload)
    for ref,anchor in episode.get('source_anchors',{}).items():
        if anchor.get('role') in ('tool','runtime') and anchor.get('kind') in (
                'tool_result','archive_read_result','retrieval_result','job_error'):
            body=C.clean(by[ref]['内容'],[])
            response='历史'+by[ref]['类型']+'：'+(body if isinstance(body,str) else C.canonical(body))
            if len(response)<=240:return None
    return 'no_complete_trainable_receipt_keep_original_only'

def replay_allowed(row,config):
    return not enabled(config) or row.get('origin',{}).get('proof',{}).get('gold_type')!='deterministic_historical_sampling'
