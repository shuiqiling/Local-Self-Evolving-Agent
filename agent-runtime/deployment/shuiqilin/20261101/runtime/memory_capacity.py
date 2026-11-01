"""Capacity reporting for a dedicated finite-size memory filesystem."""
from pathlib import Path
import os


def capacity(options,*,statvfs=None,mounted=os.path.ismount):
    if not options or not options.get('enabled'):return {'enabled':False,'write_allowed':True,'optional_write_allowed':True}
    root=Path(options['volume_root']);limit=options['limit_bytes']
    if not mounted(root):return {'enabled':True,'mounted':False,'limit_bytes':limit,'write_allowed':False,
        'optional_write_allowed':False,'status':'memory_volume_unavailable'}
    stats=(statvfs or os.statvfs)(root);free=stats.f_bavail*stats.f_frsize;total=stats.f_blocks*stats.f_frsize
    used=max(0,limit-free);ratio=used/limit
    return {'enabled':True,'mounted':True,'hard_enforced':True,'limit_bytes':limit,'limit_gib':limit/1024**3,
        'filesystem_data_bytes':total,'available_bytes':free,'used_including_filesystem_bytes':used,
        'used_percent':round(ratio*100,2),'write_allowed':free>=options.get('reserve_bytes',32*1024**2),
        'optional_write_allowed':ratio<options.get('optional_stop_ratio',.98),
        'status':'near_limit' if ratio>=options.get('warning_ratio',.9) else 'available',
        'scope':options.get('scope',[]),'volume_root':str(root),'full_policy':'stop_new_writes_preserve_existing'}


def prepare_training_cache(run,options):
    """Place future archive copies/data/teacher notes inside the same hard cap."""
    if not options or not options.get('enabled'):return None
    run=Path(run);cache=Path(options['training_cache_root'])/run.name
    cache.mkdir(parents=True,exist_ok=False)
    for name in ('data','teacher'):
        target=cache/name;target.mkdir();(run/name).symlink_to(target,target_is_directory=True)
    return cache
