"""Read-only Linux state sampling. Network rates are observed traffic, not capacity."""
from datetime import datetime,timezone
from pathlib import Path
import os
import shutil
import time
from zoneinfo import ZoneInfo


def cpu_counters(proc):
    numbers=[int(v) for v in (proc/'stat').read_text().splitlines()[0].split()[1:]]
    # guest/guest_nice are already accounted for in user/nice.
    return sum(numbers[:8]),sum(numbers[3:5])


def network_counters(proc):
    interfaces={}
    for line in (proc/'net/dev').read_text().splitlines()[2:]:
        if ':' not in line:continue
        name,data=line.split(':',1);parts=data.split()
        if len(parts)>=9:interfaces[name.strip()]={'rx_bytes':int(parts[0]),'tx_bytes':int(parts[8])}
    return interfaces


def deltas(cpu_before,cpu_after,net_before,net_after,seconds):
    total=cpu_after[0]-cpu_before[0];idle=cpu_after[1]-cpu_before[1]
    cpu=None if total<=0 else round(max(0,min(100,100*(total-idle)/total)),2)
    rates={}
    for name,after in net_after.items():
        if name not in net_before:continue
        before=net_before[name];rx=after['rx_bytes']-before['rx_bytes'];tx=after['tx_bytes']-before['tx_bytes']
        rates[name]={'rx_bytes_per_second':round(rx/seconds,2) if rx>=0 else None,
                     'tx_bytes_per_second':round(tx/seconds,2) if tx>=0 else None}
    return cpu,rates


def temperatures(sys_root):
    values=[]
    for p in [*(sys_root/'class/thermal').glob('thermal_zone*/temp'),*(sys_root/'class/hwmon').glob('hwmon*/temp*_input')]:
        try:
            value=float(p.read_text())/1000
            if not 0<value<130:continue
            label=p.parent/'type' if p.name=='temp' else p.with_name(p.stem.replace('_input','_label'))
            label_text=label.read_text().strip() if label.exists() else p.parent.name
            device_file=p.parent/'name'
            device=device_file.read_text().strip() if device_file.exists() else label_text
            is_cpu=any(t in label_text.lower() for t in ('package','core','cpu','pkg','tctl','tdie'))
            values.append({'sensor':str(p),'label':label_text,'device':device,'cpu_sensor':is_cpu,'celsius':round(value,1)})
        except (OSError,ValueError):pass
    return values


def collect(root,proc=Path('/proc'),sys_root=Path('/sys'),interval=1.,sleeper=time.sleep,clock=time.monotonic):
    cpu0=cpu_counters(proc);net0=network_counters(proc);tick=clock();sleeper(interval)
    duration=max(.001,clock()-tick)
    cpu,rates=deltas(cpu0,cpu_counters(proc),net0,network_counters(proc),duration)
    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in (proc/'meminfo').read_text().splitlines() if ':' in line}
    total=memory['MemTotal'];available=memory.get('MemAvailable',memory.get('MemFree',0));used=total-available
    physical=[name for name in rates if (sys_root/'class/net'/name/'device').exists() and name!='lo']
    selected=physical or [name for name in rates if name!='lo' and not name.startswith(('tailscale','docker','veth','tun','wg','proton','br-'))]
    rx=sum(rates[n]['rx_bytes_per_second'] or 0 for n in selected);tx=sum(rates[n]['tx_bytes_per_second'] or 0 for n in selected)
    sensors=temperatures(sys_root)
    # CPU-labelled package/core sensors preferred over unrelated device temperatures.
    cpu_sensors=[v for v in sensors if any(t in v['label'].lower() for t in ('package','core','cpu','pkg','tctl','tdie'))]
    disk=shutil.disk_usage(root);now=datetime.now(timezone.utc)
    return {'collected_at_utc':now.isoformat(),'local_time':now.astimezone(ZoneInfo('Asia/Shanghai')).isoformat(),
        'timezone':'Asia/Shanghai','sample_seconds':round(duration,3),'uptime_seconds':float((proc/'uptime').read_text().split()[0]),
        'cpu':{'usage_percent':cpu,'logical_cpus':os.cpu_count(),'load_average':list(os.getloadavg()),
            'temperature_c':max((v['celsius'] for v in cpu_sensors),default=None),'temperature_sensors':sensors},
        'memory':{'total_bytes':total,'available_bytes':available,'used_bytes':used,'used_percent':round(100*used/total,2),
            'swap_total_bytes':memory.get('SwapTotal',0),'swap_used_bytes':memory.get('SwapTotal',0)-memory.get('SwapFree',0)},
        'network':{'measurement':'observed_interface_traffic_not_internet_capacity','aggregate_interfaces':selected,
            'rx_mib_per_second':round(rx/1024**2,4),'tx_mib_per_second':round(tx/1024**2,4),'interfaces':rates},
        'disk':{'path':str(root),'total_bytes':disk.total,'free_bytes':disk.free,'used_percent':round(100*disk.used/disk.total,2)}}
