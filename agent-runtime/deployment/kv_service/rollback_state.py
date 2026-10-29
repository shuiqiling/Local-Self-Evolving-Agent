"""Reopen raw-record visibility without reverting newer messages or constraints."""
import json

def restore_legacy_cursors(store):
    restored=[]
    with store.db(True) as db:
        for row in db.execute('SELECT id,state FROM tasks').fetchall():
            state=json.loads(row['state'])
            if not state.get('native_kv'):continue
            task=store._task(db,row['id'])
            origin=state.get('legacy_state_before_native')
            if origin is None:
                # For an older experimental state with no origin, all original
                # events remain available; never manufacture a textual summary.
                origin=dict(summary='',completed=[],next_steps=[],recorded_through=0)
            origin=dict(origin);origin.pop('native_kv',None);origin.pop('legacy_state_before_native',None)
            version=task['version']+1
            db.execute('UPDATE tasks SET state=?,version=? WHERE id=?',(json.dumps(origin,ensure_ascii=False),version,task['id']))
            task=store._task(db,task['id'])
            db.execute('INSERT INTO state_versions VALUES(?,?,?)',(task['id'],version,json.dumps(task,ensure_ascii=False)))
            store._event(db,task,'runtime','native_kv_disabled',dict(recorded_through=origin.get('recorded_through',0),
                raw_events_preserved=True,constraints_preserved=True))
            restored.append(task['id'])
    for task_id in restored:store.export(task_id)
    return restored
