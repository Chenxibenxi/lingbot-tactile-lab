"""Observable simulator state audit; never a full PhysX state restoration."""
import math

REQUIRED_DIMENSIONS = {
    **{side+'.root.'+field: size for side in ('left', 'right')
       for field, size in (('position', 3), ('quaternion', 4))},
    **{name+'.'+field: size for name in ('bottle1', 'bottle2')
       for field, size in (('position', 3), ('quaternion', 4),
                           ('linear_velocity', 3), ('angular_velocity', 3))},
}


def _vector(value):
    values = value.tolist() if hasattr(value, 'tolist') else list(value)
    if not values or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError('missing or nonfinite simulator state')
    return values


def capture_state(task):
    # Import lazily: contract tests run without loading the simulator/GPU.
    from sapien.physx import PhysxRigidDynamicComponent
    values = {}
    def pose(name, value):
        values[name+'.position'] = _vector(value.p)
        values[name+'.quaternion'] = _vector(value.q)
    for side in ('left', 'right'):
        entity = getattr(task.robot, side+'_entity')
        pose(side+'.root', entity.get_root_pose())
        values[side+'.qpos'] = _vector(entity.get_qpos())
        values[side+'.qvel'] = _vector(entity.get_qvel())
        for index, joint in enumerate(entity.get_active_joints()):
            key = side+'.joint.'+str(index)+'.'+joint.get_name()
            values[key+'.drive_position'] = _vector(joint.get_drive_target())
            values[key+'.drive_velocity'] = _vector(joint.get_drive_velocity_target())
    for name in ('bottle1', 'bottle2'):
        actor = getattr(task, name).actor
        pose(name, actor.get_pose())
        bodies = [c for c in actor.get_components() if isinstance(c, PhysxRigidDynamicComponent)]
        if len(bodies) != 1:
            raise ValueError('expected exactly one dynamic bottle body')
        values[name+'.linear_velocity'] = _vector(bodies[0].get_linear_velocity())
        values[name+'.angular_velocity'] = _vector(bodies[0].get_angular_velocity())
    clock = task._force_probe.step
    if type(clock) is not int or clock < 0:
        raise ValueError('invalid physics clock')
    return dict(schema='recovery_observable_state_v2', physics_step=clock,
                values=values, training_approval=False)


def compare_states(reference, observed):
    """Exact observable-state gate; reports differences without accepting drift.

    RGB, source identity, action histories and same-run timing remain separate
    required audits. Hidden solver/contact caches are not represented here.
    """
    for state in (reference, observed):
        if (state.get('schema') != 'recovery_observable_state_v2'
                or state.get('training_approval') is not False
                or type(state.get('physics_step')) is not int or state['physics_step'] < 0
                or not isinstance(state.get('values'), dict) or not state['values']):
            raise ValueError('invalid observable state')
        for key, value in state['values'].items():
            if not isinstance(key, str) or not isinstance(value, list):
                raise ValueError('invalid state field')
            _vector(value)
        values = state['values']
        for key, size in REQUIRED_DIMENSIONS.items():
            if key not in values or len(values[key]) != size:
                raise ValueError('required state field missing or malformed: '+key)
        for side in ('left', 'right'):
            qpos, qvel = values.get(side+'.qpos'), values.get(side+'.qvel')
            if not qpos or not qvel or len(qpos) != len(qvel):
                raise ValueError('joint positions/velocities missing or inconsistent')
            drive = [k for k in values if k.startswith(side+'.joint.') and k.endswith('.drive_position')]
            if not drive or any(k[:-len('drive_position')]+'drive_velocity' not in values for k in drive):
                raise ValueError('joint drive state missing')
    a, b = reference['values'], observed['values']
    if set(a) != set(b):
        raise ValueError('state fields differ')
    deltas = {}
    for key in sorted(a):
        if len(a[key]) != len(b[key]):
            raise ValueError('state dimensions differ: '+key)
        deltas[key] = max(abs(x-y) for x,y in zip(a[key],b[key]))
    clock_equal = reference['physics_step'] == observed['physics_step']
    return dict(observable_state_equal=clock_equal and all(v == 0 for v in deltas.values()),
                clock_equal=clock_equal, max_absolute_delta_by_field=deltas,
                comparison='exact_no_tolerance', training_approval=False,
                limitation='Observable state only; excludes hidden PhysX solver/contact caches.')
