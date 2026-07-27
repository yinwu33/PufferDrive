import math

from pufferlib.atlantic.drive import binding


SPEED_THRESHOLD = 0.5
FRONT_CONTACT_COS_THRESHOLD = math.sqrt(0.5)


def classify(agent_speed, other_speed, normal_x, normal_y, agent_heading, other_heading):
    return binding.classify_collision_fault(
        agent_speed,
        other_speed,
        normal_x,
        normal_y,
        agent_heading,
        other_heading,
        SPEED_THRESHOLD,
        FRONT_CONTACT_COS_THRESHOLD,
    )


def test_stationary_ego_is_not_self_fault():
    fault = classify(0.1, 5.0, 1.0, 0.0, 0.0, math.pi)
    assert fault == binding.NON_SELF_FAULT


def test_moving_ego_front_contact_is_self_fault():
    fault = classify(5.0, 0.0, 1.0, 0.0, 0.0, math.pi)
    assert fault == binding.SELF_FAULT


def test_other_front_into_ego_side_is_not_self_fault():
    fault = classify(5.0, 5.0, 0.0, 1.0, 0.0, -math.pi / 2)
    assert fault == binding.NON_SELF_FAULT


def test_moving_ego_into_stationary_object_is_self_fault():
    fault = classify(5.0, 0.1, 0.0, 1.0, 0.0, 0.0)
    assert fault == binding.SELF_FAULT


def test_side_side_contact_is_ambiguous():
    fault = classify(5.0, 5.0, 0.0, 1.0, 0.0, 0.0)
    assert fault == binding.AMBIGUOUS_FAULT
