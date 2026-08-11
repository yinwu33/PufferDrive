import math

from pufferlib.atlantic.drive import binding


SPEED_THRESHOLD = 0.5
FRONT_CONTACT_COS_THRESHOLD = math.sqrt(0.5)
FLOAT32_TOLERANCE = 1e-6
GEOMETRY_TOLERANCE = 1e-4

CAR_LENGTH = 4.6
CAR_WIDTH = 2.0
HALF_LENGTH = CAR_LENGTH / 2.0
HALF_WIDTH = CAR_WIDTH / 2.0


def car(x, y, heading=0.0, vx=0.0, vy=0.0, length=CAR_LENGTH, width=CAR_WIDTH):
    # Fault attribution is geometric, so every participant is a full pose plus a
    # bounding box. The contact point is derived by the env's own code.
    return (x, y, heading, length, width, vx, vy)


def classify(agent, other):
    return binding.classify_collision_fault(
        agent,
        other,
        SPEED_THRESHOLD,
        FRONT_CONTACT_COS_THRESHOLD,
    )


def classify_pair(agent, other):
    return binding.classify_collision_pair(
        agent,
        other,
        SPEED_THRESHOLD,
        FRONT_CONTACT_COS_THRESHOLD,
    )


def settle_reward(
    self_fault,
    other_fault,
    *,
    stopped=False,
    already_rewarded=False,
    reward_once=True,
    self_factor=0.25,
    adversarial_factor=0.8,
):
    return binding.settle_adversarial_collision_reward(
        self_fault,
        other_fault,
        int(stopped),
        int(already_rewarded),
        int(reward_once),
        self_factor,
        adversarial_factor,
    )


# --- contact geometry -------------------------------------------------------


def test_contact_point_is_the_centroid_of_the_overlap_region():
    # Ego spans x in [-2.3, 2.3]; the other car spans x in [1.7, 6.3]. The overlap
    # is x in [1.7, 2.3] across the full 2 m width, centered at (2.0, 0).
    overlapped, x, y = binding.obb_contact_point(car(0.0, 0.0), car(4.0, 0.0))
    assert overlapped == 1
    assert math.isclose(x, 2.0, abs_tol=GEOMETRY_TOLERANCE)
    assert math.isclose(y, 0.0, abs_tol=GEOMETRY_TOLERANCE)


def test_contact_point_tracks_the_overlap_not_the_separating_axis():
    # Deep penetration: the other car's rear reaches x = -0.3, well past the point
    # where the minimum-translation axis flips from the length axis to the width
    # axis. The contact centroid stays on the ego's front half regardless.
    overlapped, x, y = binding.obb_contact_point(car(0.0, 0.0), car(2.0, 0.0, heading=math.pi))
    assert overlapped == 1
    assert math.isclose(x, 1.0, abs_tol=GEOMETRY_TOLERANCE)
    assert math.isclose(y, 0.0, abs_tol=GEOMETRY_TOLERANCE)


def test_contact_point_reports_no_overlap_for_separated_boxes():
    overlapped, _, _ = binding.obb_contact_point(car(0.0, 0.0), car(10.0, 0.0))
    assert overlapped == 0


def test_contact_point_handles_rotated_boxes():
    # Other car crosses the ego's +y flank nose first. Overlap is x in [-1, 1],
    # y in [0.7, 1.0].
    overlapped, x, y = binding.obb_contact_point(car(0.0, 0.0), car(0.0, 3.0, heading=-math.pi / 2))
    assert overlapped == 1
    assert math.isclose(x, 0.0, abs_tol=GEOMETRY_TOLERANCE)
    assert math.isclose(y, 0.85, abs_tol=GEOMETRY_TOLERANCE)


# --- fault attribution ------------------------------------------------------


def test_stationary_ego_is_not_self_fault():
    # Ego nearly stationary (speed <= threshold) is never at fault.
    fault = classify(car(0.0, 0.0, vx=0.1), car(4.0, 0.0, heading=math.pi, vx=-5.0))
    assert fault == binding.NON_SELF_FAULT


def test_moving_ego_front_contact_is_self_fault():
    # Ego drives forward into a stopped car; contact lands on the ego's front face.
    fault = classify(car(0.0, 0.0, vx=5.0), car(4.0, 0.0))
    assert fault == binding.SELF_FAULT


def test_other_front_into_ego_side_is_not_self_fault():
    # Ego moves +x; other drives its front face into the ego's +y flank.
    fault = classify(car(0.0, 0.0, vx=5.0), car(0.0, 3.0, heading=-math.pi / 2, vy=-5.0))
    assert fault == binding.NON_SELF_FAULT


def test_moving_ego_into_stationary_object_is_self_fault():
    # Contact is on the ego's flank, but the other car is parked, so the ego is
    # still the one that closed the gap.
    fault = classify(car(0.0, 0.0, vx=5.0), car(0.0, 1.8))
    assert fault == binding.SELF_FAULT


def test_side_side_contact_is_ambiguous():
    # Both moving +x, contact on the ego's +y flank: neither leading face.
    fault = classify(car(0.0, 0.0, vx=5.0), car(0.0, 1.8, vx=5.0))
    assert fault == binding.AMBIGUOUS_FAULT


def test_reversing_ego_into_other_is_self_fault():
    # Ego reverses into a parked car. The leading face follows the direction of
    # travel, so the rear face counts as leading and the reverse-into-others
    # exploit stays closed.
    fault = classify(car(0.0, 0.0, vx=-5.0), car(-4.0, 0.0))
    assert fault == binding.SELF_FAULT


def test_reversing_ego_hit_on_its_front_is_not_self_fault():
    # Ego backs away while another car drives into its front face. The contact is
    # on the ego's trailing face, so responsibility falls on the other car. A
    # velocity-direction test cannot tell this apart from the case above.
    fault = classify(car(0.0, 0.0, vx=-3.0), car(4.0, 0.0, heading=math.pi, vx=-5.0))
    assert fault == binding.NON_SELF_FAULT


def test_deep_rear_end_is_self_fault():
    # One 0.1 s step of 25 m/s closing speed buries the ego 2.5 m into the slower
    # car ahead, which is more than the 2 m lateral overlap. The minimum-translation
    # axis flips to the width axis here; the contact centroid does not.
    fault = classify(car(0.0, 0.0, vx=30.0), car(2.1, 0.0, vx=5.0))
    assert fault == binding.SELF_FAULT


def test_deep_rear_end_victim_is_not_at_fault():
    self_fault, other_fault = classify_pair(car(2.1, 0.0, vx=5.0), car(0.0, 0.0, vx=30.0))
    assert self_fault == binding.NON_SELF_FAULT
    assert other_fault == binding.SELF_FAULT


def test_offset_rear_end_is_self_fault():
    # Partially offset rear-end: 1.1 m of longitudinal penetration against only
    # 0.8 m of lateral overlap, so the separating axis is lateral even though the
    # ego plainly drove its front into the other car.
    fault = classify(car(0.0, 0.0, vx=15.0), car(3.5, 1.2, vx=5.0))
    assert fault == binding.SELF_FAULT


def test_front_corner_clip_from_behind_is_not_ego_fault():
    # Mirror image of the offset rear-end: the other car runs its front corner
    # into the ego's rear flank while the ego drives straight ahead.
    fault = classify(car(0.0, 0.0, vx=5.0), car(-3.5, 1.2, vx=15.0))
    assert fault == binding.NON_SELF_FAULT


def test_pair_classification_marks_counterparty_front_impact_as_other_fault():
    self_fault, other_fault = classify_pair(
        car(0.0, 0.0, vx=5.0), car(0.0, 3.0, heading=-math.pi / 2, vy=-5.0)
    )
    assert self_fault == binding.NON_SELF_FAULT
    assert other_fault == binding.SELF_FAULT


def test_pair_classification_head_on_marks_both_self_fault():
    self_fault, other_fault = classify_pair(car(0.0, 0.0, vx=5.0), car(4.0, 0.0, heading=math.pi, vx=-5.0))
    assert self_fault == binding.SELF_FAULT
    assert other_fault == binding.SELF_FAULT


def test_pair_classification_deep_head_on_marks_both_self_fault():
    # Closing at 30 m/s the boxes overlap by 2.6 m on the first frame contact is
    # detected, past the point where the separating axis turns lateral. Both cars
    # drove their front faces into the collision and both must be penalized.
    self_fault, other_fault = classify_pair(
        car(0.0, 0.0, vx=15.0), car(2.0, 0.0, heading=math.pi, vx=-15.0)
    )
    assert self_fault == binding.SELF_FAULT
    assert other_fault == binding.SELF_FAULT


# --- reward settlement ------------------------------------------------------


def test_reward_uses_counterparty_self_fault():
    reward, rewarded = settle_reward(binding.NON_SELF_FAULT, binding.SELF_FAULT)
    assert math.isclose(reward, 0.8, rel_tol=FLOAT32_TOLERANCE)
    assert rewarded == 1


def test_own_fault_takes_priority_when_both_are_at_fault():
    reward, rewarded = settle_reward(binding.SELF_FAULT, binding.SELF_FAULT)
    assert math.isclose(reward, -0.25, rel_tol=FLOAT32_TOLERANCE)
    assert rewarded == 0


def test_stopped_vehicle_is_not_rewarded_again_when_third_car_hits_it():
    stopped_reward, rewarded = settle_reward(
        binding.NON_SELF_FAULT,
        binding.SELF_FAULT,
        stopped=True,
        already_rewarded=True,
    )
    moving_third_car_reward, _ = settle_reward(
        binding.SELF_FAULT,
        binding.NON_SELF_FAULT,
    )
    assert math.isclose(stopped_reward, 0.0, abs_tol=FLOAT32_TOLERANCE)
    assert rewarded == 1
    assert math.isclose(moving_third_car_reward, -0.25, rel_tol=FLOAT32_TOLERANCE)


def test_stopped_vehicle_cannot_get_first_positive_reward_from_late_collision():
    reward, rewarded = settle_reward(
        binding.NON_SELF_FAULT,
        binding.SELF_FAULT,
        stopped=True,
        already_rewarded=False,
        reward_once=False,
    )
    assert math.isclose(reward, 0.0, abs_tol=FLOAT32_TOLERANCE)
    assert rewarded == 0


def test_reward_once_still_blocks_a_second_positive_event():
    reward, rewarded = settle_reward(
        binding.NON_SELF_FAULT,
        binding.SELF_FAULT,
        already_rewarded=True,
    )
    assert math.isclose(reward, 0.0, abs_tol=FLOAT32_TOLERANCE)
    assert rewarded == 1


def test_static_expert_velocity_is_live_before_global_impact_snapshot():
    vx, vy, impact_vx, impact_vy = binding.expert_impact_snapshot(7.5, -1.25)
    assert math.isclose(vx, 7.5, rel_tol=FLOAT32_TOLERANCE)
    assert math.isclose(vy, -1.25, rel_tol=FLOAT32_TOLERANCE)
    assert math.isclose(impact_vx, 7.5, rel_tol=FLOAT32_TOLERANCE)
    assert math.isclose(impact_vy, -1.25, rel_tol=FLOAT32_TOLERANCE)
