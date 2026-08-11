#include "drive.h"
#include <Python.h>
#define Env Drive
#define MY_SHARED
#define MY_PUT
static PyObject *py_obb_contact_point(PyObject *self, PyObject *args);
static PyObject *py_classify_collision_fault(PyObject *self, PyObject *args);
static PyObject *py_classify_collision_pair(PyObject *self, PyObject *args);
static PyObject *py_settle_adversarial_collision_reward(PyObject *self, PyObject *args);
static PyObject *py_expert_impact_snapshot(PyObject *self, PyObject *args);
#define MY_METHODS                                                                                                     \
    {"obb_contact_point", py_obb_contact_point, METH_VARARGS,                                                          \
     "Overlap centroid of two synthetic oriented bounding boxes"},                                                     \
    {"classify_collision_fault", py_classify_collision_fault, METH_VARARGS,                                            \
     "Classify a synthetic contact for validation and diagnostics"},                                                   \
    {"classify_collision_pair", py_classify_collision_pair, METH_VARARGS,                                              \
     "Classify both participants from one synthetic pre-impact snapshot"},                                            \
    {"settle_adversarial_collision_reward", py_settle_adversarial_collision_reward, METH_VARARGS,                     \
     "Settle a synthetic adversarial collision event"},                                                               \
    {"expert_impact_snapshot", py_expert_impact_snapshot, METH_VARARGS,                                                \
     "Move a synthetic expert and return its live and snapshotted velocity"}
#include "../env_binding.h"

// Fault attribution is geometric, so the harness describes each participant by
// its full pose and box, not just a velocity vector: (x, y, heading, length,
// width, vx, vy). The contact point is then derived by the same code the env
// runs, rather than being supplied by the test.
static int unpack_entity(PyObject *obj, Entity *entity) {
    double x, y, heading, length, width, vx, vy;
    if (!PyTuple_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "expected a tuple (x, y, heading, length, width, vx, vy)");
        return 0;
    }
    if (!PyArg_ParseTuple(obj, "ddddddd", &x, &y, &heading, &length, &width, &vx, &vy))
        return 0;

    entity->x = (float)x;
    entity->y = (float)y;
    entity->heading = (float)heading;
    entity->heading_x = cosf((float)heading);
    entity->heading_y = sinf((float)heading);
    entity->length = (float)length;
    entity->width = (float)width;
    entity->vx = (float)vx;
    entity->vy = (float)vy;
    entity->impact_vx = (float)vx;
    entity->impact_vy = (float)vy;
    return 1;
}

static int unpack_synthetic_contact(PyObject *args, Drive *env, Entity *agent, Entity *other, float *contact_x,
                                    float *contact_y) {
    PyObject *agent_obj;
    PyObject *other_obj;
    double speed_threshold, front_cos_threshold;
    if (!PyArg_ParseTuple(args, "OOdd", &agent_obj, &other_obj, &speed_threshold, &front_cos_threshold))
        return 0;
    if (!unpack_entity(agent_obj, agent) || !unpack_entity(other_obj, other))
        return 0;

    env->fault_speed_threshold = (float)speed_threshold;
    env->front_contact_cos_threshold = (float)front_cos_threshold;
    obb_contact_point(agent, other, contact_x, contact_y);
    return 1;
}

static PyObject *py_obb_contact_point(PyObject *self, PyObject *args) {
    PyObject *agent_obj;
    PyObject *other_obj;
    Entity agent = {0};
    Entity other = {0};
    if (!PyArg_ParseTuple(args, "OO", &agent_obj, &other_obj))
        return NULL;
    if (!unpack_entity(agent_obj, &agent) || !unpack_entity(other_obj, &other))
        return NULL;

    float contact_x = 0.0f;
    float contact_y = 0.0f;
    int overlapped = obb_contact_point(&agent, &other, &contact_x, &contact_y);
    return Py_BuildValue("(iff)", overlapped, contact_x, contact_y);
}

static PyObject *py_classify_collision_fault(PyObject *self, PyObject *args) {
    Drive env = {0};
    Entity agent = {0};
    Entity other = {0};
    float contact_x, contact_y;
    if (!unpack_synthetic_contact(args, &env, &agent, &other, &contact_x, &contact_y))
        return NULL;

    int fault = classify_collision_fault(&env, &agent, &other, contact_x, contact_y);
    return PyLong_FromLong(fault);
}

static PyObject *py_classify_collision_pair(PyObject *self, PyObject *args) {
    Drive env = {0};
    Entity agent = {0};
    Entity other = {0};
    float contact_x, contact_y;
    if (!unpack_synthetic_contact(args, &env, &agent, &other, &contact_x, &contact_y))
        return NULL;

    CollisionFaultPair pair = classify_collision_pair(&env, &agent, &other, contact_x, contact_y);
    return Py_BuildValue("(ii)", pair.self_fault, pair.other_fault);
}

static PyObject *py_settle_adversarial_collision_reward(PyObject *self, PyObject *args) {
    int self_fault, other_fault, stopped, already_rewarded, reward_once;
    double self_fault_factor, non_self_fault_factor;
    if (!PyArg_ParseTuple(args, "iiiiidd", &self_fault, &other_fault, &stopped, &already_rewarded, &reward_once,
                          &self_fault_factor, &non_self_fault_factor))
        return NULL;

    Entity agent = {
        .stopped = stopped,
        .non_self_fault_rewarded = already_rewarded,
        .self_fault_factor = (float)self_fault_factor,
        .non_self_fault_factor = (float)non_self_fault_factor,
    };
    float reward = settle_adversarial_collision_reward(&agent, self_fault, other_fault, reward_once);
    return Py_BuildValue("(fi)", reward, agent.non_self_fault_rewarded);
}

static PyObject *py_expert_impact_snapshot(PyObject *self, PyObject *args) {
    double trajectory_vx, trajectory_vy;
    if (!PyArg_ParseTuple(args, "dd", &trajectory_vx, &trajectory_vy))
        return NULL;

    float traj_x[1] = {0.0f};
    float traj_y[1] = {0.0f};
    float traj_z[1] = {0.0f};
    float traj_vx[1] = {(float)trajectory_vx};
    float traj_vy[1] = {(float)trajectory_vy};
    float traj_vz[1] = {0.0f};
    float traj_heading[1] = {0.0f};
    int traj_valid[1] = {1};
    Entity expert = {
        .type = VEHICLE,
        .array_size = 1,
        .traj_x = traj_x,
        .traj_y = traj_y,
        .traj_z = traj_z,
        .traj_vx = traj_vx,
        .traj_vy = traj_vy,
        .traj_vz = traj_vz,
        .traj_heading = traj_heading,
        .traj_valid = traj_valid,
    };
    Drive env = {
        .entities = &expert,
        .num_objects = 1,
        .timestep = 0,
    };

    move_expert(&env, NULL, 0);
    snapshot_impact_velocities(&env);
    return Py_BuildValue("(ffff)", expert.vx, expert.vy, expert.impact_vx, expert.impact_vy);
}

static int my_put(Env *env, PyObject *args, PyObject *kwargs) {
    PyObject *obs = PyDict_GetItemString(kwargs, "observations");
    if (!PyObject_TypeCheck(obs, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Observations must be a NumPy array");
        return 1;
    }
    PyArrayObject *observations = (PyArrayObject *)obs;
    if (!PyArray_ISCONTIGUOUS(observations)) {
        PyErr_SetString(PyExc_ValueError, "Observations must be contiguous");
        return 1;
    }
    env->observations = PyArray_DATA(observations);

    PyObject *act = PyDict_GetItemString(kwargs, "actions");
    if (!PyObject_TypeCheck(act, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Actions must be a NumPy array");
        return 1;
    }
    PyArrayObject *actions = (PyArrayObject *)act;
    if (!PyArray_ISCONTIGUOUS(actions)) {
        PyErr_SetString(PyExc_ValueError, "Actions must be contiguous");
        return 1;
    }
    env->actions = PyArray_DATA(actions);
    if (PyArray_ITEMSIZE(actions) == sizeof(double)) {
        PyErr_SetString(PyExc_ValueError, "Action tensor passed as float64 (pass np.float32 buffer)");
        return 1;
    }

    PyObject *rew = PyDict_GetItemString(kwargs, "rewards");
    if (!PyObject_TypeCheck(rew, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Rewards must be a NumPy array");
        return 1;
    }
    PyArrayObject *rewards = (PyArrayObject *)rew;
    if (!PyArray_ISCONTIGUOUS(rewards)) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be contiguous");
        return 1;
    }
    if (PyArray_NDIM(rewards) != 1) {
        PyErr_SetString(PyExc_ValueError, "Rewards must be 1D");
        return 1;
    }
    env->rewards = PyArray_DATA(rewards);

    PyObject *term = PyDict_GetItemString(kwargs, "terminals");
    if (!PyObject_TypeCheck(term, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "Terminals must be a NumPy array");
        return 1;
    }
    PyArrayObject *terminals = (PyArrayObject *)term;
    if (!PyArray_ISCONTIGUOUS(terminals)) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be contiguous");
        return 1;
    }
    if (PyArray_NDIM(terminals) != 1) {
        PyErr_SetString(PyExc_ValueError, "Terminals must be 1D");
        return 1;
    }
    env->terminals = PyArray_DATA(terminals);
    return 0;
}

static PyObject *my_shared(PyObject *self, PyObject *args, PyObject *kwargs) {
    char *map_dir = unpack_str(kwargs, "map_dir");
    int num_agents = unpack(kwargs, "num_agents");
    int num_maps = unpack(kwargs, "num_maps");
    int sample_mode = unpack(kwargs, "sample_mode"); // 0: random, 1: sequential
    int init_mode = unpack(kwargs, "init_mode");
    int control_mode = unpack(kwargs, "control_mode");
    int init_steps = unpack(kwargs, "init_steps");
    int goal_behavior = unpack(kwargs, "goal_behavior");
    float goal_target_distance = unpack(kwargs, "goal_target_distance");
    int max_controlled_agents = unpack(kwargs, "max_controlled_agents");
    int condition_sample_mode = unpack(kwargs, "condition_sample_mode");
    float lane_width_min = unpack(kwargs, "lane_width_min");
    float lane_width_max = unpack(kwargs, "lane_width_max");
    float fixed_lane_width = unpack(kwargs, "fixed_lane_width");
    int offroad_mode = unpack(kwargs, "offroad_mode");
    int centerline_only = unpack(kwargs, "centerline_only");
    float lane_width = unpack(kwargs, "lane_width");

    if (sample_mode == 0) {
        clock_gettime(CLOCK_REALTIME, &ts);
        srand(ts.tv_nsec); // Random sampling with replacement
    }

    static int sequential_cursor = 0;

    int total_agent_count = 0;
    int env_count = 0;

    int max_envs = num_agents;

    int maps_checked = 0;
    PyObject *agent_offsets = PyList_New(max_envs + 1);
    PyObject *map_ids = PyList_New(max_envs);

    // Getting env count
    while (total_agent_count < num_agents && env_count < max_envs) {
        char map_file[512];

        int map_id;
        if (sample_mode == 1) {
            map_id = sequential_cursor; // % num_maps;
            sequential_cursor++;
        } else {
            map_id = rand() % num_maps;
        }

        // printf("Sampling map_id: %d\n", map_id);

        Drive *env = calloc(1, sizeof(Drive));
        env->init_mode = init_mode;
        env->control_mode = control_mode;
        env->init_steps = init_steps;
        env->goal_behavior = goal_behavior;
        env->goal_target_distance = goal_target_distance;
        env->max_controlled_agents = max_controlled_agents;
        env->condition_sample_mode = condition_sample_mode;
        env->lane_width_min = lane_width_min;
        env->lane_width_max = lane_width_max;
        env->fixed_lane_width = fixed_lane_width;
        env->offroad_mode = offroad_mode;
        env->centerline_only = centerline_only;
        env->lane_width = lane_width;
        snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", map_dir, map_id);
        env->entities = load_map_binary(map_file, env);
        // Count the number of controllable agents in map
        set_active_agents(env);

        // Skip map if it doesn't contain any controllable agents
        if (env->active_agent_count == 0) {
            maps_checked++;

            // Safeguard: if we've checked all available maps and found no active agents, raise an error
            if (maps_checked >= num_maps) {
                for (int j = 0; j < env->num_entities; j++) {
                    free_entity(&env->entities[j]);
                }
                free(env->entities);
                free(env->active_agent_indices);
                free(env->static_agent_indices);
                free(env->expert_static_agent_indices);
                free(env->tracks_to_predict_indices);
                free(env);
                Py_DECREF(agent_offsets);
                Py_DECREF(map_ids);
                char error_msg[256];
                sprintf(error_msg, "No controllable agents found in any of the %d available maps", num_maps);
                PyErr_SetString(PyExc_ValueError, error_msg);
                return NULL;
            }

            for (int j = 0; j < env->num_entities; j++) {
                free_entity(&env->entities[j]);
            }
            free(env->entities);
            free(env->active_agent_indices);
            free(env->static_agent_indices);
            free(env->expert_static_agent_indices);
            free(env->tracks_to_predict_indices);
            free(env);
            continue;
        }

        // Store map_id
        PyObject *map_id_obj = PyLong_FromLong(map_id);
        PyList_SetItem(map_ids, env_count, map_id_obj);
        // Store agent offset
        PyObject *offset = PyLong_FromLong(total_agent_count);
        PyList_SetItem(agent_offsets, env_count, offset);
        total_agent_count += env->active_agent_count;
        env_count++;
        for (int j = 0; j < env->num_entities; j++) {
            free_entity(&env->entities[j]);
        }
        free(env->entities);
        free(env->active_agent_indices);
        free(env->static_agent_indices);
        free(env->tracks_to_predict_indices);
        free(env->expert_static_agent_indices);
        free(env);
    }

    if (total_agent_count >= num_agents) {
        total_agent_count = num_agents;
    }

    PyObject *final_total_agent_count = PyLong_FromLong(total_agent_count);
    PyList_SetItem(agent_offsets, env_count, final_total_agent_count);
    PyObject *final_env_count = PyLong_FromLong(env_count);

    // resize lists
    PyObject *resized_agent_offsets = PyList_GetSlice(agent_offsets, 0, env_count + 1);
    PyObject *resized_map_ids = PyList_GetSlice(map_ids, 0, env_count);
    PyObject *tuple = PyTuple_New(3);
    PyTuple_SetItem(tuple, 0, resized_agent_offsets);
    PyTuple_SetItem(tuple, 1, resized_map_ids);
    PyTuple_SetItem(tuple, 2, final_env_count);
    return tuple;
}

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
    env->human_agent_idx = unpack(kwargs, "human_agent_idx");
    env->ini_file = unpack_str(kwargs, "ini_file");
    env_init_config conf = {0};
    conf.centerline_only = 1; // Default: only centerline (ROAD_LANE) in grid map / observation
    conf.collision_reward_mode = COLLISION_REWARD_ADVERSARIAL;
    conf.fault_speed_threshold = 0.5f;
    conf.front_contact_cos_threshold = 0.70710678f;
    conf.non_self_fault_reward_once = 1;
    if (ini_parse(env->ini_file, handler, &conf) < 0) {
        printf("Error while loading %s", env->ini_file);
    }
    if (kwargs && PyDict_GetItemString(kwargs, "episode_length")) {
        conf.episode_length = (int)unpack(kwargs, "episode_length");
    }
    if (conf.episode_length <= 0) {
        PyErr_SetString(PyExc_ValueError, "episode_length must be > 0 (set in INI or kwargs)");
        return -1;
    }

// Allow all settings to be overridden via kwargs (ini provides defaults)
#define OVERRIDE_INT(field)                                                                                            \
    if (kwargs && PyDict_GetItemString(kwargs, #field)) {                                                              \
        conf.field = (int)unpack(kwargs, #field);                                                                      \
    }
#define OVERRIDE_FLOAT(field)                                                                                          \
    if (kwargs && PyDict_GetItemString(kwargs, #field)) {                                                              \
        conf.field = (float)unpack(kwargs, #field);                                                                    \
    }

    OVERRIDE_INT(render_mode);
    OVERRIDE_INT(action_type);
    OVERRIDE_INT(dynamics_model);
    OVERRIDE_FLOAT(self_fault_factor_min);
    OVERRIDE_FLOAT(self_fault_factor_max);
    OVERRIDE_FLOAT(non_self_fault_factor_min);
    OVERRIDE_FLOAT(non_self_fault_factor_max);
    OVERRIDE_FLOAT(offroad_factor_min);
    OVERRIDE_FLOAT(offroad_factor_max);
    OVERRIDE_INT(condition_sample_mode);
    OVERRIDE_FLOAT(fixed_self_fault_factor);
    OVERRIDE_FLOAT(fixed_non_self_fault_factor);
    OVERRIDE_FLOAT(fixed_offroad_factor);
    OVERRIDE_INT(collision_reward_mode);
    OVERRIDE_FLOAT(fault_speed_threshold);
    OVERRIDE_FLOAT(front_contact_cos_threshold);
    OVERRIDE_INT(non_self_fault_reward_once);
    OVERRIDE_FLOAT(lane_width_min);
    OVERRIDE_FLOAT(lane_width_max);
    OVERRIDE_FLOAT(fixed_lane_width);
    OVERRIDE_FLOAT(reward_goal);
    OVERRIDE_FLOAT(reward_goal_post_respawn);
    OVERRIDE_FLOAT(reward_steer_jitter);
    OVERRIDE_FLOAT(reward_time_penalty);
    OVERRIDE_FLOAT(overspeed_penalty);
    OVERRIDE_INT(collision_behavior);
    OVERRIDE_INT(offroad_behavior);
    OVERRIDE_INT(offroad_mode);
    OVERRIDE_INT(centerline_only);
    OVERRIDE_FLOAT(lane_width);
    OVERRIDE_FLOAT(dt);
    OVERRIDE_INT(termination_mode);
    OVERRIDE_INT(init_mode);
    OVERRIDE_INT(control_mode);
    OVERRIDE_INT(goal_behavior);
    OVERRIDE_FLOAT(goal_target_distance);
    OVERRIDE_FLOAT(goal_radius);
    OVERRIDE_FLOAT(goal_speed);
    OVERRIDE_INT(max_controlled_agents);

#undef OVERRIDE_INT
#undef OVERRIDE_FLOAT

    env->action_type = conf.action_type;
    env->dynamics_model = conf.dynamics_model;
    env->self_fault_factor_min = conf.self_fault_factor_min;
    env->self_fault_factor_max = conf.self_fault_factor_max;
    env->non_self_fault_factor_min = conf.non_self_fault_factor_min;
    env->non_self_fault_factor_max = conf.non_self_fault_factor_max;
    env->offroad_factor_min = conf.offroad_factor_min;
    env->offroad_factor_max = conf.offroad_factor_max;
    env->condition_sample_mode = conf.condition_sample_mode;
    env->fixed_self_fault_factor = conf.fixed_self_fault_factor;
    env->fixed_non_self_fault_factor = conf.fixed_non_self_fault_factor;
    env->fixed_offroad_factor = conf.fixed_offroad_factor;
    env->collision_reward_mode = conf.collision_reward_mode;
    env->fault_speed_threshold = conf.fault_speed_threshold;
    env->front_contact_cos_threshold = conf.front_contact_cos_threshold;
    env->non_self_fault_reward_once = conf.non_self_fault_reward_once;
    env->lane_width_min = conf.lane_width_min > 0.0f ? conf.lane_width_min : 1.0f;
    env->lane_width_max = conf.lane_width_max > 0.0f ? conf.lane_width_max : 5.0f;
    env->fixed_lane_width = conf.fixed_lane_width > 0.0f ? conf.fixed_lane_width : 3.5f;
    env->reward_goal = conf.reward_goal;
    env->reward_goal_post_respawn = conf.reward_goal_post_respawn;
    env->reward_steer_jitter = conf.reward_steer_jitter;
    env->reward_time_penalty = conf.reward_time_penalty;
    env->overspeed_penalty = conf.overspeed_penalty;
    env->episode_length = conf.episode_length;
    env->termination_mode = conf.termination_mode;
    env->collision_behavior = conf.collision_behavior;
    env->offroad_behavior = conf.offroad_behavior;
    env->offroad_mode = conf.offroad_mode;
    env->centerline_only = conf.centerline_only;
    env->lane_width = conf.lane_width > 0.0f ? conf.lane_width : 3.5f;
    env->max_controlled_agents = unpack(kwargs, "max_controlled_agents");
    env->dt = conf.dt;
    env->init_mode = (int)unpack(kwargs, "init_mode");
    env->control_mode = (int)unpack(kwargs, "control_mode");
    env->goal_behavior = (int)unpack(kwargs, "goal_behavior");
    env->goal_target_distance = (float)unpack(kwargs, "goal_target_distance");
    env->goal_radius = (float)unpack(kwargs, "goal_radius");
    env->goal_speed = (float)unpack(kwargs, "goal_speed");
    env->render_mode = (int)unpack(kwargs, "render_mode");
    char *map_dir = unpack_str(kwargs, "map_dir");
    int map_id = unpack(kwargs, "map_id");
    int max_agents = unpack(kwargs, "max_agents");
    int init_steps = unpack(kwargs, "init_steps");
    char map_file[512];
    snprintf(map_file, sizeof(map_file), "%s/map_%03d.bin", map_dir, map_id);
    env->num_agents = max_agents;
    env->map_name = strdup(map_file);
    env->init_steps = init_steps;
    env->timestep = init_steps;
    init(env);
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "n", log->n);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "offroad_rate", log->offroad_rate);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "self_fault_collision_rate", log->self_fault_collision_rate);
    assign_to_dict(dict, "non_self_fault_collision_rate", log->non_self_fault_collision_rate);
    assign_to_dict(dict, "ambiguous_collision_rate", log->ambiguous_collision_rate);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "dnf_rate", log->dnf_rate);
    assign_to_dict(dict, "completion_rate", log->completion_rate);
    assign_to_dict(dict, "lane_alignment_rate", log->lane_alignment_rate);
    assign_to_dict(dict, "perc_controlled", log->perc_controlled);
    assign_to_dict(dict, "perc_other", log->perc_other);
    assign_to_dict(dict, "offroad_per_agent", log->offroad_per_agent);
    assign_to_dict(dict, "collisions_per_agent", log->collisions_per_agent);
    assign_to_dict(dict, "self_fault_collisions_per_agent", log->self_fault_collisions_per_agent);
    assign_to_dict(dict, "non_self_fault_collisions_per_agent", log->non_self_fault_collisions_per_agent);
    assign_to_dict(dict, "ambiguous_collisions_per_agent", log->ambiguous_collisions_per_agent);
    assign_to_dict(dict, "goals_sampled_this_episode", log->goals_sampled_this_episode);
    assign_to_dict(dict, "goals_reached_this_episode", log->goals_reached_this_episode);
    assign_to_dict(dict, "speed_at_goal", log->speed_at_goal);
    // assign_to_dict(dict, "avg_displacement_error", log->avg_displacement_error);
    return 0;
}
