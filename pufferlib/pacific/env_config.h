#ifndef ENV_CONFIG_H
#define ENV_CONFIG_H

#include <../../inih-r62/ini.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Config struct for parsing INI files - contains all environment configuration
typedef struct {
    int render_mode;
    int action_type;
    int dynamics_model;
    float collision_factor_min;
    float collision_factor_max;
    float offroad_factor_min;
    float offroad_factor_max;
    int condition_sample_mode;
    float fixed_collision_factor;
    float fixed_offroad_factor;
    float lane_width_min;
    float lane_width_max;
    float fixed_lane_width;
    float reward_goal;
    float reward_goal_post_respawn;
    float reward_vehicle_collision_post_respawn;
    float reward_steer_jitter;
    float reward_time_penalty;
    float goal_radius;
    float goal_speed;
    int collision_behavior;
    int offroad_behavior;
    int offroad_mode;
    float lane_width;
    int spawn_immunity_timer;
    float dt;
    int goal_behavior;
    float goal_target_distance;
    int episode_length;
    int termination_mode;
    int init_steps;
    int init_mode;
    int control_mode;
    int max_controlled_agents;
    char map_dir[256];
} env_init_config;

// INI file parser handler - parses all environment configuration from drive.ini
static int handler(void *config, const char *section, const char *name, const char *value) {
    env_init_config *env_config = (env_init_config *)config;
#define MATCH(s, n) strcmp(section, s) == 0 && strcmp(name, n) == 0

    if (MATCH("env", "action_type")) {
        if (strcmp(value, "\"discrete\"") == 0 || strcmp(value, "discrete") == 0) {
            env_config->action_type = 0; // DISCRETE
        } else if (strcmp(value, "\"continuous\"") == 0 || strcmp(value, "continuous") == 0) {
            env_config->action_type = 1; // CONTINUOUS
        } else {
            printf("Warning: Unknown action_type value '%s', defaulting to DISCRETE\n", value);
            env_config->action_type = 0; // Default to DISCRETE
        }
    } else if (MATCH("env", "dynamics_model")) {
        if (strcmp(value, "\"classic\"") == 0 || strcmp(value, "classic") == 0) {
            env_config->dynamics_model = 0; // CLASSIC
        } else if (strcmp(value, "\"jerk\"") == 0 || strcmp(value, "jerk") == 0) {
            env_config->dynamics_model = 1; // JERK
        } else {
            printf("Warning: Unknown dynamics_model value '%s', defaulting to JERK\n", value);
            env_config->dynamics_model = 1; // Default to JERK
        }
    } else if (MATCH("env", "goal_behavior")) {
        env_config->goal_behavior = atoi(value);
    } else if (MATCH("env", "goal_target_distance")) {
        env_config->goal_target_distance = atof(value);
    } else if (MATCH("env", "collision_factor_min")) {
        env_config->collision_factor_min = atof(value);
    } else if (MATCH("env", "collision_factor_max")) {
        env_config->collision_factor_max = atof(value);
    } else if (MATCH("env", "offroad_factor_min")) {
        env_config->offroad_factor_min = atof(value);
    } else if (MATCH("env", "offroad_factor_max")) {
        env_config->offroad_factor_max = atof(value);
    } else if (MATCH("env", "condition_sample_mode")) {
        if (strcmp(value, "\"random\"") == 0 || strcmp(value, "random") == 0) {
            env_config->condition_sample_mode = 0;
        } else if (strcmp(value, "\"fixed\"") == 0 || strcmp(value, "fixed") == 0) {
            env_config->condition_sample_mode = 1;
        } else {
            printf("Warning: Unknown condition_sample_mode value '%s', defaulting to random\n", value);
            env_config->condition_sample_mode = 0;
        }
    } else if (MATCH("env", "fixed_collision_factor")) {
        env_config->fixed_collision_factor = atof(value);
    } else if (MATCH("env", "fixed_offroad_factor")) {
        env_config->fixed_offroad_factor = atof(value);
    } else if (MATCH("env", "lane_width_min")) {
        env_config->lane_width_min = atof(value);
    } else if (MATCH("env", "lane_width_max")) {
        env_config->lane_width_max = atof(value);
    } else if (MATCH("env", "fixed_lane_width")) {
        env_config->fixed_lane_width = atof(value);
    } else if (MATCH("env", "reward_goal")) {
        env_config->reward_goal = atof(value);
    } else if (MATCH("env", "reward_goal_post_respawn")) {
        env_config->reward_goal_post_respawn = atof(value);
    } else if (MATCH("env", "reward_steer_jitter")) {
        env_config->reward_steer_jitter = atof(value);
    } else if (MATCH("env", "reward_time_penalty")) {
        env_config->reward_time_penalty = atof(value);
    } else if (MATCH("env", "reward_vehicle_collision_post_respawn")) {
        env_config->reward_vehicle_collision_post_respawn = atof(value);
    } else if (MATCH("env", "goal_radius")) {
        env_config->goal_radius = atof(value);
    } else if (MATCH("env", "goal_speed")) {
        env_config->goal_speed = atof(value);
    } else if (MATCH("env", "collision_behavior")) {
        env_config->collision_behavior = atoi(value);
    } else if (MATCH("env", "offroad_behavior")) {
        env_config->offroad_behavior = atoi(value);
    } else if (MATCH("env", "offroad_mode")) {
        if (strcmp(value, "\"road_edge\"") == 0 || strcmp(value, "road_edge") == 0) {
            env_config->offroad_mode = 0;
        } else if (strcmp(value, "\"lane_center\"") == 0 || strcmp(value, "lane_center") == 0) {
            env_config->offroad_mode = 1;
        } else {
            printf("Warning: Unknown offroad_mode value '%s', defaulting to road_edge\n", value);
            env_config->offroad_mode = 0;
        }
    } else if (MATCH("env", "lane_width")) {
        env_config->lane_width = atof(value);
    } else if (MATCH("env", "spawn_immunity_timer")) {
        env_config->spawn_immunity_timer = atoi(value);
    } else if (MATCH("env", "dt")) {
        env_config->dt = atof(value);
    } else if (MATCH("env", "episode_length")) {
        env_config->episode_length = atoi(value);
    } else if (MATCH("env", "termination_mode")) {
        env_config->termination_mode = atoi(value);
    } else if (MATCH("env", "init_steps")) {
        env_config->init_steps = atoi(value);
    } else if (MATCH("env", "init_mode")) {
        if (strcmp(value, "\"create_all_valid\"") == 0 || strcmp(value, "create_all_valid") == 0) {
            env_config->init_mode = 0;
        } else if (strcmp(value, "\"create_only_controlled\"") == 0 || strcmp(value, "create_only_controlled") == 0) {
            env_config->init_mode = 1;
        } else {
            printf("Warning: Unknown init_mode value '%s', defaulting to CREATE_ALL_VALID\n", value);
            env_config->init_mode = 0; // Default to CREATE_ALL_VALID
        }
    } else if (MATCH("env", "control_mode")) {
        if (strcmp(value, "\"control_vehicles\"") == 0 || strcmp(value, "control_vehicles") == 0) {
            env_config->control_mode = 0;
        } else if (strcmp(value, "\"control_agents\"") == 0 || strcmp(value, "control_agents") == 0) {
            env_config->control_mode = 1;
        } else if (strcmp(value, "\"control_wosac\"") == 0 || strcmp(value, "control_wosac") == 0) {
            env_config->control_mode = 2;
        } else if (strcmp(value, "\"control_sdc_only\"") == 0 || strcmp(value, "control_sdc_only") == 0) {
            env_config->control_mode = 3;
        } else if (strcmp(value, "\"control_mixed_play\"") == 0 || strcmp(value, "control_mixed_play") == 0) {
            env_config->control_mode = 4;
        } else {
            printf("Warning: Unknown control_mode value '%s', defaulting to CONTROL_VEHICLES\n", value);
            env_config->control_mode = 0; // Default to CONTROL_VEHICLES
        }
    } else if (MATCH("env", "map_dir")) {
        if (sscanf(value, "\"%255[^\"]\"", env_config->map_dir) != 1) {
            strncpy(env_config->map_dir, value, sizeof(env_config->map_dir) - 1);
            env_config->map_dir[sizeof(env_config->map_dir) - 1] = '\0';
        }
        // printf("Parsed map_dir: '%s'\n", env_config->map_dir);
    } else if (MATCH("env", "max_controlled_agents")) {
        env_config->max_controlled_agents = atoi(value);
    } else {
        return 0; // Unknown section/name, indicate failure to handle
    }

#undef MATCH
    return 1;
}

#endif // ENV_CONFIG_H
