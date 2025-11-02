// Native Tron environment for PufferLib.
// Implements a multi-agent light-cycle game with partial observability.

#pragma once

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef TRON_RESTRICT
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#define TRON_RESTRICT restrict
#elif defined(_MSC_VER)
#define TRON_RESTRICT __restrict
#else
#define TRON_RESTRICT
#endif
#endif

#include "raylib.h"

#define ACTION_LEFT 0
#define ACTION_FORWARD 1
#define ACTION_RIGHT 2

static const int DIR_X[4] = {0, 1, 0, -1};
static const int DIR_Y[4] = {-1, 0, 1, 0};

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float avg_survival_time;
    float avg_alive_agents;
    float n;
} Log;

typedef struct {
    int cell_size;
    bool window_ready;
} Client;

typedef struct {
    Log log;
    Client* client;

    float* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    unsigned char* masks;

    int width;
    int height;
    int num_agents;
    int vision;
    int obs_size;

    int spawn_separation;
    int spawn_margin;
    int spawn_attempts;
    int max_round_steps;

    float reward_alive;
    float reward_step;
    float reward_win;
    float reward_loss;
    float reward_tie;
    float reward_timeout;
    float reward_kill;

    int step_count;
    int steps_in_round;
    int alive_count;

    int* pos_x;
    int* pos_y;
    unsigned char* directions;
    unsigned char* alive;
    unsigned int* life_time;
    float* life_reward;

    int* next_x;
    int* next_y;
    int* next_index;
    unsigned char* collided;

    unsigned short* move_counts;
    int* move_first;
    unsigned char* trail_owner;
    unsigned char* head_owner;
    unsigned char* crash_map;

    int move_touched_count;
    int crash_touched_count;
    int* move_touched;
    int* crash_touched;
} Tron;

static inline void set_mask(Tron* env, int agent, unsigned char value) {
    if (env->masks) {
        env->masks[agent] = value;
    }
}

static void init(Tron* env) {
    int agents = env->num_agents;
    int cells = env->width * env->height;

    env->client = NULL;
    env->obs_size = env->vision * env->vision * 2 + 5;
    env->spawn_attempts = env->spawn_attempts > 0 ? env->spawn_attempts : cells * 8;

    env->pos_x = (int*)calloc(agents, sizeof(int));
    env->pos_y = (int*)calloc(agents, sizeof(int));
    env->directions = (unsigned char*)calloc(agents, sizeof(unsigned char));
    env->alive = (unsigned char*)calloc(agents, sizeof(unsigned char));
    env->life_time = (unsigned int*)calloc(agents, sizeof(unsigned int));
    env->life_reward = (float*)calloc(agents, sizeof(float));
    env->next_x = (int*)calloc(agents, sizeof(int));
    env->next_y = (int*)calloc(agents, sizeof(int));
    env->next_index = (int*)calloc(agents, sizeof(int));
    env->collided = (unsigned char*)calloc(agents, sizeof(unsigned char));

    env->move_counts = (unsigned short*)calloc(cells, sizeof(unsigned short));
    env->move_first = (int*)malloc(cells * sizeof(int));
    for (int i = 0; i < cells; i++) {
        env->move_first[i] = -1;
    }
    env->trail_owner = (unsigned char*)calloc(cells, sizeof(unsigned char));
    env->head_owner = (unsigned char*)calloc(cells, sizeof(unsigned char));
    env->crash_map = (unsigned char*)calloc(cells, sizeof(unsigned char));

    env->move_touched = (int*)calloc(agents, sizeof(int));
    env->crash_touched = (int*)calloc(agents, sizeof(int));
    env->move_touched_count = 0;
    env->crash_touched_count = 0;
}

static inline int manhattan(int x0, int y0, int x1, int y1) {
    return abs(x0 - x1) + abs(y0 - y1);
}

static void clear_round_state(Tron* env) {
    int agents = env->num_agents;
    int cells = env->width * env->height;
    memset(env->trail_owner, 0, cells * sizeof(unsigned char));
    memset(env->head_owner, 0, cells * sizeof(unsigned char));
    memset(env->crash_map, 0, cells * sizeof(unsigned char));
    memset(env->move_counts, 0, cells * sizeof(unsigned short));
    for (int i = 0; i < cells; i++) {
        env->move_first[i] = -1;
    }
    memset(env->collided, 0, agents * sizeof(unsigned char));
    memset(env->life_time, 0, agents * sizeof(unsigned int));
    memset(env->life_reward, 0, agents * sizeof(float));
    env->steps_in_round = 0;
    env->move_touched_count = 0;
    env->crash_touched_count = 0;
}

static void spawn_agents(Tron* env) {
    clear_round_state(env);
    int agents = env->num_agents;
    int width = env->width;
    int sep = env->spawn_separation;
    int margin = env->spawn_margin;
    int height = env->height;
    if (margin < 0) {
        margin = 0;
    }

    for (int a = 0; a < agents; a++) {
        bool placed = false;
        for (int attempt = 0; attempt < env->spawn_attempts; attempt++) {
            int xmin = margin;
            int xmax = width - margin - 1;
            int ymin = margin;
            int ymax = height - margin - 1;
            if (xmax < xmin || ymax < ymin) {
                xmin = 0;
                ymin = 0;
                xmax = width - 1;
                ymax = height - 1;
            }

            int x = xmin + rand() % (xmax - xmin + 1);
            int y = ymin + rand() % (ymax - ymin + 1);

            bool valid = true;
            for (int b = 0; b < a; b++) {
                if (manhattan(x, y, env->pos_x[b], env->pos_y[b]) < sep) {
                    valid = false;
                    break;
                }
            }

            if (!valid) {
                continue;
            }

            env->pos_x[a] = x;
            env->pos_y[a] = y;
            env->directions[a] = rand() % 4;
            env->alive[a] = 1;
            set_mask(env, a, 1);
            int idx = y * width + x;
            env->trail_owner[idx] = (unsigned char)(a + 1);
            env->head_owner[idx] = (unsigned char)(a + 1);
            placed = true;
            break;
        }

        if (!placed) {
            int x = (a * sep) % width;
            int y = ((a * sep) / width) % height;
            env->pos_x[a] = x;
            env->pos_y[a] = y;
            env->directions[a] = rand() % 4;
            env->alive[a] = 1;
            set_mask(env, a, 1);
            int idx = y * width + x;
            env->trail_owner[idx] = (unsigned char)(a + 1);
            env->head_owner[idx] = (unsigned char)(a + 1);
        }
    }

    env->alive_count = agents;
}

static void write_observations(Tron* env) {
    const int agents = env->num_agents;
    const int width = env->width;
    const int height = env->height;
    const int vision = env->vision;

    const int r_lo = vision / 2;
    const int r_hi = vision - r_lo - 1;

    const size_t window_elems = (size_t)vision * (size_t)vision * 2;

    unsigned char* TRON_RESTRICT trail = env->trail_owner;
    unsigned char* TRON_RESTRICT head = env->head_owner;

    for (int a = 0; a < agents; a++) {
        float* TRON_RESTRICT obs = env->observations + (size_t)a * env->obs_size;
        if (!env->alive[a]) {
            memset(obs, 0, env->obs_size * sizeof(float));
            continue;
        }

        const int cx = env->pos_x[a];
        const int cy = env->pos_y[a];

        float* TRON_RESTRICT out = obs;

        const int y0 = cy - r_lo;
        const int y1_exclusive = cy + r_hi + 1;

        for (int wy = y0; wy < y1_exclusive; ++wy) {
            if ((unsigned)wy < (unsigned)height) {
                const int row = wy * width;

                int x0 = cx - r_lo;
                int x1_exclusive = cx + r_hi + 1;

                int left_zeros = 0;
                if (x0 < 0) {
                    left_zeros = -x0;
                    x0 = 0;
                }

                int right_zeros = 0;
                if (x1_exclusive > width) {
                    right_zeros = x1_exclusive - width;
                    x1_exclusive = width;
                }

                for (int i = 0; i < left_zeros; ++i) {
                    *out++ = 1.0f;
                    *out++ = 0.0f;
                }

                for (int x = x0; x < x1_exclusive; ++x) {
                    const int cell = row + x;
                    *out++ = (float)(trail[cell] != 0);
                    *out++ = (float)(head[cell] != 0);
                }

                for (int i = 0; i < right_zeros; ++i) {
                    *out++ = 1.0f;
                    *out++ = 0.0f;
                }
            } else {
                for (int i = 0; i < vision; ++i) {
                    *out++ = 1.0f;
                    *out++ = 0.0f;
                }
            }
        }

        float* TRON_RESTRICT orient = obs + window_elems;
        orient[0] = orient[1] = orient[2] = orient[3] = 0.0f;
        orient[env->directions[a] & 3] = 1.0f;
        orient[4] = 1.0f;
    }
}

static void reset_round(Tron* env) {
    spawn_agents(env);
    write_observations(env);
}

static void c_reset(Tron* env) {
    env->log = (Log){0};
    env->step_count = 0;
    env->steps_in_round = 0;
    env->alive_count = env->num_agents;

    memset(env->rewards, 0, env->num_agents * sizeof(float));
    memset(env->terminals, 0, env->num_agents * sizeof(unsigned char));
    if (env->masks) {
        for (int a = 0; a < env->num_agents; a++) {
            env->masks[a] = 1;
        }
    }

    reset_round(env);
}

static void accumulate_log(Tron* env, int survivors) {
    float total_survival = 0.0f;
    float total_reward = 0.0f;
    for (int a = 0; a < env->num_agents; a++) {
        total_survival += (float)env->life_time[a];
        total_reward += env->life_reward[a];
    }

    env->log.perf += survivors / (float)env->num_agents;
    env->log.score += (float)env->steps_in_round;
    env->log.episode_return += total_reward / env->num_agents;
    env->log.episode_length += (float)env->steps_in_round;
    env->log.avg_survival_time += total_survival / env->num_agents;
    env->log.avg_alive_agents += survivors / (float)env->num_agents;
    env->log.n += 1.0f;
}

static void c_step(Tron* env) {
    env->step_count += 1;
    env->steps_in_round += 1;

    int agents = env->num_agents;
    int width = env->width;
    int height = env->height;

    for (int i = 0; i < env->move_touched_count; i++) {
        int idx = env->move_touched[i];
        env->move_counts[idx] = 0;
        env->move_first[idx] = -1;
    }
    env->move_touched_count = 0;

    for (int i = 0; i < env->crash_touched_count; i++) {
        env->crash_map[env->crash_touched[i]] = 0;
    }
    env->crash_touched_count = 0;

    memset(env->rewards, 0, agents * sizeof(float));
    memset(env->terminals, 0, agents * sizeof(unsigned char));
    memset(env->collided, 0, agents * sizeof(unsigned char));

    for (int a = 0; a < agents; a++) {
        if (!env->alive[a]) {
            set_mask(env, a, 0);
            continue;
        }

        env->rewards[a] += env->reward_alive + env->reward_step;
        env->life_reward[a] += env->reward_alive + env->reward_step;
        env->life_time[a] += 1;

        int cell = env->pos_y[a] * width + env->pos_x[a];
        env->trail_owner[cell] = (unsigned char)(a + 1);
        env->head_owner[cell] = 0;

        int action = env->actions[a];
        if (action == ACTION_LEFT) {
            env->directions[a] = (unsigned char)((env->directions[a] + 3) & 3);
        } else if (action == ACTION_RIGHT) {
            env->directions[a] = (unsigned char)((env->directions[a] + 1) & 3);
        }

        int dir = env->directions[a];
        int nx = env->pos_x[a] + DIR_X[dir];
        int ny = env->pos_y[a] + DIR_Y[dir];
        env->next_x[a] = nx;
        env->next_y[a] = ny;
        env->next_index[a] = -1;

        if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
            env->collided[a] = 1;
            continue;
        }

        int next_cell = ny * width + nx;
        unsigned char owner = env->trail_owner[next_cell];
        if (owner) {
            env->collided[a] = 1;
            if (env->reward_kill != 0.0f) {
                int killer = (int)owner - 1;
                if (killer >= 0 && killer < agents && killer != a && env->alive[killer]) {
                    env->rewards[killer] += env->reward_kill;
                    env->life_reward[killer] += env->reward_kill;
                }
            }
        } else {
            env->next_index[a] = next_cell;
            unsigned short count = env->move_counts[next_cell];
            if (count == 0) {
                if (env->move_touched_count < agents) {
                    env->move_touched[env->move_touched_count++] = next_cell;
                }
                env->move_counts[next_cell] = 1;
                env->move_first[next_cell] = a;
            } else {
                if (count == 1) {
                    int first = env->move_first[next_cell];
                    if (first >= 0) {
                        env->collided[first] = 1;
                    }
                }
                env->collided[a] = 1;
                env->move_counts[next_cell] = (unsigned short)(count + 1);
            }
        }
    }

    env->alive_count = 0;
    int survivors = 0;
    for (int a = 0; a < agents; a++) {
        if (!env->alive[a]) {
            continue;
        }

        if (env->collided[a]) {
            env->alive[a] = 0;
            set_mask(env, a, 0);
            env->rewards[a] += env->reward_loss;
            env->life_reward[a] += env->reward_loss;
            int idx = env->next_index[a];
            if (idx >= 0) {
                if (env->crash_map[idx] == 0 && env->crash_touched_count < agents) {
                    env->crash_touched[env->crash_touched_count++] = idx;
                }
                env->trail_owner[idx] = (unsigned char)(a + 1);
                env->crash_map[idx] = (unsigned char)(a + 1);
            }
        } else {
            env->pos_x[a] = env->next_x[a];
            env->pos_y[a] = env->next_y[a];
            int idx = env->next_index[a];
            env->trail_owner[idx] = (unsigned char)(a + 1);
            env->head_owner[idx] = (unsigned char)(a + 1);
            survivors += 1;
            env->alive_count += 1;
            set_mask(env, a, 1);
        }
    }

    bool timed_out = env->max_round_steps > 0 && env->steps_in_round >= env->max_round_steps;
    bool round_over = env->alive_count <= 1 || timed_out;

    if (round_over) {
        if (env->alive_count == 0) {
            for (int a = 0; a < agents; a++) {
                env->rewards[a] += env->reward_tie;
                env->life_reward[a] += env->reward_tie;
            }
        } else {
            for (int a = 0; a < agents; a++) {
                if (env->alive[a]) {
                    env->rewards[a] += env->reward_win;
                    env->life_reward[a] += env->reward_win;
                } else if (timed_out) {
                    env->rewards[a] += env->reward_timeout;
                    env->life_reward[a] += env->reward_timeout;
                }
            }
        }

        for (int a = 0; a < agents; a++) {
            env->terminals[a] = 1;
            set_mask(env, a, 1);
        }

        accumulate_log(env, env->alive_count);
        reset_round(env);
        return;
    }

    write_observations(env);
}

static Color HEAD_COLORS[] = {
    (Color){255, 160, 89, 255},
    (Color){122, 200, 255, 255},
    (Color){166, 255, 129, 255},
    (Color){255, 129, 162, 255},
    (Color){255, 214, 102, 255},
    (Color){156, 135, 255, 255},
};

static Color TRAIL_COLORS[] = {
    (Color){180, 90, 40, 255},
    (Color){60, 120, 190, 255},
    (Color){98, 170, 92, 255},
    (Color){180, 70, 110, 255},
    (Color){190, 150, 60, 255},
    (Color){100, 90, 200, 255},
};

static void ensure_client(Tron* env) {
    if (env->client) {
        return;
    }

    env->client = (Client*)calloc(1, sizeof(Client));
    env->client->cell_size = 12;
    env->client->window_ready = true;
    InitWindow(env->width * env->client->cell_size,
        env->height * env->client->cell_size, "PufferLib Tron");
    SetTargetFPS(30);
}

static void c_render(Tron* env) {
    if (IsKeyDown(KEY_ESCAPE)) {
        CloseWindow();
        exit(0);
    }

    ensure_client(env);
    int cell = env->client->cell_size;

    BeginDrawing();
    ClearBackground((Color){12, 18, 24, 255});

    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            int idx = y * env->width + x;
            Color color = (Color){20, 28, 36, 255};

            unsigned char trail = env->trail_owner[idx];
            unsigned char head = env->head_owner[idx];
            unsigned char crash = env->crash_map[idx];

            if (trail) {
                int palette = (trail - 1) % (sizeof(TRAIL_COLORS) / sizeof(Color));
                color = TRAIL_COLORS[palette];
            }

            if (head) {
                int palette = (head - 1) % (sizeof(HEAD_COLORS) / sizeof(Color));
                color = HEAD_COLORS[palette];
            }

            if (crash) {
                color = (Color){245, 66, 86, 255};
            }

            DrawRectangle(x * cell, y * cell, cell, cell, color);
        }
    }

    EndDrawing();
}

static void c_close(Tron* env) {
    if (env->client != NULL) {
        CloseWindow();
        free(env->client);
        env->client = NULL;
    }

    free(env->pos_x);
    free(env->pos_y);
    free(env->directions);
    free(env->alive);
    free(env->life_time);
    free(env->life_reward);
    free(env->next_x);
    free(env->next_y);
    free(env->next_index);
    free(env->collided);
    free(env->move_counts);
    free(env->move_first);
    free(env->trail_owner);
    free(env->head_owner);
    free(env->crash_map);
    free(env->move_touched);
    free(env->crash_touched);
}
