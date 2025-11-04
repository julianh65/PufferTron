#include "tron.h"

#define Env Tron
#define MY_GET
#include "../env_binding.h"

static double get_double(PyObject* kwargs, const char* key, double default_value, bool required) {
    PyObject* val = PyDict_GetItemString(kwargs, key);
    if (val == NULL) {
        if (required) {
            PyErr_Format(PyExc_KeyError, "Missing required keyword '%s'", key);
        }
        return default_value;
    }

    if (PyFloat_Check(val)) {
        return PyFloat_AsDouble(val);
    }

    if (PyLong_Check(val)) {
        return (double)PyLong_AsLong(val);
    }

    PyErr_Format(PyExc_TypeError, "Keyword '%s' must be numeric", key);
    return default_value;
}

static long get_long(PyObject* kwargs, const char* key, long default_value, bool required) {
    PyObject* val = PyDict_GetItemString(kwargs, key);
    if (val == NULL) {
        if (required) {
            PyErr_Format(PyExc_KeyError, "Missing required keyword '%s'", key);
        }
        return default_value;
    }

    if (PyLong_Check(val)) {
        return PyLong_AsLong(val);
    }

    if (PyFloat_Check(val)) {
        return (long)PyFloat_AsDouble(val);
    }

    PyErr_Format(PyExc_TypeError, "Keyword '%s' must be numeric", key);
    return default_value;
}

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = (int)get_long(kwargs, "map_width", 0, true);
    if (PyErr_Occurred()) return 1;
    env->height = (int)get_long(kwargs, "map_height", 0, true);
    if (PyErr_Occurred()) return 1;
    env->num_agents = (int)get_long(kwargs, "num_agents", 0, true);
    if (PyErr_Occurred()) return 1;
    env->vision = (int)get_long(kwargs, "vision_size", 0, true);
    if (PyErr_Occurred()) return 1;
    if (env->vision < 3) {
        env->vision = 3;
    }
    env->spawn_separation = (int)get_long(kwargs, "spawn_separation", 3, true);
    if (PyErr_Occurred()) return 1;
    env->spawn_margin = (int)get_long(kwargs, "spawn_margin", 1, false);
    if (PyErr_Occurred()) return 1;
    env->spawn_attempts = (int)get_long(kwargs, "spawn_attempts", 0, false);
    if (PyErr_Occurred()) return 1;
    env->max_round_steps = (int)get_long(kwargs, "max_round_steps", 0, false);
    if (PyErr_Occurred()) return 1;

    env->reward_alive = (float)get_double(kwargs, "reward_alive", 0.0, false);
    if (PyErr_Occurred()) return 1;
    env->reward_step = (float)get_double(kwargs, "reward_step", 0.0, false);
    if (PyErr_Occurred()) return 1;
    env->reward_win = (float)get_double(kwargs, "reward_win", 1.0, false);
    if (PyErr_Occurred()) return 1;
    env->reward_loss = (float)get_double(kwargs, "reward_loss", -1.0, false);
    if (PyErr_Occurred()) return 1;
    env->reward_tie = (float)get_double(kwargs, "reward_tie", 0.0, false);
    if (PyErr_Occurred()) return 1;
    env->reward_timeout = (float)get_double(kwargs, "reward_timeout", 0.0, false);
    if (PyErr_Occurred()) return 1;
    env->reward_kill = (float)get_double(kwargs, "reward_kill", 0.0, false);
    if (PyErr_Occurred()) return 1;

    PyObject* masks_ptr = PyDict_GetItemString(kwargs, "masks_ptr");
    env->masks = NULL;
    if (masks_ptr && PyLong_Check(masks_ptr)) {
        env->masks = (unsigned char*)PyLong_AsVoidPtr(masks_ptr);
    }

    init(env);
    c_reset(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "normalized_episode_length", log->normalized_episode_length);
    assign_to_dict(dict, "avg_survival_time", log->avg_survival_time);
    assign_to_dict(dict, "avg_alive_agents", log->avg_alive_agents);
    return 0;
}

static PyObject* my_get(PyObject* dict, Env* env) {
    PyObject* width = PyLong_FromLong(env->width);
    PyDict_SetItemString(dict, "width", width);
    Py_DECREF(width);

    PyObject* height = PyLong_FromLong(env->height);
    PyDict_SetItemString(dict, "height", height);
    Py_DECREF(height);

    PyObject* agents = PyLong_FromLong(env->num_agents);
    PyDict_SetItemString(dict, "num_agents", agents);
    Py_DECREF(agents);

    PyObject* steps = PyLong_FromLong(env->steps_in_round);
    PyDict_SetItemString(dict, "steps_in_round", steps);
    Py_DECREF(steps);

    PyObject* trails = PyBytes_FromStringAndSize((const char*)env->trail_owner, env->width * env->height);
    PyDict_SetItemString(dict, "trail_owner", trails);
    Py_DECREF(trails);

    PyObject* heads = PyBytes_FromStringAndSize((const char*)env->head_owner, env->width * env->height);
    PyDict_SetItemString(dict, "head_owner", heads);
    Py_DECREF(heads);

    PyObject* crashes = PyBytes_FromStringAndSize((const char*)env->crash_map, env->width * env->height);
    PyDict_SetItemString(dict, "crash_map", crashes);
    Py_DECREF(crashes);

    PyObject* alive = PyBytes_FromStringAndSize((const char*)env->alive, env->num_agents);
    PyDict_SetItemString(dict, "alive", alive);
    Py_DECREF(alive);

    return dict;
}
