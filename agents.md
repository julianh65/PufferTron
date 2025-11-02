# Building an Ocean Environment (Quick Tutorial)

This is the checklist we follow when adding a new native environment to the `pufferlib.ocean` package. The goal is to end up with a PufferEnv class you can train with `puffer train puffer_<env>` and visualise via `puffer eval`.

---

## 1. Project setup

```bash
pip install -e .                    # editable install of PufferLib
pip install -r requirements-dev.txt # optional dev extras
```

Create the directory `pufferlib/ocean/<env>/`.

Choose an env name of the form `puffer_<env>`; this is the name users will pass to the CLI.

---

## 2. Native environment implementation

Inside `pufferlib/ocean/<env>/` add:

| File | Purpose |
|------|---------|
| `<env>.h` / `<env>.c(pp)` | Define the `struct C<Env>` containing pointers to observations/actions/rewards/terminals and any internal state. Implement `init`, `c_reset`, `c_step`, `c_render` (optional), `c_close`, and helpers. |
| `binding.c` | Include your header, `#define Env C<Env>`, then include `../env_binding.h`. Override `my_init`, `my_log`, and any optional hooks so the template knows how to initialise your struct and emit stats. |
| `<env>.py` | Thin Python wrapper subclassing `pufferlib.PufferEnv`. Set `single_observation_space`, `single_action_space`, `num_agents`, call `super().__init__`, then forward `reset`, `step`, `render`, `close` into the binding functions (`vec_init`, `vec_step`, ...). |

The binding template (`env_binding.h`) automatically allocates one `Env` struct per environment copy, hooks up shared-memory buffers, and exposes the Python C-extension API. Your `c_*` functions must update the struct and write outputs into those buffers.

---

## 3. Registering the environment

1. Edit `pufferlib/ocean/environment.py` and add your env to `MAKE_FUNCTIONS`, e.g.:
   ```python
   MAKE_FUNCTIONS = {
       # ...
       "tron": "Tron",
   }
   ```
   `env_creator('puffer_tron')` will now return your `Tron` class.

2. (Optional) Add `pufferlib/config/ocean/<env>.ini` with default `[vec]`, `[env]`, and `[train]` sections so users can run `puffer train puffer_<env>` without extra flags.

3. (Optional) Provide custom policies in `pufferlib/ocean/torch.py` (e.g. `class TronPolicy(nn.Module)`) and configure `policy_name` in your INI. Otherwise the default MLP (`pufferlib.models.Default`) is used.

---

## 4. Build the binding

Any time you touch the C/C++ files or `binding.c`, rebuild the extension:

```bash
python setup.py build_ext --inplace
# or
pip install .
```

This produces/updates `binding.cpython-*.so` which the Python wrapper imports.

---

## 5. Quick test loop

1. **Smoke-test training**
   ```bash
   python -m pufferlib.pufferl train puffer_<env> \
       --train-total-timesteps 100000 \
       --train-batch-size 2048
   ```

2. **Visual check / playthrough**
   ```bash
   python -m pufferlib.pufferl eval puffer_<env> \
       --render-mode human \
       --load-model-path experiments/puffer_<env>_*/*.pt
   ```
   Tweak `--render-mode` (`human`, `ansi`, `rgb_array`) to match what your `c_render` supports. Use `--save-frames N --gif-path out.gif` to record gameplay.

3. **Manual or scripted opponents**  
   Copy the loop in `eval` and replace the policy action with keyboard input or a hard-coded policy, then call `vecenv.step(action)` manually.

---

## 6. Optional sweeps

Once the env is stable, configure `[sweep]` in the INI (distribution, min/max, etc.), enable a logger (`--wandb` or `--neptune`), and launch a search:

```bash
python -m pufferlib.pufferl sweep puffer_<env> --wandb --wandb-project myenv
```

Choose the sweep algorithm by setting `method = Protein` (default GP-based) or `method = Carbs` (external CARBS optimiser).

---

## Summary

1. Implement the native env (`struct`, `c_step`, etc.) and binding hooks.  
2. Provide the Python wrapper (`PufferEnv` subclass).  
3. Register the env name and optional configs/policies.  
4. Rebuild the extension after native changes.  
5. Train (`puffer train`) and visualise (`puffer eval`) to verify behaviour.  
6. Use sweeps when you’re ready to tune hyperparameters.

Following these steps ensures the environment plugs cleanly into PufferLib’s vectorisation, policy loading, training, evaluation, and profiling pipelines.

