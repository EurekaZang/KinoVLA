---
name: isaac-lab-dev
description: Use when writing or debugging any Isaac Lab / Isaac Sim / PhysX code in this repo - env setup, headless launch, Go2 asset, contact materials, D6 joints, scripted collider swaps, determinism seeding, GPU parallel envs. Contains known pitfalls (no FEM softbody/rope/fluid at RL scale - use compliant contact / spring joints / pendulum payload per spec section 8.2).
---
# Isaac Lab Development Conventions
- Always launch headless for tests: pass --headless; assert seeded determinism via trajectory hash.
- Per-region friction: PhysX physics-material API (operator O1). Compliance: contact stiffness/damping, NOT FEM (O2).
- Dynamic attachments: create/destroy D6 spring-damper joints at contact events (O4); add break-force.
- Collider swap with hysteresis for collapse events (O3).
- Record privileged params via the uniform get_privileged_state() operator API; never hardcode theta.
- Machines: dev laptop = NO GPU (Isaac unavailable; sim tests auto-skip via tests/conftest.py). Sim box = RTX 5090, Blackwell sm_120 -> Isaac Sim >= 5.x + torch cu128 ONLY (Isaac Sim <= 4.5 lacks Blackwell kernels). Install pins in README "GPU machine setup".
- Local pytest: run with `PYTHONPATH= pytest ...` (ROS Humble py3.10 site-packages leak in via shell profile and break plugin autoload).
- Project env: conda `kinovla` (py3.11) — matches Isaac Sim 5.x python.
