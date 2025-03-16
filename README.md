# Extended Python MARL framework - EPyMARL

details here
https://github.com/uoe-agents/epymarl

Added melting-pot clean_up

use
```
python src/main.py --config=qmix --env-config=meltingpo with env_args.substrate_name='clean_up'
```

clean_up config:
```yaml
env_args:
  render_observation: "WORLD.RGB" # 
  config_overrides: {} # useless now
  game_ascii_map: None
  substrate_name: "clean_up"
  time_limit: 200
  fps: 8
  action_possibility: 0.1
  action_latency: 5
```