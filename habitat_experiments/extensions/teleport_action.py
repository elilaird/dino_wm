from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
from habitat.tasks.nav.nav import SimulatorTaskAction
from habitat.core.registry import registry


@dataclass
class TeleportActuationSpec:
    # no params for now; behavior driven by episode metadata
    pass


@registry.register_task_action
class TeleportAction(SimulatorTaskAction):
    name: str = "TELEPORT"

    def reset(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        sim = self._sim
        # Get episode-defined teleport links
        ep = self._dataset.episodes[self._dataset.current_episode_index]
        links: Dict[str, Dict[str, Any]] = getattr(ep, "teleport_links", {})

        # Compute current cell/key (e.g., round agent pos to grid cell or region id)
        cur_pos = sim.get_agent_state().position
        # Expect links like: {"roomA":{"aabb":[minx,miny,minz,maxx,maxy,maxz], "dest":[x,y,z, qw,qx,qy,qz]}, ...}
        for _, rec in links.items():
            aabb = rec["aabb"]
            mn = np.array(aabb[:3])
            mx = np.array(aabb[3:])
            if np.all(cur_pos >= mn) and np.all(cur_pos <= mx):
                dest = rec["dest"]
                pos = np.array(dest[:3], dtype=np.float32)
                quat = np.array(dest[3:], dtype=np.float32)
                sim.set_agent_state(pos, quat)  # warp
                break
