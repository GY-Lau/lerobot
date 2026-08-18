#!/usr/bin/env python3
"""Run lerobot's async policy server, reloading the checkpoint only when it changes.

`PolicyServer.SendPolicyInstructions` calls `from_pretrained` unconditionally, so
every client that connects pays the load again. On the Jetson that is 39 s for
SmolVLA, which is most of a trial. Nothing about it is per-client: the same
checkpoint on the same device produces the same policy.

This wraps that one method so an identical request reuses what is already
loaded, and leaves everything else upstream untouched. Start it once and run as
many trials as you like against it.

Reuse is safe here because the server is stateless between observations: it calls
`predict_action_chunk`, which maps one observation to one chunk. The action queue
that does carry state lives in the client. `reset()` is called anyway, so a
policy that keeps internal state does not carry it across trials.

Takes the same arguments as `python -m lerobot.async_inference.policy_server`.
"""

import pickle  # nosec  -- same trust boundary as the upstream server
import time

from lerobot.async_inference import policy_server as ps


def _cache_key(specs) -> tuple:
    return (
        specs.policy_type,
        specs.pretrained_name_or_path,
        specs.device,
        tuple(sorted(getattr(specs, "rename_map", {}).items())),
    )


_load_policy = ps.PolicyServer.SendPolicyInstructions


def _reuse_if_unchanged(self, request, context):
    if not self.running:
        return _load_policy(self, request, context)

    specs = pickle.loads(request.data)  # nosec
    key = _cache_key(specs)

    if getattr(self, "_loaded_key", None) == key and self.policy is not None:
        # Fields the client may legitimately vary without needing a reload.
        self.lerobot_features = specs.lerobot_features
        self.actions_per_chunk = specs.actions_per_chunk
        start = time.perf_counter()
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self.logger.info(
            f"Reusing the policy already loaded from {specs.pretrained_name_or_path} "
            f"({(time.perf_counter() - start) * 1000:.0f} ms instead of a full load)"
        )
        return ps.services_pb2.Empty()

    result = _load_policy(self, request, context)
    self._loaded_key = key
    return result


ps.PolicyServer.SendPolicyInstructions = _reuse_if_unchanged

if __name__ == "__main__":
    ps.serve()
