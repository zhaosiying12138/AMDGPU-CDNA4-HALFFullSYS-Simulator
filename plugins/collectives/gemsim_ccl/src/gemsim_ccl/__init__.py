"""Functional collectives for the gemsim_amd Triton backend."""

__all__ = [
    "AllReduceEngine",
    "AllReduceSegment",
    "CollectiveEvent",
    "CollectiveTimeoutError",
    "DeviceSumCounters",
    "DeviceSumExecutor",
    "EngineError",
    "EngineForkError",
    "EngineState",
    "EngineStateError",
    "GroupAbortedError",
    "GroupSpec",
    "RankBootstrap",
    "SequenceExhaustedError",
    "TransferInfo",
    "plan_allreduce_segments",
]
__version__ = "0.1.0"


def __getattr__(name: str):
    if name in ("DeviceSumCounters", "DeviceSumExecutor"):
        from .device import DeviceSumCounters, DeviceSumExecutor

        return {
            "DeviceSumCounters": DeviceSumCounters,
            "DeviceSumExecutor": DeviceSumExecutor,
        }[name]
    if name in __all__:
        from .engine import (
            AllReduceEngine,
            AllReduceSegment,
            CollectiveEvent,
            CollectiveTimeoutError,
            EngineError,
            EngineForkError,
            EngineState,
            EngineStateError,
            GroupAbortedError,
            GroupSpec,
            RankBootstrap,
            SequenceExhaustedError,
            TransferInfo,
            plan_allreduce_segments,
        )

        return {
            "AllReduceEngine": AllReduceEngine,
            "AllReduceSegment": AllReduceSegment,
            "CollectiveEvent": CollectiveEvent,
            "CollectiveTimeoutError": CollectiveTimeoutError,
            "EngineError": EngineError,
            "EngineForkError": EngineForkError,
            "EngineState": EngineState,
            "EngineStateError": EngineStateError,
            "GroupAbortedError": GroupAbortedError,
            "GroupSpec": GroupSpec,
            "RankBootstrap": RankBootstrap,
            "SequenceExhaustedError": SequenceExhaustedError,
            "TransferInfo": TransferInfo,
            "plan_allreduce_segments": plan_allreduce_segments,
        }[name]
    raise AttributeError(name)
