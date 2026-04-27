from __future__ import annotations

from .config import AppConfig
from .context import ContextBuilder
from .decision import DecisionLayer
from .encoders import FallbackVisionEncoder, build_encoder
from .memory import EventMemory
from .models import VerifierBundle
from .types import Episode, EventPacket, StepObservation, SUBGOALS, VerificationOutput
from .watcher import EventWatcher


class EventDrivenVerifierPipeline:
    def __init__(self, bundle: VerifierBundle, config: AppConfig | None = None) -> None:
        self.bundle = bundle
        self.config = config or AppConfig()
        self.encoder = None
        self.context_builder = ContextBuilder(window=self.config.memory.window)
        self.memory = EventMemory(capacity=self.config.memory.capacity, num_subgoals=bundle.num_subgoals, num_horizons=len(bundle.horizons))
        self.watcher = EventWatcher(self.config.watcher)
        self.decision = DecisionLayer(self.config.decision)
        self.terminated = False
        self.termination_timestamp: int | None = None
        self.terminal_decision: str | None = None
        self.termination_reason: str | None = None

    def reset(self) -> None:
        self.context_builder.reset()
        self.memory.reset()
        self.watcher.reset()
        self.decision.reset()
        self.terminated = False
        self.termination_timestamp = None
        self.terminal_decision = None
        self.termination_reason = None

    def status(self) -> dict[str, object]:
        return {
            'terminated': self.terminated,
            'termination_timestamp': self.termination_timestamp,
            'terminal_decision': self.terminal_decision,
            'termination_reason': self.termination_reason,
        }

    def _terminate(self, timestamp: int, decision: str, reason: str) -> None:
        self.terminated = True
        self.termination_timestamp = int(timestamp)
        self.terminal_decision = decision
        self.termination_reason = reason

    @staticmethod
    def _policy_uncertainty(observation: StepObservation) -> float:
        """Return a dedicated uncertainty scalar when one is explicitly provided.

        The public fixture stores generic policy statistics rather than a calibrated
        uncertainty estimate. To avoid treating action magnitude as epistemic risk,
        the pipeline only consumes an uncertainty value when the caller provides a
        fourth policy_stats entry dedicated to that purpose.
        """
        if len(observation.policy_stats) >= 4:
            return float(max(0.0, min(1.0, observation.policy_stats[3])))
        return 0.0

    def step(self, observation: StepObservation) -> VerificationOutput:
        if self.terminated:
            return VerificationOutput(
                timestamp=observation.timestamp,
                triggered=False,
                event_score=0.0,
                visual_drift=0.0,
                stall_score=0.0,
                uncertainty_score=0.0,
                high_stakes_score=0.0,
                heartbeat_due=False,
                decision='terminated',
                reason='autonomy has already been terminated; call reset() before continuing',
                terminated=True,
                terminal_decision=self.terminal_decision,
                termination_reason=self.termination_reason,
                termination_timestamp=self.termination_timestamp,
                post_termination=True,
            )
        if self.encoder is None:
            if observation.precomputed_vector is not None:
                self.encoder = FallbackVisionEncoder(self.config.encoder)
            else:
                self.encoder = build_encoder(self.config.encoder)
        snapshot = self.encoder.extract(observation)
        self.context_builder.update(snapshot)
        progress_proxy = self.context_builder.heuristic_progress()
        policy_uncertainty = self._policy_uncertainty(observation)
        watcher_state = self.watcher.step(
            timestamp=observation.timestamp,
            snapshot=snapshot,
            progress_proxy=progress_proxy,
            policy_uncertainty=policy_uncertainty,
            action_type=observation.action_type,
        )
        if not watcher_state['trigger']:
            return VerificationOutput(
                timestamp=observation.timestamp,
                triggered=False,
                event_score=float(watcher_state['score']),
                visual_drift=float(watcher_state['visual_drift']),
                stall_score=float(watcher_state['stall']),
                uncertainty_score=float(watcher_state['uncertainty']),
                high_stakes_score=float(watcher_state['high_stakes']),
                heartbeat_due=bool(watcher_state['heartbeat_due']),
                terminated=False,
            )
        context = self.context_builder.build(observation.timestamp, snapshot, self.memory)
        subgoal_pred, completion, done, risk = self.bundle.predict(context)
        subgoal_idx = int(subgoal_pred[0])
        completion_value = float(completion[0])
        done_probability = float(done[0])
        risk_value = risk[0].astype('float32')
        decision, reason = self.decision.decide(risk_value, policy_uncertainty, done_probability)
        if decision in set(self.config.decision.terminal_decisions):
            self._terminate(observation.timestamp, decision, reason)
        self.memory.add(
            EventPacket(
                timestamp=observation.timestamp,
                subgoal=subgoal_idx,
                completion=completion_value,
                done=done_probability,
                risk=risk_value,
                visual_embedding=snapshot.visual_embedding,
                note=decision,
            )
        )
        return VerificationOutput(
            timestamp=observation.timestamp,
            triggered=True,
            event_score=float(watcher_state['score']),
            visual_drift=float(watcher_state['visual_drift']),
            stall_score=float(watcher_state['stall']),
            uncertainty_score=float(watcher_state['uncertainty']),
            high_stakes_score=float(watcher_state['high_stakes']),
            heartbeat_due=bool(watcher_state['heartbeat_due']),
            subgoal=SUBGOALS[subgoal_idx],
            completion=completion_value,
            done_probability=done_probability,
            risk=risk_value,
            decision=decision,
            reason=reason,
            terminated=self.terminated,
            terminal_decision=self.terminal_decision,
            termination_reason=self.termination_reason,
            termination_timestamp=self.termination_timestamp,
        )

    def run_episode(self, episode: Episode, stop_on_termination: bool = True) -> dict[str, object]:
        self.reset()
        events: list[dict[str, object]] = []
        last_timestamp = episode.steps[-1].observation.timestamp if episode.steps else None
        for step in episode.steps:
            out = self.step(step.observation)
            if out.triggered or out.post_termination:
                events.append(
                    {
                        't': out.timestamp,
                        'triggered': out.triggered,
                        'subgoal': out.subgoal,
                        'completion': None if out.completion is None else round(out.completion, 3),
                        'risk': None if out.risk is None else [round(float(v), 3) for v in out.risk.tolist()],
                        'decision': out.decision,
                        'reason': out.reason,
                        'terminated': out.terminated,
                        'terminal_decision': out.terminal_decision,
                    }
                )
            if stop_on_termination and self.terminated:
                last_timestamp = step.observation.timestamp
                break
            last_timestamp = step.observation.timestamp

        lead_time: int | None = None
        if self.terminated and episode.failure_onset is not None and self.termination_timestamp is not None:
            lead_time = int(episode.failure_onset - self.termination_timestamp)
        return {
            'episode_outcome': episode.outcome,
            'failure_onset': episode.failure_onset,
            'terminated': self.terminated,
            'terminated_at': self.termination_timestamp,
            'terminal_decision': self.terminal_decision,
            'termination_reason': self.termination_reason,
            'lead_time_to_failure': lead_time,
            'last_processed_timestamp': last_timestamp,
            'events': events,
        }
