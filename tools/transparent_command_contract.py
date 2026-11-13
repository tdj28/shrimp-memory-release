"""Exact overt command contract and conventional digital-latch control.

This module neither imports a model runner nor executes numerical dynamics.
It validates a visible request before invoking a supplied dispatcher. The
contract is ordinary deterministic parsing, not semantic generalization.
"""

from collections.abc import Callable
from dataclasses import dataclass


class CommandRejected(ValueError):
    """The complete input is outside the declared two-message language."""


class ActionMismatch(ValueError):
    """A prepared action disagrees with the visible command or its schedule."""


@dataclass(frozen=True)
class Stage:
    name: str
    alpha: float
    log10_temperature: float
    steps: int


@dataclass(frozen=True)
class WriteAction:
    operation: str
    register: str
    bit: int
    attractor_label: str
    stages: tuple[Stage, Stage]


ZERO_MESSAGE = "Write bit 0 to register R0."
ONE_MESSAGE = "Write bit 1 to register R0."
HOLD = Stage("hold", 0.07293697809562831, -0.15395151820007266, 240)


def parse_command(text: str) -> WriteAction:
    """Recognize the whole string exactly, without stripping or repairing it."""
    if type(text) is not str:
        raise CommandRejected("input must be an exact command string")
    if text == ZERO_MESSAGE:
        bit, label = 0, "ad"
        endpoint = Stage("endpoint", 0.07293697809562831, -0.15350, 120)
    elif text == ONE_MESSAGE:
        bit, label = 1, "td"
        endpoint = Stage("endpoint", 0.07299785589927693, -0.15395151820007266, 120)
    else:
        raise CommandRejected("input is outside the exact two-message contract")
    return WriteAction("write", "R0", bit, label, (endpoint, HOLD))


def verify_action(text: str, action: WriteAction) -> None:
    """Bind a prepared action to its complete visible input before dispatch.

    Dataclass equality alone accepts booleans as integers. Check field types
    explicitly before comparing the public bit, label, controls and durations.
    This is a runtime guard; the test oracle is authored separately.
    """
    expected = parse_command(text)
    if (type(action) is not WriteAction
            or type(action.operation) is not str
            or type(action.register) is not str
            or type(action.bit) is not int
            or type(action.attractor_label) is not str
            or type(action.stages) is not tuple
            or len(action.stages) != 2):
        raise ActionMismatch("invalid typed action")
    for stage in action.stages:
        if (type(stage) is not Stage or type(stage.name) is not str
                or type(stage.alpha) is not float
                or type(stage.log10_temperature) is not float
                or type(stage.steps) is not int):
            raise ActionMismatch("invalid typed stage")
    if action != expected:
        raise ActionMismatch("action or schedule differs from the visible command")


def dispatch_command(
    text: str,
    dispatcher: Callable[[WriteAction], None],
    *,
    prepared_action: WriteAction | None = None,
) -> WriteAction:
    """Validate first, then call the supplied dispatcher exactly once.

    Rejected input and mismatched actions never reach the dispatcher. Dispatcher
    exceptions propagate; they are execution failures, not command rejections.
    No numerical dispatcher is supplied by this module.
    """
    parsed = parse_command(text)
    action = parsed if prepared_action is None else prepared_action
    verify_action(text, action)
    dispatcher(action)
    return action


class DigitalLatch:
    """One conventional bit using the same visible command contract.

    This positive control applies assignments immediately. It has no trajectory,
    attractor, geometric-target calculation or model-forward operation.
    """

    def __init__(self, initial_bit: int):
        if type(initial_bit) is not int or initial_bit not in (0, 1):
            raise ValueError("initial bit must be integer zero or one")
        self._state = bytearray([initial_bit])
        self._dispatch_count = 0

    @property
    def state_bytes(self) -> bytes:
        return bytes(self._state)

    @property
    def bit(self) -> int:
        return self._state[0]

    @property
    def dispatch_count(self) -> int:
        return self._dispatch_count

    def execute(self, text: str, *, prepared_action: WriteAction | None = None) -> WriteAction:
        def assign(action: WriteAction) -> None:
            self._state[0] = action.bit
            self._dispatch_count += 1

        return dispatch_command(text, assign, prepared_action=prepared_action)
