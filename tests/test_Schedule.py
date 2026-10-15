#!/usr/bin/env python
# -*- coding utf-8 -*-

"""
Tests for BAC0/core/functions/Schedule.py (weekly occupancy schedules).

Pure encode/decode tests instantiate Schedule() directly. Only the send/read
integration tests use the session-scoped ``network_and_devices`` fixture.
Suspected library bugs are documented (not fixed) with xfail or
current-behaviour assertions.
"""

import asyncio
import copy
import gc
import inspect
import warnings

import pytest
from bacpypes3.apdu import WritePropertyRequest
from bacpypes3.basetypes import (
    DailySchedule,
    DeviceObjectPropertyReference,
    Reliability,
    TimeValue,
)
from bacpypes3.constructeddata import Any, AnyAtomic
from bacpypes3.object import ScheduleObject
from bacpypes3.pdu import Address
from bacpypes3.primitivedata import (
    Enumerated,
    Integer,
    ObjectIdentifier,
    ObjectType,
    PropertyIdentifier,
    Real,
    Unsigned,
)

from BAC0.core.functions.Schedule import Schedule

DAYS = Schedule.days
MULTISTATE_STATES = list(Schedule.schedule_example_multistate["states"].keys())
BINARY_STATES = ["inactive", "active"]
DEST = "127.0.0.1:47809"


def _week(events_per_day, states="analog"):
    return {"states": states, "week": {d: list(events_per_day) for d in DAYS}}


def _ref(obj_type, instance):
    return DeviceObjectPropertyReference(
        objectIdentifier=(obj_type, instance), propertyIdentifier="present-value"
    )


# ---------------------------------------------------------------------------
# create_weeklySchedule
# ---------------------------------------------------------------------------


def test_create_weeklySchedule_multistate_uses_enumerated():
    ws = Schedule().create_weeklySchedule(Schedule.schedule_example_multistate)
    assert isinstance(ws, Schedule.WeeklySchedule)
    assert len(ws) == 7
    for i, day in enumerate(DAYS):
        expected = Schedule.schedule_example_multistate["week"][day]
        daily = ws[i]
        assert isinstance(daily, DailySchedule)
        assert len(daily.daySchedule) == len(expected)
        for tv, (time_str, state) in zip(daily.daySchedule, expected, strict=False):
            hour, minute = (int(x) for x in time_str.split(":"))
            assert tuple(tv.time) == (hour, minute, 0, 0)
            assert isinstance(tv.value, AnyAtomic)
            value = tv.value.get_value()
            assert type(value) is Enumerated
            assert value == Schedule.schedule_example_multistate["states"][state]


def test_create_weeklySchedule_binary_uses_integer_zero_one():
    ws = Schedule().create_weeklySchedule(Schedule.schedule_example_binary)
    monday = ws[0].daySchedule
    assert [type(tv.value.get_value()) for tv in monday] == [Integer, Integer]
    # "active" -> 1, "inactive" -> 0
    assert [tv.value.get_value() for tv in monday] == [1, 0]
    for i, day in enumerate(DAYS):
        for tv, (_, state) in zip(
            ws[i].daySchedule,
            Schedule.schedule_example_binary["week"][day],
            strict=False,
        ):
            assert tv.value.get_value() == BINARY_STATES.index(state)


def test_create_weeklySchedule_binary_detection_is_order_insensitive():
    sched = copy.deepcopy(Schedule.schedule_example_binary)
    sched["states"] = {"active": 1, "inactive": 0}
    ws = Schedule().create_weeklySchedule(sched)
    assert type(ws[0].daySchedule[0].value.get_value()) is Integer


def test_create_weeklySchedule_two_state_non_binary_names_use_enumerated():
    """Only the exact {"inactive": 0, "active": 1} mapping is treated as binary."""
    sched = _week([("8:00", "on"), ("17:00", "off")], states={"off": 0, "on": 1})
    ws = Schedule().create_weeklySchedule(sched)
    values = [tv.value.get_value() for tv in ws[0].daySchedule]
    assert [type(v) for v in values] == [Enumerated, Enumerated]
    assert values == [1, 0]


@pytest.mark.parametrize("states", ["analog", "Analog", "ANALOG"])
def test_create_weeklySchedule_analog_uses_real(states):
    sched = copy.deepcopy(Schedule.schedule_example_analog)
    sched["states"] = states
    ws = Schedule().create_weeklySchedule(sched)
    for i, day in enumerate(DAYS):
        for tv, (_, value) in zip(
            ws[i].daySchedule,
            Schedule.schedule_example_analog["week"][day],
            strict=False,
        ):
            assert type(tv.value.get_value()) is Real
            assert tv.value.get_value() == pytest.approx(float(value))


def test_create_weeklySchedule_midnight_and_end_of_day_events():
    sched = _week([("0:00", 21.5), ("23:59", 18)])
    ws = Schedule().create_weeklySchedule(sched)
    times = [tuple(tv.time) for tv in ws[6].daySchedule]
    assert times == [(0, 0, 0, 0), (23, 59, 0, 0)]


def test_create_weeklySchedule_empty_days():
    sched = _week([])
    sched["week"]["wednesday"] = [("7:00", 22)]
    ws = Schedule().create_weeklySchedule(sched)
    assert [len(ws[i].daySchedule) for i in range(7)] == [0, 0, 1, 0, 0, 0, 0]


def test_create_weeklySchedule_stores_object_reference():
    s = Schedule()
    sched = copy.deepcopy(Schedule.schedule_example_analog)
    s.create_weeklySchedule(sched, object_reference="schedule:42")
    try:
        assert s.schedules["schedule:42"] is sched
        # ``schedules`` is a class attribute, so it is shared by all instances.
        assert Schedule().schedules["schedule:42"] is sched
    finally:
        Schedule.schedules.pop("schedule:42", None)


def test_create_weeklySchedule_without_object_reference_does_not_store():
    before = dict(Schedule.schedules)
    Schedule().create_weeklySchedule(Schedule.schedule_example_analog)
    assert Schedule.schedules == before


def test_create_weeklySchedule_missing_day_raises_keyerror():
    """
    Suspected bug (current behaviour asserted): every one of the 7 days must
    be present in ``week``. A schedule that omits a day (e.g. a weekday-only
    dict without "saturday"/"sunday") fails with a bare KeyError
    (Schedule.py line 101) instead of treating the day as empty or raising
    a descriptive error.
    """
    sched = copy.deepcopy(Schedule.schedule_example_analog)
    del sched["week"]["saturday"]
    del sched["week"]["sunday"]
    with pytest.raises(KeyError, match="saturday"):
        Schedule().create_weeklySchedule(sched)


def test_create_weeklySchedule_unknown_state_name_raises_keyerror():
    sched = _week([("8:00", "Occupied"), ("17:00", "Typo")], states={"Occupied": 1})
    with pytest.raises(KeyError, match="Typo"):
        Schedule().create_weeklySchedule(sched)


@pytest.mark.parametrize("states", ["digital", "", "occupancy"])
def test_create_weeklySchedule_malformed_string_states_yield_none_values(states):
    """
    Suspected bug (current behaviour asserted): ``_set_value`` (Schedule.py
    lines 89-97) only handles "analog" or a dict. Any other string makes it
    fall through and return None, so every TimeValue silently gets
    ``value=None`` instead of an error being raised.
    """
    sched = _week([("8:00", 1), ("17:00", 0)], states=states)
    ws = Schedule().create_weeklySchedule(sched)
    for i in range(7):
        for tv in ws[i].daySchedule:
            assert tv.value is None


def test_create_weeklySchedule_none_states_raises_typeerror():
    sched = _week([("8:00", 1)], states=None)
    with pytest.raises(TypeError):
        Schedule().create_weeklySchedule(sched)


# ---------------------------------------------------------------------------
# decode_dailySchedule / decode_weeklySchedule
# ---------------------------------------------------------------------------


def _expected_week(template, value_map=lambda v: v):
    return {
        day: [("{:02d}:{}".format(*_hm(t)), value_map(v)) for t, v in events]
        for day, events in template["week"].items()
    }


def _hm(time_str):
    hour, minute = time_str.split(":")
    return int(hour), minute


def test_decode_weeklySchedule_multistate_round_trip_offset_1():
    s = Schedule()
    ws = s.create_weeklySchedule(Schedule.schedule_example_multistate)
    week = s.decode_weeklySchedule(ws, MULTISTATE_STATES, offset_MV=1)
    assert list(week.keys()) == DAYS
    assert week == _expected_week(Schedule.schedule_example_multistate)


def test_decode_weeklySchedule_multistate_offset_0_shifts_states():
    """With offset_MV=0 the 1-based enumeration values map one state later."""
    s = Schedule()
    ws = s.create_weeklySchedule(Schedule.schedule_example_multistate)
    week = s.decode_weeklySchedule(ws, MULTISTATE_STATES, offset_MV=0)
    assert week["monday"] == [("01:00", "UnOccupied"), ("17:00", "Standby")]


def test_decode_weeklySchedule_binary_round_trip():
    s = Schedule()
    ws = s.create_weeklySchedule(Schedule.schedule_example_binary)
    # offset_MV is ignored for binary schedules
    for offset in (0, 1):
        week = s.decode_weeklySchedule(ws, BINARY_STATES, offset_MV=offset)
        assert week == _expected_week(Schedule.schedule_example_binary)


def test_decode_weeklySchedule_analog_round_trip():
    s = Schedule()
    ws = s.create_weeklySchedule(Schedule.schedule_example_analog)
    week = s.decode_weeklySchedule(ws, "analog", offset_MV=0)
    assert week == _expected_week(Schedule.schedule_example_analog, float)
    assert all(isinstance(v, float) for ev in week.values() for _, v in ev)


def test_decode_weeklySchedule_unlinked_schedule_uses_raw_values():
    """read_weeklySchedule passes range(255) when no object is referenced."""
    s = Schedule()
    ws = s.create_weeklySchedule(Schedule.schedule_example_multistate)
    week = s.decode_weeklySchedule(ws, range(255), offset_MV=0)
    assert week["friday"] == [("05:00", 1), ("17:00", 2)]


def test_decode_dailySchedule_empty_day():
    s = Schedule()
    assert s.decode_dailySchedule(DailySchedule(daySchedule=[]), "analog", 0) == []
    ws = s.create_weeklySchedule(_week([]))
    assert s.decode_weeklySchedule(ws, "analog", 0) == {d: [] for d in DAYS}


def test_decode_dailySchedule_midnight_event():
    s = Schedule()
    ws = s.create_weeklySchedule(_week([("0:00", 20), ("0:30", 21)]))
    assert s.decode_dailySchedule(ws[0], "analog", 0) == [
        ("00:00", 20.0),
        ("00:30", 21.0),
    ]


def test_decode_dailySchedule_out_of_range_multistate_falls_back():
    s = Schedule()
    daily = DailySchedule(
        daySchedule=[
            TimeValue(time="8:00", value=Enumerated(2)),
            TimeValue(time="9:00", value=Enumerated(9)),
        ]
    )
    assert s.decode_dailySchedule(daily, ["Occupied", "UnOccupied"], 1) == [
        ("08:00", "UnOccupied"),
        ("09:00", "??? (9)"),
    ]


def test_decode_dailySchedule_out_of_range_binary_falls_back():
    s = Schedule()
    daily = DailySchedule(daySchedule=[TimeValue(time="8:00", value=Integer(2))])
    assert s.decode_dailySchedule(daily, BINARY_STATES, 0) == [("08:00", "??? (2)")]


@pytest.mark.xfail(
    strict=False,
    reason="Suspected bug: value 0 with offset_MV=1 indexes states[-1] (Schedule.py line 264)",
)
def test_decode_dailySchedule_zero_with_offset_should_fall_back():
    """
    Suspected bug: multistate values are 1-based, so a value of 0 is invalid.
    With offset_MV=1 the code computes ``states[0 - 1]``; Python negative
    indexing silently returns the *last* state instead of hitting the
    "??? (0)" fallback (Schedule.py line 264).
    """
    s = Schedule()
    daily = DailySchedule(daySchedule=[TimeValue(time="8:00", value=Enumerated(0))])
    assert s.decode_dailySchedule(daily, MULTISTATE_STATES, 1) == [("08:00", "??? (0)")]


def test_decode_dailySchedule_zero_with_offset_current_behaviour():
    """Current behaviour of the bug above: the last state name is returned."""
    s = Schedule()
    daily = DailySchedule(daySchedule=[TimeValue(time="8:00", value=Enumerated(0))])
    assert s.decode_dailySchedule(daily, MULTISTATE_STATES, 1) == [("08:00", "Not Set")]


def test_decode_weeklySchedule_short_array_raises_indexerror():
    s = Schedule()
    ws = s.create_weeklySchedule(Schedule.schedule_example_analog)
    short = [ws[i] for i in range(5)]
    with pytest.raises(IndexError):
        s.decode_weeklySchedule(short, "analog", 0)


# ---------------------------------------------------------------------------
# make_weeklySchedule_request / send_weeklyschedule_request / write_weeklySchedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "template",
    [
        Schedule.schedule_example_multistate,
        Schedule.schedule_example_binary,
        Schedule.schedule_example_analog,
    ],
)
def test_make_weeklySchedule_request(template):
    s = Schedule()
    ws = s.create_weeklySchedule(template)
    request = s.make_weeklySchedule_request(DEST, 7, ws)
    assert isinstance(request, WritePropertyRequest)
    assert request.objectIdentifier == ObjectIdentifier(("schedule", 7))
    assert request.objectIdentifier[0] == ObjectType("schedule")
    assert request.objectIdentifier[1] == 7
    assert request.propertyIdentifier == PropertyIdentifier("weeklySchedule")
    assert request.pduDestination == Address(DEST)
    assert request.priority == 15
    assert isinstance(request.propertyValue, Any)
    decoded = request.propertyValue.cast_out(Schedule.WeeklySchedule)
    assert len(decoded) == 7
    assert [len(d.daySchedule) for d in decoded] == [len(d.daySchedule) for d in ws]


def test_make_weeklySchedule_request_encodes():
    s = Schedule()
    ws = s.create_weeklySchedule(Schedule.schedule_example_multistate)
    apdu = s.make_weeklySchedule_request(DEST, 1, ws).encode()
    assert apdu is not None


class _FakeApp:
    def __init__(self):
        self.requests = []

    def request(self, request):
        self.requests.append(request)


class _FakeBAC0App:
    def __init__(self):
        self.app = _FakeApp()


def _offline_schedule():
    s = Schedule()
    s.this_application = _FakeBAC0App()
    return s


def test_send_weeklyschedule_request_hands_request_to_app():
    s = _offline_schedule()
    request = s.make_weeklySchedule_request(
        DEST, 3, s.create_weeklySchedule(Schedule.schedule_example_analog)
    )
    assert s.send_weeklyschedule_request(request) is None
    assert s.this_application.app.requests == [request]


def test_send_weeklyschedule_request_timeout_is_unused():
    """
    Suspected bug (current behaviour asserted): ``send_weeklyschedule_request``
    accepts ``timeout`` (Schedule.py line 121) but never uses it. The request
    is fire-and-forget: nothing waits for the device's acknowledgement, so
    callers cannot tell whether the write was accepted.
    """
    assert (
        inspect.signature(Schedule.send_weeklyschedule_request)
        .parameters["timeout"]
        .default
        == 10
    )
    s = _offline_schedule()
    request = s.make_weeklySchedule_request(
        DEST, 3, s.create_weeklySchedule(Schedule.schedule_example_analog)
    )
    result = s.send_weeklyschedule_request(request, timeout=0)
    assert result is None and not inspect.isawaitable(result)
    assert len(s.this_application.app.requests) == 1


def test_write_weeklySchedule_builds_and_sends_request():
    s = _offline_schedule()
    s.write_weeklySchedule(DEST, 12, Schedule.schedule_example_binary)
    (request,) = s.this_application.app.requests
    assert request.objectIdentifier == ObjectIdentifier(("schedule", 12))
    assert request.propertyIdentifier == PropertyIdentifier("weeklySchedule")
    assert request.pduDestination == Address(DEST)
    assert request.priority == 15
    decoded = request.propertyValue.cast_out(Schedule.WeeklySchedule)
    assert s.decode_weeklySchedule(decoded, BINARY_STATES, 0) == _expected_week(
        Schedule.schedule_example_binary
    )


# ---------------------------------------------------------------------------
# read_weeklySchedule with fake read primitives (no network)
# ---------------------------------------------------------------------------


class _FakeReader(Schedule):
    """Schedule with stubbed readMultiple/read returning canned values."""

    def __init__(self, rpm_result=None, reads=None, rpm_exc=None, read_exc=None):
        self.rpm_result = rpm_result
        self.reads = reads or {}
        self.rpm_exc = rpm_exc
        self.read_exc = read_exc
        self.read_calls = []

    async def readMultiple(self, args):
        if self.rpm_exc:
            raise self.rpm_exc
        return self.rpm_result

    async def read(self, args):
        self.read_calls.append(args)
        if self.read_exc:
            raise self.read_exc
        return self.reads[args.split(" ", 2)[1] + " " + args.split(" ", 2)[2]]


def _rpm(template, refs, present_value, reliability="noFaultDetected"):
    ws = Schedule().create_weeklySchedule(template)
    return [ws, refs, Reliability(reliability), Unsigned(15), AnyAtomic(present_value)]


async def test_read_weeklySchedule_analog():
    s = _FakeReader(
        rpm_result=_rpm(
            Schedule.schedule_example_analog, [_ref("analog-value", 1)], Real(21.5)
        ),
        reads={"analog-value:1 objectName": "ZN-SP"},
    )
    result = await s.read_weeklySchedule("2:5", 1)
    assert result["states"] == "analog"
    assert result["presentValue"] == "21.5"
    assert result["priority"] == 15
    assert result["reliability"] == Reliability("noFaultDetected")
    assert result["object_references"] == [ObjectIdentifier(("analog-value", 1))]
    assert result["references_names"] == ["ZN-SP"]
    assert result["week"] == _expected_week(Schedule.schedule_example_analog, float)


async def test_read_weeklySchedule_binary():
    s = _FakeReader(
        rpm_result=_rpm(
            Schedule.schedule_example_binary,
            [_ref("binary-value", 3), _ref("binary-output", 4)],
            Integer(1),
        ),
        reads={
            "binary-value:3 objectName": "OCC-CMD",
            "binary-output:4 objectName": "AHU-SS",
        },
    )
    result = await s.read_weeklySchedule("2:5", 2)
    assert result["states"] == {"inactive": 0, "active": 1}
    assert result["presentValue"] == "active (1)"
    assert result["references_names"] == ["OCC-CMD", "AHU-SS"]
    assert result["week"] == _expected_week(Schedule.schedule_example_binary)


async def test_read_weeklySchedule_multistate_reads_state_text():
    s = _FakeReader(
        rpm_result=_rpm(
            Schedule.schedule_example_multistate,
            [_ref("multi-state-value", 5)],
            Unsigned(2),
        ),
        reads={
            "multi-state-value:5 stateText": MULTISTATE_STATES,
            "multi-state-value:5 objectName": "OCC-MODE",
        },
    )
    result = await s.read_weeklySchedule("2:5", 3)
    assert result["states"] == Schedule.schedule_example_multistate["states"]
    assert result["presentValue"] == "UnOccupied (2)"
    assert result["week"] == _expected_week(Schedule.schedule_example_multistate)
    # Output can be written back unchanged
    ws = Schedule().create_weeklySchedule(result)
    assert ws[0].daySchedule[0].value.get_value() == 1


async def test_read_weeklySchedule_unlinked_schedule():
    """No referenced object: offset 0, raw integers, empty references."""
    s = _FakeReader(
        rpm_result=_rpm(Schedule.schedule_example_multistate, [], Unsigned(1))
    )
    result = await s.read_weeklySchedule("2:5", 4)
    assert result["object_references"] == []
    assert result["references_names"] == []
    assert result["states"] == {i: i for i in range(255)}
    assert result["presentValue"] == "1 (1)"
    assert result["week"]["monday"] == [("01:00", 1), ("17:00", 2)]


async def test_read_weeklySchedule_unknown_reference_type_crashes():
    """
    Suspected bug (current behaviour asserted): only binary, multi-state and
    analog references are recognised (Schedule.py lines 184-191). For any
    other referenced type (e.g. integer-value, positive-integer-value) the
    states stay None and decode_dailySchedule crashes with TypeError at
    Schedule.py line 264 instead of returning raw values.
    """
    s = _FakeReader(
        rpm_result=_rpm(
            Schedule.schedule_example_analog, [_ref("integer-value", 1)], Integer(7)
        ),
        reads={"integer-value:1 objectName": "INT"},
    )
    with pytest.raises(TypeError, match="not subscriptable"):
        await s.read_weeklySchedule("2:5", 5)


@pytest.mark.xfail(
    strict=False,
    reason="Suspected bug: unsupported reference types leave states=None (Schedule.py lines 184-191, 264)",
)
async def test_read_weeklySchedule_integer_value_reference_should_decode():
    s = _FakeReader(
        rpm_result=_rpm(
            Schedule.schedule_example_analog, [_ref("integer-value", 1)], Integer(7)
        ),
        reads={"integer-value:1 objectName": "INT"},
    )
    result = await s.read_weeklySchedule("2:5", 5)
    assert len(result["week"]["monday"]) == 2


class _TrackedAwaitable:
    """Awaitable stand-in for a read() result that records whether it was awaited."""

    def __init__(self, value):
        self.value = value
        self.awaited = False

    def __await__(self):
        self.awaited = True
        if False:  # pragma: no cover - makes this a generator
            yield
        return self.value


class _AwaitableReader(_FakeReader):
    """read() returns awaitables (like the real async read) without coroutines."""

    def read(self, args):
        key = args.split(" ", 2)[1] + " " + args.split(" ", 2)[2]
        awaitable = _TrackedAwaitable(self.reads.get(key))
        self.read_calls.append(awaitable)
        return awaitable


async def test_read_weeklySchedule_fallback_reads_are_not_awaited():
    """
    Suspected bug (current behaviour asserted): when ReadPropertyMultiple
    fails, the single-read fallback (Schedule.py lines 164-172) calls the
    async ``self.read`` without ``await``. The variables receive un-awaited
    awaitables and the method then crashes with TypeError at line 179
    (``len(object_references)``) instead of returning the schedule.
    """
    s = _AwaitableReader(rpm_exc=RuntimeError("RPM not supported"))
    with pytest.raises(TypeError, match="has no len"):
        await s.read_weeklySchedule("2:5", 1)
    assert len(s.read_calls) == 5
    assert not any(a.awaited for a in s.read_calls)


@pytest.mark.xfail(
    strict=False,
    reason="Suspected bug: single-read fallback is not awaited (Schedule.py lines 164-172)",
)
async def test_read_weeklySchedule_fallback_should_work_without_rpm():
    """Desired behaviour: a device without RPM support is still readable."""
    template = Schedule.schedule_example_analog
    ws = Schedule().create_weeklySchedule(template)
    reads = {
        "schedule:1 weeklySchedule": ws,
        "schedule:1 listOfObjectPropertyReferences": [_ref("analog-value", 1)],
        "schedule:1 reliability": Reliability("noFaultDetected"),
        "schedule:1 priorityForWriting": 15,
        "schedule:1 presentValue": AnyAtomic(Real(21.0)),
        "analog-value:1 objectName": "ZN-SP",
    }
    s = _AwaitableReader(rpm_exc=RuntimeError("RPM not supported"), reads=reads)
    result = await s.read_weeklySchedule("2:5", 1)
    assert result["week"] == _expected_week(template, float)


class _SyncFailingReader(_FakeReader):
    def read(self, args):  # synchronous, raises immediately
        raise ValueError("device unreachable")


async def test_read_weeklySchedule_fallback_bare_raise_tuple():
    """
    Suspected bug (current behaviour asserted): if the fallback reads raise,
    Schedule.py line 175 executes ``raise ()``. Raising a tuple is itself a
    TypeError ("exceptions must derive from BaseException"), so the real
    cause is replaced by a confusing error. A bare ``raise`` was probably
    intended.
    """
    s = _SyncFailingReader(rpm_exc=RuntimeError("RPM not supported"))
    with pytest.raises(TypeError, match="exceptions must derive from BaseException"):
        await s.read_weeklySchedule("2:5", 1)


@pytest.mark.xfail(
    strict=False, reason="Suspected bug: `raise ()` at Schedule.py line 175"
)
async def test_read_weeklySchedule_fallback_should_propagate_original_error():
    s = _SyncFailingReader(rpm_exc=RuntimeError("RPM not supported"))
    with pytest.raises(ValueError, match="device unreachable"):
        await s.read_weeklySchedule("2:5", 1)


# ---------------------------------------------------------------------------
# Integration with the virtual devices (network_and_devices fixture)
# ---------------------------------------------------------------------------

SCHEDULE_INSTANCES = {"analog": 9101, "binary": 9102, "multistate": 9103}


def _device_address(device_app):
    return f"{device_app.localIPAddr.addrTuple[0]}:47809"


def _get_or_add_schedule(app, instance, reference, default):
    oid = ObjectIdentifier(("schedule", instance))
    obj = app.get_object_id(oid)
    if obj is None:
        obj = ScheduleObject(
            objectIdentifier=oid,
            objectName=f"TEST-SCHEDULE-{instance}",
            presentValue=default,
            scheduleDefault=default,
            listOfObjectPropertyReferences=[reference],
            priorityForWriting=15,
            reliability="noFaultDetected",
            statusFlags=[0, 0, 0, 0],
            outOfService=False,
        )
        app.add_object(obj)
    return obj


def _multistate_with_state_text(app):
    big_alarm = app.get_object_name("BIG-ALARM")
    return big_alarm.objectIdentifier, list(big_alarm.stateText)


@pytest.mark.asyncio(loop_scope="session")
async def test_write_weeklySchedule_to_virtual_device_does_not_raise(
    network_and_devices,
):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    app = device_app.this_application.app
    _get_or_add_schedule(
        app, SCHEDULE_INSTANCES["analog"], _ref("analog-value", 1), Real(21.0)
    )
    address = _device_address(device_app)
    for template in (
        Schedule.schedule_example_analog,
        Schedule.schedule_example_binary,
        Schedule.schedule_example_multistate,
    ):
        bacnet.write_weeklySchedule(address, SCHEDULE_INSTANCES["analog"], template)
    await asyncio.sleep(0.5)


@pytest.mark.xfail(
    strict=False,
    reason="Suspected bug: priority=15 on non-commandable weeklySchedule is rejected (Schedule.py line 118)",
)
@pytest.mark.asyncio(loop_scope="session")
async def test_write_weeklySchedule_is_applied_on_virtual_device(network_and_devices):
    """
    Suspected bug: ``make_weeklySchedule_request`` always sets priority=15,
    but weeklySchedule is not a commandable property. The bacpypes3 device
    rejects the write ("property has no priority"), and because sending is
    fire-and-forget the caller is never told. The schedule is not stored.
    """
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    app = device_app.this_application.app
    obj = _get_or_add_schedule(
        app, SCHEDULE_INSTANCES["analog"], _ref("analog-value", 1), Real(21.0)
    )
    obj.weeklySchedule = None
    bacnet.write_weeklySchedule(
        _device_address(device_app),
        SCHEDULE_INSTANCES["analog"],
        Schedule.schedule_example_analog,
    )
    await asyncio.sleep(1)
    assert obj.weeklySchedule is not None
    assert Schedule().decode_weeklySchedule(
        obj.weeklySchedule, "analog", 0
    ) == _expected_week(Schedule.schedule_example_analog, float)


@pytest.mark.asyncio(loop_scope="session")
async def test_weeklySchedule_without_priority_is_applied(network_and_devices):
    """The request body itself is valid: without priority it is accepted."""
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    app = device_app.this_application.app
    obj = _get_or_add_schedule(
        app, SCHEDULE_INSTANCES["analog"], _ref("analog-value", 1), Real(21.0)
    )
    obj.weeklySchedule = None
    request = bacnet.make_weeklySchedule_request(
        _device_address(device_app),
        SCHEDULE_INSTANCES["analog"],
        bacnet.create_weeklySchedule(Schedule.schedule_example_analog),
    )
    request.priority = None
    await bacnet.this_application.app.request(request)
    assert bacnet.decode_weeklySchedule(
        obj.weeklySchedule, "analog", 0
    ) == _expected_week(Schedule.schedule_example_analog, float)


@pytest.mark.asyncio(loop_scope="session")
async def test_read_weeklySchedule_analog_from_virtual_device(network_and_devices):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    app = device_app.this_application.app
    obj = _get_or_add_schedule(
        app, SCHEDULE_INSTANCES["analog"], _ref("analog-value", 1), Real(21.0)
    )
    obj.weeklySchedule = Schedule().create_weeklySchedule(
        Schedule.schedule_example_analog
    )
    result = await bacnet.read_weeklySchedule(
        _device_address(device_app), SCHEDULE_INSTANCES["analog"]
    )
    assert result["states"] == "analog"
    assert result["presentValue"] == "21.0"
    assert result["object_references"] == [ObjectIdentifier(("analog-value", 1))]
    assert result["references_names"] == ["AV-1"]
    assert result["week"] == _expected_week(Schedule.schedule_example_analog, float)


@pytest.mark.asyncio(loop_scope="session")
async def test_read_weeklySchedule_binary_from_virtual_device(network_and_devices):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    app = device_app.this_application.app
    obj = _get_or_add_schedule(
        app, SCHEDULE_INSTANCES["binary"], _ref("binary-value", 0), Integer(1)
    )
    obj.weeklySchedule = Schedule().create_weeklySchedule(
        Schedule.schedule_example_binary
    )
    result = await bacnet.read_weeklySchedule(
        _device_address(device_app), SCHEDULE_INSTANCES["binary"]
    )
    assert result["states"] == {"inactive": 0, "active": 1}
    assert result["presentValue"] == "active (1)"
    assert result["references_names"] == ["BV"]
    assert result["week"] == _expected_week(Schedule.schedule_example_binary)


@pytest.mark.asyncio(loop_scope="session")
async def test_read_weeklySchedule_multistate_from_virtual_device(
    network_and_devices,
):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    app = device_app.this_application.app
    msv_id, state_text = _multistate_with_state_text(app)
    obj = _get_or_add_schedule(
        app,
        SCHEDULE_INSTANCES["multistate"],
        _ref(msv_id[0], msv_id[1]),
        Unsigned(1),
    )
    template = {
        "states": {name: i + 1 for i, name in enumerate(state_text)},
        "week": {d: [("0:00", state_text[0]), ("8:00", state_text[1])] for d in DAYS},
    }
    obj.weeklySchedule = Schedule().create_weeklySchedule(template)
    result = await bacnet.read_weeklySchedule(
        _device_address(device_app), SCHEDULE_INSTANCES["multistate"]
    )
    assert result["states"] == template["states"]
    assert result["presentValue"] == f"{state_text[0]} (1)"
    assert result["references_names"] == ["BIG-ALARM"]
    assert result["week"]["sunday"] == [
        ("00:00", state_text[0]),
        ("08:00", state_text[1]),
    ]


@pytest.mark.asyncio(loop_scope="session")
async def test_read_weeklySchedule_missing_object_raises(network_and_devices):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    """
    Reading a schedule that does not exist hits the broken single-read
    fallback (see test_read_weeklySchedule_fallback_reads_are_not_awaited):
    the caller gets a TypeError about coroutines rather than an
    "unknown object" error.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(TypeError, match="coroutine"):
            await bacnet.read_weeklySchedule(_device_address(device_app), 9199)
        gc.collect()
