#!/usr/bin/env python
# -*- coding utf-8 -*-

"""
Tests for BAC0.core.functions.Schedule (weekly occupancy schedules).

Unit tests instantiate ``Schedule`` directly and mock ``this_application`` /
``read`` / ``readMultiple``; the integration tests at the bottom use the
``network_and_devices`` fixture and a bacpypes3 ``ScheduleObject`` added to
``device_app`` from test code only.

Suspected bugs found while writing these tests (documented, NOT fixed here):

* ``read_weeklySchedule``: when the single-property fallback fails it executes
  ``raise ()``, which raises ``TypeError: exceptions must derive from
  BaseException`` and hides the real error. The fallback also calls the async
  ``self.read`` without ``await``, so it would return coroutines.
* ``decode_dailySchedule`` passes the BACnet *hundredths* of a second as the
  ``microsecond`` argument of ``datetime.time`` (50 hundredths -> 50 us).
  Harmless today because only ``%H:%M`` is printed, and a wildcard time
  (255) would raise ``ValueError``.
* ``create_weeklySchedule._set_value`` returns ``None`` for any string
  ``states`` other than ``"analog"`` (e.g. a typo), silently producing
  ``TimeValue``s with no value instead of raising.
* ``decode_dailySchedule`` treats only ``None`` as "no value", but a BACnet
  Null (relinquish) event decodes to ``()``, which crashes with ``TypeError``.
* ``send_weeklyschedule_request(..., timeout=10)``: ``timeout`` is unused; the
  request is fire-and-forget, so device errors are never reported.
* ``make_weeklySchedule_request`` always sets ``priority = 15``; weeklySchedule
  is not commandable and a bacpypes3 device rejects the write.
* ``Schedule.schedules`` is a class-level dict, so schedules stored via
  ``create_weeklySchedule(..., object_reference=...)`` are shared by every
  instance (every BAC0 network object in the process).
"""

import asyncio
import copy
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from bacpypes3.basetypes import (
    DailySchedule,
    DeviceObjectPropertyReference,
    EngineeringUnits,
    EventState,
    Reliability,
    StatusFlags,
    TimeValue,
)
from bacpypes3.constructeddata import Any, AnyAtomic
from bacpypes3.object import (
    AnalogValueObject,
    BinaryValueObject,
    MultiStateValueObject,
    ScheduleObject,
)
from bacpypes3.pdu import Address
from bacpypes3.primitivedata import (
    Enumerated,
    Integer,
    Null,
    ObjectIdentifier,
    ObjectType,
    PropertyIdentifier,
    Real,
    Unsigned,
)

from BAC0.core.devices.local.factory import ObjectFactory
from BAC0.core.functions.Schedule import Schedule

MULTISTATE = Schedule.schedule_example_multistate
BINARY = Schedule.schedule_example_binary
ANALOG = Schedule.schedule_example_analog

# states / offset_MV as read_weeklySchedule computes them for a linked schedule
DECODE_ARGS = {
    "multistate": (MULTISTATE, list(MULTISTATE["states"]), 1),
    "binary": (BINARY, ["inactive", "active"], 0),
    "analog": (ANALOG, "analog", 0),
}


@pytest.fixture
def sched():
    return Schedule()


def _week(events):
    return {day: list(events) for day in Schedule.days}


def _hhmm(time_str):
    hour, minute = (int(x) for x in time_str.split(":"))
    return f"{hour:02d}:{minute:02d}"


def _expected_week(dict_schedule):
    """The week dict read_weeklySchedule returns: zero-padded times, and floats
    for analog values."""
    analog = dict_schedule["states"] == "analog"
    return {
        day: [
            (_hhmm(t), float(v) if analog else v) for t, v in dict_schedule["week"][day]
        ]
        for day in Schedule.days
    }


def _daily(*events):
    return DailySchedule(daySchedule=[TimeValue(time=t, value=v) for t, v in events])


# ---------------------------------------------------------------------------
# create_weeklySchedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "example, value_type",
    [(MULTISTATE, Enumerated), (BINARY, Integer), (ANALOG, Real)],
    ids=["multistate", "binary", "analog"],
)
def test_create_weeklySchedule_value_types(sched, example, value_type):
    weekly = sched.create_weeklySchedule(example)
    assert isinstance(weekly, Schedule.WeeklySchedule)
    assert len(weekly) == 7
    for daily in weekly:
        assert len(daily.daySchedule) == 2
        for tv in daily.daySchedule:
            assert isinstance(tv.value.get_value(), value_type)


@pytest.mark.parametrize("example", [MULTISTATE, BINARY, ANALOG])
def test_create_weeklySchedule_days_in_order(sched, example):
    weekly = sched.create_weeklySchedule(example)
    # examples start Monday at 1:00, Tuesday 2:00 ... Sunday 7:00
    for i, daily in enumerate(weekly):
        assert tuple(daily.daySchedule[0].time) == (i + 1, 0, 0, 0)


def test_create_weeklySchedule_values(sched):
    weekly = sched.create_weeklySchedule(MULTISTATE)
    monday = weekly[0].daySchedule
    assert tuple(monday[1].time) == (17, 0, 0, 0)
    assert monday[0].value.get_value() == 1  # Occupied
    assert monday[1].value.get_value() == 2  # UnOccupied

    weekly = sched.create_weeklySchedule(BINARY)
    assert [tv.value.get_value() for tv in weekly[6].daySchedule] == [1, 0]

    weekly = sched.create_weeklySchedule(ANALOG)
    assert [tv.value.get_value() for tv in weekly[3].daySchedule] == [22.0, 19.0]


def test_create_weeklySchedule_occupied_7am_unoccupied_5pm(sched):
    schedule = {
        "states": {"Occupied": 1, "UnOccupied": 2},
        "week": {
            **_week([("7:00", "Occupied"), ("17:00", "UnOccupied")]),
            "saturday": [],
            "sunday": [],
        },
    }
    weekly = sched.create_weeklySchedule(schedule)
    for daily in weekly[:5]:
        assert [(tuple(tv.time), tv.value.get_value()) for tv in daily.daySchedule] == [
            ((7, 0, 0, 0), 1),
            ((17, 0, 0, 0), 2),
        ]
    assert len(weekly[5].daySchedule) == 0
    assert len(weekly[6].daySchedule) == 0


def test_create_weeklySchedule_empty_day(sched):
    schedule = copy.deepcopy(MULTISTATE)
    schedule["week"]["wednesday"] = []
    weekly = sched.create_weeklySchedule(schedule)
    assert len(weekly) == 7
    assert len(weekly[2].daySchedule) == 0
    assert len(weekly[3].daySchedule) == 2


def test_create_weeklySchedule_midnight_event(sched):
    schedule = {"states": "analog", "week": _week([("00:00", 18), ("0:00", 19.5)])}
    weekly = sched.create_weeklySchedule(schedule)
    for daily in weekly:
        assert [tuple(tv.time) for tv in daily.daySchedule] == [
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ]
        assert daily.daySchedule[1].value.get_value() == 19.5


def test_create_weeklySchedule_missing_day_raises_keyerror(sched):
    schedule = copy.deepcopy(BINARY)
    del schedule["week"]["sunday"]
    with pytest.raises(KeyError):
        sched.create_weeklySchedule(schedule)


def test_create_weeklySchedule_bad_state_name_raises_keyerror(sched):
    schedule = copy.deepcopy(MULTISTATE)
    schedule["week"]["friday"] = [("8:00", "Occupado")]
    with pytest.raises(KeyError):
        sched.create_weeklySchedule(schedule)


def test_create_weeklySchedule_stores_object_reference(sched):
    try:
        sched.create_weeklySchedule(BINARY, object_reference="schedule:99")
        assert sched.schedules["schedule:99"] is BINARY
    finally:
        Schedule.schedules.pop("schedule:99", None)


@pytest.mark.xfail(
    strict=False,
    reason="Suspected bug: Schedule.schedules is a class-level dict shared by "
    "every instance",
)
def test_schedules_not_shared_between_instances():
    first, second = Schedule(), Schedule()
    try:
        first.create_weeklySchedule(BINARY, object_reference="schedule:98")
        assert "schedule:98" not in second.schedules
    finally:
        Schedule.schedules.pop("schedule:98", None)


@pytest.mark.xfail(
    strict=False,
    reason="Suspected bug: _set_value returns None for unknown string states, "
    "silently creating events without a value",
)
def test_create_weeklySchedule_unknown_string_states_rejected(sched):
    schedule = {"states": "analgo", "week": _week([("8:00", 21)])}
    with pytest.raises((ValueError, TypeError, KeyError)):
        sched.create_weeklySchedule(schedule)


# ---------------------------------------------------------------------------
# decode_dailySchedule / decode_weeklySchedule
# ---------------------------------------------------------------------------


def test_decode_dailySchedule_multistate(sched):
    daily = _daily(((7, 0, 0, 0), Unsigned(1)), ((17, 30, 0, 0), Unsigned(3)))
    states = ["Occupied", "UnOccupied", "Standby", "Not Set"]
    assert sched.decode_dailySchedule(daily, states, 1) == [
        ("07:00", "Occupied"),
        ("17:30", "Standby"),
    ]


def test_decode_dailySchedule_binary(sched):
    daily = _daily(((6, 15, 0, 0), Enumerated(1)), ((18, 0, 0, 0), Enumerated(0)))
    assert sched.decode_dailySchedule(daily, ["inactive", "active"], 0) == [
        ("06:15", "active"),
        ("18:00", "inactive"),
    ]


def test_decode_dailySchedule_analog(sched):
    daily = _daily(((0, 0, 0, 0), Real(21.5)), ((23, 59, 0, 0), Real(18)))
    assert sched.decode_dailySchedule(daily, "analog", 0) == [
        ("00:00", 21.5),
        ("23:59", 18.0),
    ]


def test_decode_dailySchedule_empty(sched):
    assert sched.decode_dailySchedule(_daily(), "analog", 0) == []


def test_decode_dailySchedule_ignores_seconds_and_hundredths(sched):
    daily = _daily(((7, 30, 15, 50), Real(20)))
    assert sched.decode_dailySchedule(daily, "analog", 0) == [("07:30", 20.0)]


@pytest.mark.parametrize(
    "states, offset, raw",
    [
        (["Occupied", "UnOccupied"], 1, 9),
        (["inactive", "active"], 0, 2),
        (range(3), 0, 7),
    ],
    ids=["multistate", "binary", "unlinked"],
)
def test_decode_dailySchedule_out_of_range(sched, states, offset, raw):
    daily = _daily(((8, 0, 0, 0), Unsigned(1 + offset)), ((9, 0, 0, 0), Unsigned(raw)))
    events = sched.decode_dailySchedule(daily, states, offset)
    assert events[1] == ("09:00", f"??? ({raw})")
    assert events[0][0] == "08:00"


def test_decode_dailySchedule_unlinked_returns_raw_index(sched):
    daily = _daily(((8, 0, 0, 0), Unsigned(4)))
    assert sched.decode_dailySchedule(daily, range(255), 0) == [("08:00", 4)]


@pytest.mark.xfail(
    strict=False,
    reason="Suspected bug: a Null (relinquish) event value decodes to () and "
    "crashes int()",
)
def test_decode_dailySchedule_null_value(sched):
    daily = _daily(((8, 0, 0, 0), Null(())))
    assert sched.decode_dailySchedule(daily, ["a", "b"], 1) == [("08:00", None)]


@pytest.mark.parametrize("kind", list(DECODE_ARGS))
def test_decode_weeklySchedule_round_trip(sched, kind):
    example, states, offset = DECODE_ARGS[kind]
    weekly = sched.create_weeklySchedule(example)
    week = sched.decode_weeklySchedule(weekly, states, offset)
    assert list(week) == Schedule.days
    assert week == _expected_week(example)


def test_decode_weeklySchedule_handcrafted(sched):
    weekly = Schedule.WeeklySchedule(
        [_daily(((7, 0, 0, 0), Unsigned(1)), ((17, 0, 0, 0), Unsigned(2)))] * 5
        + [_daily(), _daily(((10, 0, 0, 0), Unsigned(42)))]
    )
    week = sched.decode_weeklySchedule(weekly, ["Occupied", "UnOccupied"], 1)
    for day in Schedule.days[:5]:
        assert week[day] == [("07:00", "Occupied"), ("17:00", "UnOccupied")]
    assert week["saturday"] == []
    assert week["sunday"] == [("10:00", "??? (42)")]


# ---------------------------------------------------------------------------
# make / send / write requests
# ---------------------------------------------------------------------------


def test_make_weeklySchedule_request(sched):
    weekly = sched.create_weeklySchedule(BINARY)
    request = sched.make_weeklySchedule_request("192.168.1.10", 5, weekly)
    assert request.objectIdentifier[0] == ObjectType("schedule")
    assert request.objectIdentifier[1] == 5
    assert request.propertyIdentifier == PropertyIdentifier("weeklySchedule")
    assert request.priority == 15
    assert request.pduDestination == Address("192.168.1.10")
    assert isinstance(request.propertyValue, Any)
    encoded = request.propertyValue.cast_out(Schedule.WeeklySchedule)
    assert sched.decode_weeklySchedule(
        encoded, ["inactive", "active"], 0
    ) == _expected_week(BINARY)


def test_send_weeklyschedule_request(sched):
    app = MagicMock()
    sched.this_application = SimpleNamespace(app=app)
    request = sched.make_weeklySchedule_request(
        "192.168.1.10", 5, sched.create_weeklySchedule(ANALOG)
    )
    sched.send_weeklyschedule_request(request)
    app.request.assert_called_once_with(request)


def test_write_weeklySchedule_builds_and_sends(sched):
    app = MagicMock()
    sched.this_application = SimpleNamespace(app=app)
    sched.write_weeklySchedule("192.168.1.10:47809", 7, MULTISTATE)
    app.request.assert_called_once()
    (request,), _ = app.request.call_args
    assert request.objectIdentifier[1] == 7
    assert request.pduDestination == Address("192.168.1.10:47809")
    encoded = request.propertyValue.cast_out(Schedule.WeeklySchedule)
    states, offset = DECODE_ARGS["multistate"][1:]
    assert sched.decode_weeklySchedule(encoded, states, offset) == _expected_week(
        MULTISTATE
    )


# ---------------------------------------------------------------------------
# read_weeklySchedule with mocked reads
# ---------------------------------------------------------------------------


def _ref(obj_type, instance):
    return DeviceObjectPropertyReference(
        objectIdentifier=ObjectIdentifier((obj_type, instance)),
        propertyIdentifier="presentValue",
    )


def _mock_reads(sched, example, refs, present_value, names=None, state_text=None):
    weekly = sched.create_weeklySchedule(example)

    async def read_multiple(args):
        return [
            weekly,
            refs,
            Reliability("noFaultDetected"),
            16,
            AnyAtomic(present_value),
        ]

    async def read(args):
        if args.endswith("stateText"):
            return state_text
        return names.pop(0)

    sched.readMultiple = read_multiple
    sched.read = read


async def test_read_weeklySchedule_binary_mocked(sched):
    _mock_reads(sched, BINARY, [_ref("binaryValue", 2)], Enumerated(1), ["BV-2"])
    result = await sched.read_weeklySchedule("10.0.0.5", 1)
    assert result["states"] == {"inactive": 0, "active": 1}
    assert result["presentValue"] == "active (1)"
    assert result["object_references"] == [ObjectIdentifier(("binaryValue", 2))]
    assert result["references_names"] == ["BV-2"]
    assert result["priority"] == 16
    assert result["week"] == _expected_week(BINARY)


async def test_read_weeklySchedule_analog_mocked(sched):
    _mock_reads(sched, ANALOG, [_ref("analogValue", 3)], Real(22), ["SP"])
    result = await sched.read_weeklySchedule("10.0.0.5", 1)
    assert result["states"] == "analog"
    assert result["presentValue"] == "22.0"
    assert result["week"] == _expected_week(ANALOG)


async def test_read_weeklySchedule_multistate_mocked(sched):
    state_text = list(MULTISTATE["states"])
    _mock_reads(
        sched,
        MULTISTATE,
        [_ref("multiStateValue", 4), _ref("multiStateValue", 5)],
        Unsigned(3),
        ["MODE-A", "MODE-B"],
        state_text,
    )
    result = await sched.read_weeklySchedule("10.0.0.5", 1)
    assert result["states"] == MULTISTATE["states"]
    assert result["presentValue"] == "Standby (3)"
    assert result["references_names"] == ["MODE-A", "MODE-B"]
    assert result["week"] == _expected_week(MULTISTATE)


async def test_read_weeklySchedule_no_references_mocked(sched):
    raw = {"states": {"a": 0, "b": 1, "c": 2}, "week": _week([("8:00", "c")])}
    _mock_reads(sched, raw, [], Unsigned(2))
    result = await sched.read_weeklySchedule("10.0.0.5", 1)
    assert result["object_references"] == []
    assert result["references_names"] == []
    assert result["states"] == {i: i for i in range(255)}
    assert result["presentValue"] == "2 (2)"
    assert result["week"] == _week([("08:00", 2)])


async def test_read_weeklySchedule_single_read_fallback(sched):
    weekly = sched.create_weeklySchedule(
        {"states": {"x": 1}, "week": _week([("9:00", "x")])}
    )
    values = {
        "weeklySchedule": weekly,
        "listOfObjectPropertyReferences": [],
        "reliability": "noFaultDetected",
        "priorityForWriting": 16,
        "presentValue": AnyAtomic(Unsigned(1)),
    }

    async def read_multiple(args):
        raise RuntimeError("segmentation not supported")

    # non-coroutine stub: the fallback calls self.read() without await
    sched.readMultiple = read_multiple
    sched.read = MagicMock(side_effect=lambda args: values[args.split()[-1]])
    result = await sched.read_weeklySchedule("10.0.0.5", 1)
    assert sched.read.call_count == 5
    assert result["week"] == _week([("09:00", 1)])
    assert result["presentValue"] == "1 (1)"


@pytest.mark.xfail(
    strict=False,
    raises=TypeError,
    reason="Suspected bug: bare `raise ()` turns the read error into "
    "TypeError: exceptions must derive from BaseException",
)
async def test_read_weeklySchedule_fallback_error_propagates(sched):
    async def read_multiple(args):
        raise RuntimeError("segmentation not supported")

    sched.readMultiple = read_multiple
    sched.read = MagicMock(side_effect=TimeoutError("no response"))
    with pytest.raises(TimeoutError):
        await sched.read_weeklySchedule("10.0.0.5", 1)


# ---------------------------------------------------------------------------
# Integration: real schedule objects on device_app (network_and_devices)
# ---------------------------------------------------------------------------

SCHED_INSTANCES = {"multistate": 9001, "binary": 9002, "analog": 9003, "unlinked": 9004}


def _status():
    return StatusFlags([0, 0, 0, 0])


def _schedule_object(instance, refs, default):
    return ScheduleObject(
        objectIdentifier=("schedule", instance),
        objectName=f"TEST-SCHEDULE-{instance}",
        presentValue=default,
        scheduleDefault=default,
        weeklySchedule=[DailySchedule(daySchedule=[]) for _ in range(7)],
        listOfObjectPropertyReferences=refs,
        priorityForWriting=16,
        reliability=Reliability.noFaultDetected,
        statusFlags=_status(),
        outOfService=False,
    )


def _ensure_schedule_objects(device_app):
    """Add the referenced points and schedules once (test-side only)."""
    app = device_app.this_application.app
    if app.get_object_name("TEST-SCHEDULE-9001") is not None:
        return

    targets = {
        "multistate": MultiStateValueObject(
            objectIdentifier=("multiStateValue", 9001),
            objectName="SCHED-OCC-MODE",
            presentValue=1,
            numberOfStates=4,
            stateText=list(MULTISTATE["states"]),
            statusFlags=_status(),
            eventState=EventState.normal,
            outOfService=False,
        ),
        "binary": BinaryValueObject(
            objectIdentifier=("binaryValue", 9002),
            objectName="SCHED-OCC-BV",
            presentValue="inactive",
            statusFlags=_status(),
            eventState=EventState.normal,
            outOfService=False,
        ),
        "analog": AnalogValueObject(
            objectIdentifier=("analogValue", 9003),
            objectName="SCHED-ZN-SP",
            presentValue=21.0,
            units=EngineeringUnits.degreesCelsius,
            statusFlags=_status(),
            eventState=EventState.normal,
            outOfService=False,
        ),
    }
    defaults = {
        "multistate": Unsigned(1),
        "binary": Enumerated(0),
        "analog": Real(21.0),
        "unlinked": Unsigned(2),
    }
    objects = {}
    for kind, instance in SCHED_INSTANCES.items():
        refs = []
        if kind in targets:
            target = targets[kind]
            objects[target.objectName] = target
            refs = [
                DeviceObjectPropertyReference(
                    objectIdentifier=target.objectIdentifier,
                    propertyIdentifier="presentValue",
                )
            ]
        sched_obj = _schedule_object(instance, refs, defaults[kind])
        objects[sched_obj.objectName] = sched_obj

    holder = SimpleNamespace(
        objects=objects,
        log=lambda *args, **kwargs: None,
        _log=logging.getLogger("test_Schedule"),
    )
    ObjectFactory.add_objects_to_application(holder, device_app)


def _address(device_app):
    return f"{device_app.localIPAddr.addrTuple[0]}:47809"


def _write_without_priority(bacnet, monkeypatch):
    """weeklySchedule is not commandable; drop the priority=15 the library adds
    so the bacpypes3 device accepts the write (see xfail test below)."""
    original = bacnet.make_weeklySchedule_request

    def make_request(*args, **kwargs):
        request = original(*args, **kwargs)
        request.priority = None
        return request

    monkeypatch.setattr(bacnet, "make_weeklySchedule_request", make_request)


OFFICE_HOURS = {
    "states": {"Occupied": 1, "UnOccupied": 2, "Standby": 3, "Not Set": 4},
    "week": {
        **_week([("07:00", "Occupied"), ("17:00", "UnOccupied")]),
        "saturday": [("09:00", "Standby"), ("12:00", "UnOccupied")],
        "sunday": [],
    },
}

ROUND_TRIP_CASES = {
    "multistate": OFFICE_HOURS,
    "binary": BINARY,
    "analog": ANALOG,
}


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("kind", list(ROUND_TRIP_CASES))
async def test_write_read_weeklySchedule_round_trip(
    network_and_devices, monkeypatch, kind
):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    _ensure_schedule_objects(device_app)
    _write_without_priority(bacnet, monkeypatch)
    address = _address(device_app)
    instance = SCHED_INSTANCES[kind]
    schedule = ROUND_TRIP_CASES[kind]

    bacnet.write_weeklySchedule(address, instance, schedule)
    await asyncio.sleep(0.5)
    result = await bacnet.read_weeklySchedule(address, instance)

    assert result["week"] == _expected_week(schedule)
    assert result["priority"] == 16
    assert len(result["object_references"]) == 1
    if kind == "multistate":
        assert result["states"] == schedule["states"]
        assert result["references_names"] == ["SCHED-OCC-MODE"]
        assert result["presentValue"] == "Occupied (1)"
    elif kind == "binary":
        assert result["states"] == {"inactive": 0, "active": 1}
        assert result["references_names"] == ["SCHED-OCC-BV"]
        assert result["presentValue"] == "inactive (0)"
    else:
        assert result["states"] == "analog"
        assert result["references_names"] == ["SCHED-ZN-SP"]
        assert result["presentValue"] == "21.0"


@pytest.mark.asyncio(loop_scope="session")
async def test_write_read_weeklySchedule_no_object_references(
    network_and_devices, monkeypatch
):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    _ensure_schedule_objects(device_app)
    _write_without_priority(bacnet, monkeypatch)
    address = _address(device_app)
    instance = SCHED_INSTANCES["unlinked"]
    schedule = {
        "states": {"Occupied": 1, "UnOccupied": 2},
        "week": _week([("06:30", "Occupied"), ("22:00", "UnOccupied")]),
    }

    bacnet.write_weeklySchedule(address, instance, schedule)
    await asyncio.sleep(0.5)
    result = await bacnet.read_weeklySchedule(address, instance)

    # unlinked schedules are decoded as raw numbers
    assert result["object_references"] == []
    assert result["references_names"] == []
    assert result["states"] == {i: i for i in range(255)}
    assert result["presentValue"] == "2 (2)"
    assert result["week"] == _week([("06:30", 1), ("22:00", 2)])


@pytest.mark.xfail(
    strict=False,
    reason="Suspected bug: write_weeklySchedule always sends priority=15, which "
    "the device rejects for the non-commandable weeklySchedule property; the "
    "request is fire-and-forget so the error is never reported",
)
@pytest.mark.asyncio(loop_scope="session")
async def test_write_weeklySchedule_default_priority_lands(network_and_devices):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    _ensure_schedule_objects(device_app)
    address = _address(device_app)
    instance = SCHED_INSTANCES["multistate"]
    schedule = {
        "states": OFFICE_HOURS["states"],
        "week": _week([("05:45", "Standby"), ("20:00", "Not Set")]),
    }

    bacnet.write_weeklySchedule(address, instance, schedule)
    await asyncio.sleep(0.5)
    result = await bacnet.read_weeklySchedule(address, instance)
    assert result["week"] == _expected_week(schedule)
