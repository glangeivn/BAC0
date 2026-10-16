#!/usr/bin/env python
# -*- coding utf-8 -*-

"""
Tests for BAC0.core.functions.Schedule (weekly schedules)
"""

import asyncio
import copy
from typing import AsyncGenerator

import pytest
from bacpypes3.basetypes import (
    DailySchedule,
    DeviceObjectPropertyReference,
    PropertyIdentifier,
    Reliability,
    StatusFlags,
    TimeValue,
)
from bacpypes3.object import ScheduleObject
from bacpypes3.pdu import Address
from bacpypes3.primitivedata import (
    Enumerated,
    Integer,
    ObjectIdentifier,
    ObjectType,
    Real,
    Time,
    Unsigned,
)

from BAC0.core.functions.Schedule import Schedule

MULTISTATE = Schedule.schedule_example_multistate
BINARY = Schedule.schedule_example_binary
ANALOG = Schedule.schedule_example_analog

# (example, states argument for decode, offset_MV) as read_weeklySchedule
# would compute them for a schedule linked to that kind of object
DECODE_ARGS = {
    "multistate": (MULTISTATE, list(MULTISTATE["states"].keys()), 1),
    "binary": (BINARY, ["inactive", "active"], 0),
    "analog": (ANALOG, "analog", 0),
}


@pytest.fixture
def sched():
    return Schedule()


def _week(events):
    return {day: list(events) for day in Schedule.days}


# ---------------------------------------------------------------------------
# create_weeklySchedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "example, value_type",
    [(MULTISTATE, Enumerated), (BINARY, Integer), (ANALOG, Real)],
    ids=["multistate", "binary", "analog"],
)
def test_create_weeklySchedule_examples(sched, example, value_type):
    weekly = sched.create_weeklySchedule(example)

    assert isinstance(weekly, Schedule.WeeklySchedule)
    assert len(weekly) == 7
    for day_index, day in enumerate(Schedule.days):
        daily = weekly[day_index]
        assert isinstance(daily, DailySchedule)
        expected_events = example["week"][day]
        assert len(daily.daySchedule) == len(expected_events)
        for time_value, (time_str, state) in zip(daily.daySchedule, expected_events):
            assert isinstance(time_value, TimeValue)
            assert isinstance(time_value.time, Time)
            hour, minute = (int(x) for x in time_str.split(":"))
            assert tuple(time_value.time) == (hour, minute, 0, 0)
            # TimeValue.value is AnyAtomic; get_value() gives the typed primitive
            value = time_value.value.get_value()
            assert isinstance(value, value_type)
            if isinstance(example["states"], dict):
                assert value == example["states"][state]
            else:
                assert value == state


@pytest.mark.parametrize("kind", DECODE_ARGS.keys())
def test_weeklySchedule_round_trip(sched, kind):
    example, states, offset_MV = DECODE_ARGS[kind]
    weekly = sched.create_weeklySchedule(example)

    decoded = sched.decode_weeklySchedule(weekly, states, offset_MV)

    expected = {
        day: [
            (f"{int(t.split(':')[0]):02d}:{t.split(':')[1]}", v)
            for t, v in example["week"][day]
        ]
        for day in Schedule.days
    }
    if kind == "analog":
        expected = {
            day: [(t, float(v)) for t, v in events] for day, events in expected.items()
        }
    assert decoded == expected


@pytest.mark.parametrize("kind", DECODE_ARGS.keys())
def test_decode_dailySchedule_single_day(sched, kind):
    example, states, offset_MV = DECODE_ARGS[kind]
    weekly = sched.create_weeklySchedule(example)
    decoded = sched.decode_dailySchedule(weekly[0], states, offset_MV)
    assert [v for _, v in decoded] == [
        float(v) if kind == "analog" else v for _, v in example["week"]["monday"]
    ]


def test_decode_unlinked_schedule_returns_raw_numbers(sched):
    # read_weeklySchedule uses range(255) / offset 0 when no object is linked
    weekly = sched.create_weeklySchedule(MULTISTATE)
    decoded = sched.decode_weeklySchedule(weekly, range(255), 0)
    assert decoded["monday"] == [("01:00", 1), ("17:00", 2)]


def test_empty_day_list(sched):
    schedule = copy.deepcopy(BINARY)
    schedule["week"]["monday"] = []
    weekly = sched.create_weeklySchedule(schedule)
    assert len(weekly[0].daySchedule) == 0
    decoded = sched.decode_weeklySchedule(weekly, ["inactive", "active"], 0)
    assert decoded["monday"] == []
    assert decoded["tuesday"] == [("02:00", "active"), ("16:00", "inactive")]


def test_missing_day_key_raises_keyerror(sched):
    # Documented behaviour: every one of the 7 days must be present in "week"
    schedule = copy.deepcopy(BINARY)
    del schedule["week"]["sunday"]
    with pytest.raises(KeyError, match="sunday"):
        sched.create_weeklySchedule(schedule)


@pytest.mark.parametrize("example", [MULTISTATE, BINARY], ids=["multistate", "binary"])
def test_unknown_state_name_raises_keyerror(sched, example):
    schedule = copy.deepcopy(example)
    schedule["week"]["monday"] = [("8:00", "NoSuchState")]
    with pytest.raises(KeyError, match="NoSuchState"):
        sched.create_weeklySchedule(schedule)


def test_midnight_time(sched):
    schedule = {"states": "analog", "week": _week([("0:00", 18), ("00:00", 19)])}
    weekly = sched.create_weeklySchedule(schedule)
    assert tuple(weekly[0].daySchedule[0].time) == (0, 0, 0, 0)
    assert tuple(weekly[0].daySchedule[1].time) == (0, 0, 0, 0)
    decoded = sched.decode_dailySchedule(weekly[0], "analog", 0)
    assert decoded == [("00:00", 18.0), ("00:00", 19.0)]


def test_decode_out_of_range_multistate_value(sched):
    daily = DailySchedule(
        daySchedule=[
            TimeValue(time="8:00", value=Enumerated(2)),
            TimeValue(time="9:00", value=Enumerated(9)),
        ]
    )
    decoded = sched.decode_dailySchedule(daily, ["Occupied", "UnOccupied"], 1)
    assert decoded == [("08:00", "UnOccupied"), ("09:00", "??? (9)")]


def test_decode_out_of_range_binary_value(sched):
    daily = DailySchedule(daySchedule=[TimeValue(time="8:00", value=Integer(2))])
    decoded = sched.decode_dailySchedule(daily, ["inactive", "active"], 0)
    assert decoded == [("08:00", "??? (2)")]


@pytest.mark.xfail(
    strict=False,
    reason="Suspected bug: value 0 with offset_MV=1 indexes states[-1] and "
    "silently returns the last state instead of '??? (0)'",
)
def test_decode_value_below_offset_is_flagged(sched):
    daily = DailySchedule(daySchedule=[TimeValue(time="8:00", value=Enumerated(0))])
    decoded = sched.decode_dailySchedule(daily, ["Occupied", "UnOccupied"], 1)
    assert decoded == [("08:00", "??? (0)")]


# ---------------------------------------------------------------------------
# schedules registry
# ---------------------------------------------------------------------------


def test_create_weeklySchedule_records_object_reference(sched):
    sched.create_weeklySchedule(BINARY, object_reference="test:ref")
    try:
        assert sched.schedules["test:ref"] is BINARY
    finally:
        Schedule.schedules.pop("test:ref", None)


@pytest.mark.xfail(
    strict=False,
    reason="Suspected bug: Schedule.schedules is a class-level mutable dict, so "
    "schedules recorded by one instance leak into every other instance",
)
def test_schedules_registry_not_shared_between_instances():
    first, second = Schedule(), Schedule()
    first.create_weeklySchedule(BINARY, object_reference="test:shared")
    try:
        assert "test:shared" not in second.schedules
    finally:
        Schedule.schedules.pop("test:shared", None)


# ---------------------------------------------------------------------------
# make_weeklySchedule_request
# ---------------------------------------------------------------------------


def test_make_weeklySchedule_request(sched):
    weekly = sched.create_weeklySchedule(MULTISTATE)
    request = sched.make_weeklySchedule_request(
        destination="192.168.1.10:47809", object_instance=7, weeklySchedule=weekly
    )
    assert request.objectIdentifier == ObjectIdentifier((ObjectType("schedule"), 7))
    assert request.propertyIdentifier == PropertyIdentifier("weeklySchedule")
    assert request.pduDestination == Address("192.168.1.10:47809")
    assert request.priority == 15
    decoded = request.propertyValue.cast_out(Schedule.WeeklySchedule)
    assert sched.decode_weeklySchedule(
        decoded, DECODE_ARGS["multistate"][1], 1
    ) == sched.decode_weeklySchedule(weekly, DECODE_ARGS["multistate"][1], 1)


# ---------------------------------------------------------------------------
# read_weeklySchedule error handling (no network)
# ---------------------------------------------------------------------------


class _UnreachableSchedule(Schedule):
    async def readMultiple(self, *args, **kwargs):
        raise ConnectionError("readMultiple not supported")

    def read(self, *args, **kwargs):
        raise ConnectionError("device unreachable")


@pytest.mark.xfail(
    strict=False,
    raises=TypeError,
    reason="Suspected bug: bare `raise ()` in read_weeklySchedule raises "
    "TypeError ('exceptions must derive from BaseException') instead of the "
    "original read error",
)
@pytest.mark.asyncio(loop_scope="session")
async def test_read_weeklySchedule_propagates_read_error():
    with pytest.raises(ConnectionError):
        await _UnreachableSchedule().read_weeklySchedule("192.168.1.10", 1)


# ---------------------------------------------------------------------------
# Integration: write/read against a ScheduleObject on device_app
# ---------------------------------------------------------------------------


def _device_address(device_app):
    return f"{device_app.localIPAddr.addrTuple[0]}:47809"


def _first_object_of_type(app, object_type):
    for oid in app.objectIdentifier:
        if oid[0] == ObjectType(object_type):
            return oid
    raise LookupError(object_type)


def _add_schedule_object(device_app, instance, reference, present_value):
    app = device_app.this_application.app
    oid = ObjectIdentifier((ObjectType("schedule"), instance))
    obj = app.get_object_id(oid)
    if obj is None:
        obj = ScheduleObject(
            objectIdentifier=oid,
            objectName=f"SCHED-{instance}",
            presentValue=present_value,
            scheduleDefault=present_value,
            listOfObjectPropertyReferences=[
                DeviceObjectPropertyReference(
                    objectIdentifier=reference, propertyIdentifier="presentValue"
                )
            ],
            priorityForWriting=16,
            reliability=Reliability("noFaultDetected"),
            statusFlags=StatusFlags([0, 0, 0, 0]),
            outOfService=False,
        )
        app.add_object(obj)
    return obj


def _setup_schedules(device_app):
    app = device_app.this_application.app
    big_alarm = app.get_object_name("BIG-ALARM").objectIdentifier
    zn_t = app.get_object_name("ZN-T").objectIdentifier
    bv = _first_object_of_type(app, "binaryValue")
    return {
        "multistate": (
            _add_schedule_object(device_app, 9001, big_alarm, Unsigned(1)),
            {
                "states": {"Normal": 1, "Alarm": 2, "Super Emergency": 3},
                "week": _week([("7:00", "Alarm"), ("18:30", "Normal")]),
            },
        ),
        "binary": (
            _add_schedule_object(device_app, 9002, bv, Enumerated(0)),
            {
                "states": {"inactive": 0, "active": 1},
                "week": _week([("0:00", "inactive"), ("6:15", "active")]),
            },
        ),
        "analog": (
            _add_schedule_object(device_app, 9003, zn_t, Real(21.0)),
            {"states": "analog", "week": _week([("6:00", 22.5), ("22:00", 18.0)])},
        ),
    }


def _expected_week(schedule):
    return {
        day: [
            (
                f"{int(t.split(':')[0]):02d}:{t.split(':')[1]}",
                float(v) if schedule["states"] == "analog" else v,
            )
            for t, v in events
        ]
        for day, events in schedule["week"].items()
    }


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("kind", ["multistate", "binary", "analog"])
async def test_weeklySchedule_network_round_trip(
    network_and_devices: AsyncGenerator, kind
):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    obj, schedule = _setup_schedules(device_app)[kind]
    address = _device_address(device_app)
    instance = obj.objectIdentifier[1]

    request = bacnet.make_weeklySchedule_request(
        destination=address,
        object_instance=instance,
        weeklySchedule=bacnet.create_weeklySchedule(schedule),
    )
    # weeklySchedule is not commandable; the bacpypes3 device rejects a write
    # carrying priority 15, so send it without one (see xfail test below)
    request.priority = None
    bacnet.send_weeklyschedule_request(request)
    await asyncio.sleep(0.5)

    result = await bacnet.read_weeklySchedule(address, instance)

    assert result["week"] == _expected_week(schedule)
    assert result["object_references"] == [
        obj.listOfObjectPropertyReferences[0].objectIdentifier
    ]
    assert result["priority"] == 16
    if kind == "analog":
        assert result["states"] == "analog"
        assert result["presentValue"] == "21.0"
    elif kind == "binary":
        assert result["states"] == {"inactive": 0, "active": 1}
        assert result["presentValue"] == "inactive (0)"
    else:
        assert result["references_names"] == ["BIG-ALARM"]
        assert result["states"] == schedule["states"]
        assert result["presentValue"] == "Normal (1)"


@pytest.mark.xfail(
    strict=False,
    reason="Suspected bug: write_weeklySchedule always sends priority=15, which a "
    "BACnet device can reject for the non-commandable weeklySchedule property; "
    "the request is fire-and-forget, so the error is never reported to the caller",
)
@pytest.mark.asyncio(loop_scope="session")
async def test_write_weeklySchedule_network_round_trip(
    network_and_devices: AsyncGenerator,
):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    obj, _ = _setup_schedules(device_app)["multistate"]
    schedule = {
        "states": {"Normal": 1, "Alarm": 2, "Super Emergency": 3},
        "week": _week([("5:45", "Super Emergency"), ("20:00", "Normal")]),
    }
    address = _device_address(device_app)

    bacnet.write_weeklySchedule(address, obj.objectIdentifier[1], schedule)
    await asyncio.sleep(0.5)

    result = await bacnet.read_weeklySchedule(address, obj.objectIdentifier[1])
    assert result["week"] == _expected_week(schedule)
