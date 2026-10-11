#!/usr/bin/env python
# -*- coding utf-8 -*-

"""
Unit tests for BAC0.core.functions.Schedule (weekly occupancy schedules).

The create/decode helpers are pure functions and are tested without any
network. Network send/read paths are tested with the application mocked.
"""

import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from bacpypes3.apdu import WritePropertyRequest
from bacpypes3.basetypes import DailySchedule, TimeValue
from bacpypes3.pdu import Address
from bacpypes3.primitivedata import (
    Enumerated,
    Integer,
    ObjectType,
    PropertyIdentifier,
    Real,
)

from BAC0.core.functions.Schedule import Schedule

DAYS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]

MULTISTATE = Schedule.schedule_example_multistate
BINARY = Schedule.schedule_example_binary
ANALOG = Schedule.schedule_example_analog

MULTISTATE_STATES = list(MULTISTATE["states"].keys())
BINARY_STATES = ["inactive", "active"]


@pytest.fixture
def schedule():
    return Schedule()


def _hhmm(time_str):
    hour, minute = (int(x) for x in time_str.split(":"))
    return f"{hour:02d}:{minute:02d}"


def _expected_week(dict_schedule, cast=lambda v: v):
    return {
        day: [(_hhmm(t), cast(v)) for t, v in dict_schedule["week"][day]]
        for day in DAYS
    }


def _week_with(**days):
    week = {day: [] for day in DAYS}
    week.update(days)
    return week


def _daily(*events):
    """Build a DailySchedule from (time, Atomic value) pairs."""
    return DailySchedule(daySchedule=[TimeValue(time=t, value=v) for t, v in events])


# --- create_weeklySchedule ---------------------------------------------------


@pytest.mark.parametrize(
    "dict_schedule, expected_type",
    [
        (MULTISTATE, Enumerated),
        (BINARY, Integer),
        (ANALOG, Real),
    ],
    ids=["multistate", "binary", "analog"],
)
def test_create_weeklySchedule_value_types(schedule, dict_schedule, expected_type):
    weekly = schedule.create_weeklySchedule(dict_schedule)

    assert len(weekly) == 7
    for i, day in enumerate(DAYS):
        daily = weekly[i]
        assert isinstance(daily, DailySchedule)
        events = dict_schedule["week"][day]
        assert len(daily.daySchedule) == len(events)
        for tv, (time_str, _) in zip(daily.daySchedule, events):
            assert isinstance(tv, TimeValue)
            assert type(tv.value.get_value()) is expected_type
            hour, minute = (int(x) for x in time_str.split(":"))
            assert tuple(tv.time) == (hour, minute, 0, 0)


def test_create_weeklySchedule_monday_to_sunday_order(schedule):
    # Each example day starts at a distinct hour: monday=1:00 ... sunday=7:00
    weekly = schedule.create_weeklySchedule(MULTISTATE)
    first_hours = [weekly[i].daySchedule[0].time[0] for i in range(7)]
    assert first_hours == [1, 2, 3, 4, 5, 6, 7]


def test_create_weeklySchedule_encoded_values(schedule):
    multistate = schedule.create_weeklySchedule(MULTISTATE)
    assert [tv.value.get_value() for tv in multistate[0].daySchedule] == [1, 2]

    binary = schedule.create_weeklySchedule(BINARY)
    assert [tv.value.get_value() for tv in binary[0].daySchedule] == [1, 0]

    analog = schedule.create_weeklySchedule(ANALOG)
    assert [tv.value.get_value() for tv in analog[0].daySchedule] == [22.0, 19.0]


def test_create_weeklySchedule_analog_states_case_insensitive(schedule):
    dict_schedule = copy.deepcopy(ANALOG)
    dict_schedule["states"] = "Analog"
    weekly = schedule.create_weeklySchedule(dict_schedule)
    assert type(weekly[0].daySchedule[0].value.get_value()) is Real


def test_create_weeklySchedule_stores_object_reference(schedule):
    ref = "test-schedule-ref"
    try:
        schedule.create_weeklySchedule(BINARY, object_reference=ref)
        assert Schedule.schedules[ref] is BINARY
    finally:
        Schedule.schedules.pop(ref, None)


def test_create_weeklySchedule_without_reference_does_not_store(schedule):
    before = dict(Schedule.schedules)
    schedule.create_weeklySchedule(BINARY)
    assert Schedule.schedules == before


def test_create_weeklySchedule_missing_day_raises(schedule):
    dict_schedule = copy.deepcopy(BINARY)
    del dict_schedule["week"]["sunday"]
    with pytest.raises(KeyError):
        schedule.create_weeklySchedule(dict_schedule)


# --- round trip create -> decode ---------------------------------------------


def test_roundtrip_multistate(schedule):
    weekly = schedule.create_weeklySchedule(MULTISTATE)
    decoded = schedule.decode_weeklySchedule(weekly, MULTISTATE_STATES, 1)
    assert decoded == _expected_week(MULTISTATE)


def test_roundtrip_binary(schedule):
    weekly = schedule.create_weeklySchedule(BINARY)
    decoded = schedule.decode_weeklySchedule(weekly, BINARY_STATES, 0)
    assert decoded == _expected_week(BINARY)


def test_roundtrip_analog(schedule):
    weekly = schedule.create_weeklySchedule(ANALOG)
    decoded = schedule.decode_weeklySchedule(weekly, "analog", 0)
    assert decoded == _expected_week(ANALOG, cast=float)
    assert all(
        isinstance(value, float) for events in decoded.values() for _, value in events
    )


# --- edge cases --------------------------------------------------------------


def test_roundtrip_weekend_unoccupied(schedule):
    dict_schedule = copy.deepcopy(MULTISTATE)
    dict_schedule["week"]["saturday"] = []
    dict_schedule["week"]["sunday"] = []

    weekly = schedule.create_weeklySchedule(dict_schedule)
    assert len(weekly) == 7
    assert len(weekly[5].daySchedule) == 0
    assert len(weekly[6].daySchedule) == 0

    decoded = schedule.decode_weeklySchedule(weekly, MULTISTATE_STATES, 1)
    assert decoded["saturday"] == []
    assert decoded["sunday"] == []
    assert decoded["friday"] == [("05:00", "Occupied"), ("17:00", "UnOccupied")]


def test_roundtrip_all_days_empty(schedule):
    dict_schedule = {"states": BINARY["states"], "week": _week_with()}
    weekly = schedule.create_weeklySchedule(dict_schedule)
    decoded = schedule.decode_weeklySchedule(weekly, BINARY_STATES, 0)
    assert decoded == {day: [] for day in DAYS}


def test_roundtrip_midnight_and_end_of_day(schedule):
    dict_schedule = {
        "states": BINARY["states"],
        "week": _week_with(monday=[("0:00", "active"), ("23:59", "inactive")]),
    }
    weekly = schedule.create_weeklySchedule(dict_schedule)
    assert tuple(weekly[0].daySchedule[0].time) == (0, 0, 0, 0)
    assert tuple(weekly[0].daySchedule[1].time) == (23, 59, 0, 0)

    decoded = schedule.decode_weeklySchedule(weekly, BINARY_STATES, 0)
    assert decoded["monday"] == [("00:00", "active"), ("23:59", "inactive")]


def test_roundtrip_single_event_day(schedule):
    dict_schedule = {
        "states": "analog",
        "week": _week_with(wednesday=[("12:30", 21.5)]),
    }
    weekly = schedule.create_weeklySchedule(dict_schedule)
    decoded = schedule.decode_weeklySchedule(weekly, "analog", 0)
    assert decoded["wednesday"] == [("12:30", 21.5)]
    assert all(decoded[day] == [] for day in DAYS if day != "wednesday")


# --- decode_dailySchedule ----------------------------------------------------


def test_decode_dailySchedule_time_format(schedule):
    daily = _daily(
        ("7:05", Real(1.0)),
        ("13:00", Real(2.0)),
        ("23:59:59.99", Real(3.0)),
    )
    events = schedule.decode_dailySchedule(daily, "analog", 0)
    assert [t for t, _ in events] == ["07:05", "13:00", "23:59"]


def test_decode_dailySchedule_multistate_offset(schedule):
    daily = _daily(("8:00", Enumerated(1)), ("18:00", Enumerated(4)))
    events = schedule.decode_dailySchedule(daily, MULTISTATE_STATES, 1)
    assert events == [("08:00", "Occupied"), ("18:00", "Not Set")]


def test_decode_dailySchedule_multistate_out_of_range(schedule):
    daily = _daily(("8:00", Enumerated(9)))
    events = schedule.decode_dailySchedule(daily, MULTISTATE_STATES, 1)
    assert events == [("08:00", "??? (9)")]


def test_decode_dailySchedule_binary_out_of_range(schedule):
    daily = _daily(("8:00", Integer(5)))
    events = schedule.decode_dailySchedule(daily, BINARY_STATES, 0)
    assert events == [("08:00", "??? (5)")]


def test_decode_dailySchedule_unlinked_schedule_uses_raw_index(schedule):
    # read_weeklySchedule falls back to range(255) when no object is linked
    daily = _daily(("8:00", Enumerated(3)))
    events = schedule.decode_dailySchedule(daily, range(255), 0)
    assert events == [("08:00", 3)]


def test_decode_dailySchedule_empty(schedule):
    assert schedule.decode_dailySchedule(_daily(), BINARY_STATES, 0) == []


def test_decode_dailySchedule_none_value(schedule):
    entry = SimpleNamespace(
        time=(9, 0, 0, 0), value=SimpleNamespace(get_value=lambda: None)
    )
    daily = SimpleNamespace(daySchedule=[entry])
    assert schedule.decode_dailySchedule(daily, BINARY_STATES, 0) == [("09:00", None)]


@pytest.mark.xfail(
    reason=(
        "suspected bug: a multistate value of 0 (invalid, states start at 1) "
        "is decoded via a negative list index and silently shown as the last "
        "state instead of the '??? (0)' fallback"
    ),
    strict=False,
)
def test_decode_dailySchedule_multistate_zero_is_flagged(schedule):
    daily = _daily(("8:00", Enumerated(0)))
    events = schedule.decode_dailySchedule(daily, MULTISTATE_STATES, 1)
    assert events == [("08:00", "??? (0)")]


# --- request building / sending (network mocked) -----------------------------


def test_make_weeklySchedule_request(schedule):
    weekly = schedule.create_weeklySchedule(MULTISTATE)
    request = schedule.make_weeklySchedule_request(
        destination="2:5", object_instance=10, weeklySchedule=weekly
    )

    assert isinstance(request, WritePropertyRequest)
    assert request.objectIdentifier == (ObjectType("schedule"), 10)
    assert str(request.objectIdentifier[0]) == "schedule"
    assert request.objectIdentifier[1] == 10
    assert request.propertyIdentifier == PropertyIdentifier("weeklySchedule")
    assert request.priority == 15
    assert request.pduDestination == Address("2:5")


def test_make_weeklySchedule_request_payload_roundtrip(schedule):
    weekly = schedule.create_weeklySchedule(BINARY)
    request = schedule.make_weeklySchedule_request("192.168.1.10", 1, weekly)

    payload = request.propertyValue.cast_out(Schedule.WeeklySchedule)
    decoded = schedule.decode_weeklySchedule(payload, BINARY_STATES, 0)
    assert decoded == _expected_week(BINARY)


def test_send_weeklyschedule_request_uses_app(schedule):
    schedule.this_application = MagicMock()
    weekly = schedule.create_weeklySchedule(BINARY)
    request = schedule.make_weeklySchedule_request("2:5", 3, weekly)

    schedule.send_weeklyschedule_request(request)

    schedule.this_application.app.request.assert_called_once_with(request)


def test_write_weeklySchedule_builds_and_sends(schedule):
    schedule.this_application = MagicMock()

    schedule.write_weeklySchedule("2:5", 42, ANALOG)

    app_request = schedule.this_application.app.request
    app_request.assert_called_once()
    (sent,), _ = app_request.call_args
    assert isinstance(sent, WritePropertyRequest)
    assert sent.objectIdentifier == (ObjectType("schedule"), 42)
    assert sent.propertyIdentifier == PropertyIdentifier("weeklySchedule")
    assert sent.priority == 15
    payload = sent.propertyValue.cast_out(Schedule.WeeklySchedule)
    assert schedule.decode_weeklySchedule(payload, "analog", 0) == _expected_week(
        ANALOG, cast=float
    )


# --- read_weeklySchedule (network mocked) ------------------------------------


def _ref(obj_type, instance):
    return SimpleNamespace(objectIdentifier=(obj_type, instance))


def _pv(value):
    return SimpleNamespace(get_value=lambda: value)


@pytest.mark.asyncio(loop_scope="session")
async def test_read_weeklySchedule_multistate(schedule):
    weekly = schedule.create_weeklySchedule(MULTISTATE)
    schedule.readMultiple = AsyncMock(
        return_value=[weekly, [_ref("multi-state-value", 1)], "no-fault", 15, _pv(2)]
    )

    async def fake_read(args):
        if args.endswith("stateText"):
            return MULTISTATE_STATES
        if args.endswith("objectName"):
            return "MSV-1"
        raise AssertionError(args)

    schedule.read = AsyncMock(side_effect=fake_read)

    result = await schedule.read_weeklySchedule("2:5", 1)

    assert result["object_references"] == [("multi-state-value", 1)]
    assert result["references_names"] == ["MSV-1"]
    assert result["states"] == MULTISTATE["states"]
    assert result["presentValue"] == "UnOccupied (2)"
    assert result["reliability"] == "no-fault"
    assert result["priority"] == 15
    assert result["week"] == _expected_week(MULTISTATE)


@pytest.mark.asyncio(loop_scope="session")
async def test_read_weeklySchedule_binary(schedule):
    weekly = schedule.create_weeklySchedule(BINARY)
    schedule.readMultiple = AsyncMock(
        return_value=[weekly, [_ref("binary-value", 3)], "no-fault", 16, _pv(1)]
    )
    schedule.read = AsyncMock(return_value="BV-3")

    result = await schedule.read_weeklySchedule("2:5", 1)

    assert result["states"] == {"inactive": 0, "active": 1}
    assert result["presentValue"] == "active (1)"
    assert result["references_names"] == ["BV-3"]
    assert result["week"] == _expected_week(BINARY)


@pytest.mark.asyncio(loop_scope="session")
async def test_read_weeklySchedule_analog(schedule):
    weekly = schedule.create_weeklySchedule(ANALOG)
    schedule.readMultiple = AsyncMock(
        return_value=[weekly, [_ref("analog-value", 7)], "no-fault", 16, _pv(21.5)]
    )
    schedule.read = AsyncMock(return_value="AV-7")

    result = await schedule.read_weeklySchedule("2:5", 1)

    assert result["states"] == "analog"
    assert result["presentValue"] == "21.5"
    assert result["week"] == _expected_week(ANALOG, cast=float)


@pytest.mark.asyncio(loop_scope="session")
async def test_read_weeklySchedule_unlinked(schedule):
    weekly = schedule.create_weeklySchedule(MULTISTATE)
    schedule.readMultiple = AsyncMock(return_value=[weekly, [], "no-fault", 16, _pv(2)])
    schedule.read = AsyncMock()

    result = await schedule.read_weeklySchedule("2:5", 1)

    assert result["object_references"] == []
    assert result["references_names"] == []
    assert result["presentValue"] == "2 (2)"
    assert result["week"]["monday"] == [("01:00", 1), ("17:00", 2)]
    schedule.read.assert_not_called()


@pytest.mark.xfail(
    reason=(
        "suspected bug: when readMultiple is not supported, the single-read "
        "fallback calls self.read(...) without await, so coroutines (not "
        "values) are decoded and the call fails"
    ),
    strict=False,
)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.asyncio(loop_scope="session")
async def test_read_weeklySchedule_readMultiple_fallback(schedule):
    weekly = schedule.create_weeklySchedule(BINARY)
    values = {
        "weeklySchedule": weekly,
        "listOfObjectPropertyReferences": [_ref("binary-value", 3)],
        "reliability": "no-fault",
        "priorityForWriting": 16,
        "presentValue": _pv(1),
        "objectName": "BV-3",
    }

    async def fake_read(args):
        return values[args.split()[-1]]

    schedule.readMultiple = AsyncMock(side_effect=NotImplementedError("no RPM"))
    schedule.read = AsyncMock(side_effect=fake_read)

    result = await schedule.read_weeklySchedule("2:5", 1)

    assert result["week"] == _expected_week(BINARY)


@pytest.mark.xfail(
    reason=(
        "suspected bug: when both readMultiple and the single-read fallback "
        "fail, the code executes `raise ()`, which replaces the real error "
        "with an unrelated TypeError"
    ),
    strict=False,
)
@pytest.mark.asyncio(loop_scope="session")
async def test_read_weeklySchedule_fallback_error_is_preserved(schedule):
    schedule.readMultiple = AsyncMock(side_effect=ConnectionError("no RPM"))
    schedule.read = MagicMock(side_effect=ConnectionError("device offline"))

    with pytest.raises(ConnectionError):
        await schedule.read_weeklySchedule("2:5", 1)
