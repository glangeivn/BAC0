#!/usr/bin/env python
# -*- coding utf-8 -*-

"""
Unit tests for BAC0.core.functions.Calendar (holiday / exception calendars).

The create/decode helpers are pure functions and are tested without any
network. Network send/read paths are tested with the application mocked.
"""

import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from bacpypes3.apdu import WritePropertyRequest
from bacpypes3.basetypes import CalendarEntry, DateRange
from bacpypes3.constructeddata import Any
from bacpypes3.pdu import Address
from bacpypes3.primitivedata import ObjectType, PropertyIdentifier

from BAC0.core.functions.Calendar import Calendar

EXAMPLE = Calendar.datelist_example


@pytest.fixture
def calendar():
    return Calendar()


def _bacnet_date(date_str):
    """Expected (year - 1900, month, day, weekday) with Monday=1..Sunday=7."""
    year, month, day = (int(x) for x in date_str.split("/"))
    weekday = datetime.date(year, month, day).isoweekday()
    return (year - 1900, month, day, weekday)


# --- create_calendar ---------------------------------------------------------


def test_create_calendar_example_entry_count_and_order(calendar):
    date_list = calendar.create_calendar(EXAMPLE)
    entries = list(date_list)

    assert len(entries) == 4
    assert all(isinstance(e, CalendarEntry) for e in entries)
    # dates are listed first, then date ranges
    assert entries[0].date is not None and entries[0].dateRange is None
    assert entries[1].date is not None and entries[1].dateRange is None
    assert entries[2].dateRange is not None and entries[2].date is None
    assert entries[3].dateRange is not None and entries[3].date is None


def test_create_calendar_non_recurring_date(calendar):
    entry = calendar.create_calendar(EXAMPLE)[0]
    # 2021/3/14 was a Sunday
    assert tuple(entry.date) == (121, 3, 14, 7)
    assert tuple(entry.date) == _bacnet_date("2021/3/14")


def test_create_calendar_recurring_date_weekday_wildcard(calendar):
    entry = calendar.create_calendar(EXAMPLE)[1]
    assert entry.date[3] == 255
    assert tuple(entry.date)[1:3] == (3, 10)


def test_create_calendar_date_ranges(calendar):
    entries = list(calendar.create_calendar(EXAMPLE))
    for entry, expected in zip(entries[2:], EXAMPLE["dateRanges"]):
        assert isinstance(entry.dateRange, DateRange)
        assert tuple(entry.dateRange.startDate) == _bacnet_date(expected["startDate"])
        assert tuple(entry.dateRange.endDate) == _bacnet_date(expected["endDate"])


def test_create_calendar_only_dates(calendar):
    dict_calendar = {"dates": EXAMPLE["dates"]}
    entries = list(calendar.create_calendar(dict_calendar))
    assert len(entries) == 2
    assert all(e.date is not None and e.dateRange is None for e in entries)


def test_create_calendar_only_date_ranges(calendar):
    dict_calendar = {"dateRanges": EXAMPLE["dateRanges"]}
    entries = list(calendar.create_calendar(dict_calendar))
    assert len(entries) == 2
    assert all(e.dateRange is not None and e.date is None for e in entries)


def test_create_calendar_empty(calendar):
    assert len(calendar.create_calendar({})) == 0
    assert len(calendar.create_calendar({"dates": [], "dateRanges": []})) == 0


@pytest.mark.parametrize(
    "date_str, weekday",
    [
        ("2024/1/1", 1),  # Monday
        ("2024/1/3", 3),  # Wednesday
        ("2024/1/6", 6),  # Saturday
        ("2024/1/7", 7),  # Sunday
    ],
)
def test_create_calendar_weekday_numbering(calendar, date_str, weekday):
    entry = calendar.create_calendar(
        {"dates": [{"date": date_str, "recurring": False}]}
    )[0]
    assert entry.date[3] == weekday


# --- edge cases --------------------------------------------------------------


def test_create_calendar_leap_day(calendar):
    entry = calendar.create_calendar(
        {"dates": [{"date": "2024/2/29", "recurring": False}]}
    )[0]
    # 2024/2/29 was a Thursday
    assert tuple(entry.date) == (124, 2, 29, 4)


def test_create_calendar_invalid_leap_day_raises(calendar):
    with pytest.raises(ValueError):
        calendar.create_calendar({"dates": [{"date": "2023/2/29", "recurring": False}]})


def test_create_calendar_date_range_across_year_boundary(calendar):
    dict_calendar = {"dateRanges": [{"startDate": "2023/12/24", "endDate": "2024/1/2"}]}
    entry = calendar.create_calendar(dict_calendar)[0]
    assert tuple(entry.dateRange.startDate) == (123, 12, 24, 7)
    assert tuple(entry.dateRange.endDate) == (124, 1, 2, 2)


# --- round trip create -> decode ---------------------------------------------


def test_roundtrip_example(calendar):
    decoded = calendar.decode_dateList(calendar.create_calendar(EXAMPLE))
    assert decoded == EXAMPLE


def test_roundtrip_recurring_flag(calendar):
    decoded = calendar.decode_dateList(calendar.create_calendar(EXAMPLE))
    flags = {d["date"]: d["recurring"] for d in decoded["dates"]}
    assert flags == {"2021/3/14": False, "2021/3/10": True}


def test_roundtrip_only_dates(calendar):
    dict_calendar = {"dates": EXAMPLE["dates"]}
    decoded = calendar.decode_dateList(calendar.create_calendar(dict_calendar))
    assert decoded == {"dates": EXAMPLE["dates"], "dateRanges": []}


def test_roundtrip_only_date_ranges(calendar):
    dict_calendar = {"dateRanges": EXAMPLE["dateRanges"]}
    decoded = calendar.decode_dateList(calendar.create_calendar(dict_calendar))
    assert decoded == {"dates": [], "dateRanges": EXAMPLE["dateRanges"]}


def test_roundtrip_leap_day_and_year_boundary(calendar):
    dict_calendar = {
        "dates": [{"date": "2024/2/29", "recurring": False}],
        "dateRanges": [{"startDate": "2023/12/24", "endDate": "2024/1/2"}],
    }
    decoded = calendar.decode_dateList(calendar.create_calendar(dict_calendar))
    assert decoded == dict_calendar


def test_decode_dateList_empty(calendar):
    assert calendar.decode_dateList([]) == {"dates": [], "dateRanges": []}


# --- suspected bugs ----------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "suspected bug: recurring dates only wildcard the weekday; the year is "
        "still encoded as a concrete value (year - 1900), so 'every March 10' "
        "is sent to the controller as March 10 of one specific year"
    ),
    strict=False,
)
def test_create_calendar_recurring_date_wildcards_year(calendar):
    entry = calendar.create_calendar(
        {"dates": [{"date": "2021/3/10", "recurring": True}]}
    )[0]
    year, month, day, weekday = tuple(entry.date)
    assert year == 255  # BACnet "any year"
    assert (month, day) == (3, 10)
    assert weekday == 255


@pytest.mark.xfail(
    reason=(
        "suspected bug: decode_dateList does not understand a standard "
        "recurring date whose year is wildcarded (255); it reports year 2155"
    ),
    strict=False,
)
def test_decode_dateList_wildcard_year(calendar):
    entry = CalendarEntry(date=(255, 12, 25, 255))
    decoded = calendar.decode_dateList([entry])
    assert decoded["dates"][0]["recurring"] is True
    assert not decoded["dates"][0]["date"].startswith("2155")


# --- request building / sending (network mocked) -----------------------------


@pytest.mark.xfail(
    reason=(
        "suspected bug: make_calendar_request starts with Any(None), which "
        "bacpypes3 rejects with EncodingError('Any has no content'), so no "
        "calendar write request can be built"
    ),
    strict=False,
)
def test_make_calendar_request(calendar):
    date_list = calendar.create_calendar(EXAMPLE)
    request = calendar.make_calendar_request(
        destination="2:5", object_instance=4, dateList=date_list
    )

    assert isinstance(request, WritePropertyRequest)
    assert request.objectIdentifier == (ObjectType("calendar"), 4)
    assert request.propertyIdentifier == PropertyIdentifier("dateList")
    assert request.pduDestination == Address("2:5")
    payload = request.propertyValue.cast_out(Calendar.DateList)
    assert calendar.decode_dateList(payload) == EXAMPLE


@pytest.mark.xfail(
    reason=(
        "suspected bug: write_calendar_dateList depends on make_calendar_request, "
        "which fails, so nothing is ever sent to the device"
    ),
    strict=False,
)
def test_write_calendar_dateList_builds_and_sends(calendar):
    calendar.this_application = MagicMock()

    calendar.write_calendar_dateList("2:5", 4, EXAMPLE)

    calendar.this_application.app.request.assert_called_once()
    (sent,), _ = calendar.this_application.app.request.call_args
    assert sent.objectIdentifier == (ObjectType("calendar"), 4)
    assert sent.propertyIdentifier == PropertyIdentifier("dateList")


def test_send_calendar_request_uses_app(calendar):
    calendar.this_application = MagicMock()
    request = WritePropertyRequest(
        objectIdentifier=("calendar", 4), propertyIdentifier="dateList"
    )
    request.pduDestination = Address("2:5")
    request.propertyValue = Any(calendar.create_calendar(EXAMPLE))

    calendar.send_calendar_request(request)

    calendar.this_application.app.request.assert_called_once_with(request)


# --- read_calendar_dateList (network mocked) ---------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_read_calendar_dateList(calendar):
    calendar.read = AsyncMock(return_value=calendar.create_calendar(EXAMPLE))

    result = await calendar.read_calendar_dateList("2:5", 4)

    assert result == EXAMPLE
    calendar.read.assert_awaited_once_with("2:5 calendar:4 dateList")


@pytest.mark.asyncio(loop_scope="session")
async def test_read_calendar_dateList_error_returns_empty(calendar):
    calendar.read = AsyncMock(side_effect=ConnectionError("device offline"))

    assert await calendar.read_calendar_dateList("2:5", 4) == {}
