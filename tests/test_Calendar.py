#!/usr/bin/env python
# -*- coding utf-8 -*-

"""
Tests for BAC0.core.functions.Calendar (holiday calendars / dateList).

These are pure unit tests: ``Calendar`` is instantiated directly and network
access is replaced by mocks, so no BACnet stack is needed.

Suspected bugs found while writing these tests (documented, NOT fixed here):

* ``make_calendar_request`` builds ``Any(None)`` and then calls ``cast_in``;
  with bacpypes3 this raises ``EncodingError('Any has no content')``, so
  ``write_calendar_dateList`` can never send a request (see xfail tests).
* ``create_calendar`` with ``recurring: True`` only sets weekday=255 and keeps
  the year, so a "recurring" holiday still matches a single year only (BACnet
  needs year=255 as well for an every-year date).
* ``create_calendar``'s ``if weekday > 7: weekday = 1`` branch is dead code:
  ``date.weekday() + 1`` is always 1..7 and BACnet already uses Sunday=7.
* ``send_calendar_request(..., timeout=10)``: ``timeout`` is never used and the
  request is fire-and-forget, so device errors are never reported.
* ``decode_dateList`` adds 1900 to every year, so a wildcard year (255) is
  shown as 2155.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from bacpypes3.basetypes import CalendarEntry, DateRange
from bacpypes3.constructeddata import Any
from bacpypes3.pdu import Address
from bacpypes3.primitivedata import Date, ObjectType, PropertyIdentifier

from BAC0.core.functions.Calendar import Calendar


@pytest.fixture
def cal():
    return Calendar()


def _date_tuple(entry_date):
    return tuple(entry_date)


# ---------------------------------------------------------------------------
# create_calendar
# ---------------------------------------------------------------------------


def test_create_calendar_returns_datelist(cal):
    date_list = cal.create_calendar(Calendar.datelist_example)
    assert isinstance(date_list, Calendar.DateList)
    assert len(date_list) == 4
    assert all(isinstance(entry, CalendarEntry) for entry in date_list)


def test_create_calendar_recurring_date_uses_weekday_wildcard(cal):
    date_list = cal.create_calendar(
        {"dates": [{"date": "2021/3/10", "recurring": True}]}
    )
    assert _date_tuple(date_list[0].date) == (121, 3, 10, 255)
    assert date_list[0].dateRange is None


@pytest.mark.parametrize(
    "date_str, expected_weekday",
    [
        ("2024/3/11", 1),  # Monday
        ("2024/3/12", 2),
        ("2024/3/13", 3),
        ("2024/3/14", 4),
        ("2024/3/15", 5),
        ("2024/3/16", 6),
        ("2024/3/17", 7),  # Sunday -> BACnet 7, never wrapped past 7
        ("2021/3/14", 7),
        ("2024/12/25", 3),  # Christmas 2024 is a Wednesday
    ],
)
def test_create_calendar_non_recurring_weekday(cal, date_str, expected_weekday):
    date_list = cal.create_calendar({"dates": [{"date": date_str, "recurring": False}]})
    year, month, day = (int(x) for x in date_str.split("/"))
    assert _date_tuple(date_list[0].date) == (
        year - 1900,
        month,
        day,
        expected_weekday,
    )
    assert 1 <= date_list[0].date[3] <= 7


def test_create_calendar_sunday_matches_bacpypes3_date(cal):
    date_list = cal.create_calendar(
        {"dates": [{"date": "2024/3/17", "recurring": False}]}
    )
    assert str(date_list[0].date) == "2024-3-17 sun"
    assert _date_tuple(date_list[0].date) == _date_tuple(Date("2024-03-17"))


def test_create_calendar_date_ranges(cal):
    date_list = cal.create_calendar(
        {
            "dateRanges": [
                {"startDate": "2024/12/23", "endDate": "2025/1/5"},
                {"startDate": "2024/3/16", "endDate": "2024/3/17"},
            ]
        }
    )
    assert len(date_list) == 2
    first = date_list[0]
    assert first.date is None
    assert isinstance(first.dateRange, DateRange)
    assert _date_tuple(first.dateRange.startDate) == (124, 12, 23, 1)  # Monday
    assert _date_tuple(first.dateRange.endDate) == (125, 1, 5, 7)  # Sunday
    second = date_list[1]
    assert _date_tuple(second.dateRange.startDate) == (124, 3, 16, 6)
    assert _date_tuple(second.dateRange.endDate) == (124, 3, 17, 7)


def test_create_calendar_mixed_dates_then_ranges(cal):
    date_list = cal.create_calendar(Calendar.datelist_example)
    # dates first (in order), then dateRanges (in order)
    assert _date_tuple(date_list[0].date) == (121, 3, 14, 7)
    assert _date_tuple(date_list[1].date) == (121, 3, 10, 255)
    assert date_list[2].date is None
    assert _date_tuple(date_list[2].dateRange.startDate) == (121, 3, 16, 2)
    assert _date_tuple(date_list[2].dateRange.endDate) == (121, 3, 21, 7)
    assert _date_tuple(date_list[3].dateRange.startDate) == (121, 3, 5, 5)
    assert _date_tuple(date_list[3].dateRange.endDate) == (121, 3, 7, 7)


@pytest.mark.parametrize(
    "dict_calendar",
    [{}, {"dates": []}, {"dateRanges": []}, {"dates": [], "dateRanges": []}],
)
def test_create_calendar_empty_inputs(cal, dict_calendar):
    date_list = cal.create_calendar(dict_calendar)
    assert isinstance(date_list, Calendar.DateList)
    assert len(date_list) == 0


def test_create_calendar_missing_recurring_raises_keyerror(cal):
    with pytest.raises(KeyError):
        cal.create_calendar({"dates": [{"date": "2024/7/4"}]})


def test_create_calendar_missing_range_end_raises_keyerror(cal):
    with pytest.raises(KeyError):
        cal.create_calendar({"dateRanges": [{"startDate": "2024/7/4"}]})


@pytest.mark.parametrize(
    "bad_date",
    ["2024-07-04", "2024/07", "2024/7/4/1", "July 4 2024", "2024/2/30", "2024/13/1"],
)
def test_create_calendar_malformed_date_raises_valueerror(cal, bad_date):
    with pytest.raises(ValueError):
        cal.create_calendar({"dates": [{"date": bad_date, "recurring": False}]})


def test_create_calendar_malformed_range_raises_valueerror(cal):
    with pytest.raises(ValueError):
        cal.create_calendar(
            {"dateRanges": [{"startDate": "2024/7/4", "endDate": "tomorrow"}]}
        )


# ---------------------------------------------------------------------------
# decode_dateList
# ---------------------------------------------------------------------------


def test_decode_dateList_round_trip(cal):
    decoded = cal.decode_dateList(cal.create_calendar(Calendar.datelist_example))
    assert decoded == Calendar.datelist_example


def test_decode_dateList_empty(cal):
    assert cal.decode_dateList(Calendar.DateList([])) == {
        "dates": [],
        "dateRanges": [],
    }


# ---------------------------------------------------------------------------
# requests / network wrappers (mocked)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    reason="Suspected bug: Any(None).cast_in(dateList) raises EncodingError "
    "('Any has no content') with bacpypes3",
)
def test_make_calendar_request(cal):
    date_list = cal.create_calendar(Calendar.datelist_example)
    request = cal.make_calendar_request("192.168.1.20", 3, date_list)
    assert request.objectIdentifier[0] == ObjectType("calendar")
    assert request.objectIdentifier[1] == 3
    assert request.propertyIdentifier == PropertyIdentifier("dateList")
    assert request.pduDestination == Address("192.168.1.20")
    assert isinstance(request.propertyValue, Any)


def test_send_calendar_request_uses_application(cal):
    app = MagicMock()
    cal.this_application = SimpleNamespace(app=app)
    request = SimpleNamespace(pduDestination=Address("192.168.1.20"))
    cal.send_calendar_request(request)
    app.request.assert_called_once_with(request)


@pytest.mark.xfail(
    strict=False,
    reason="Suspected bug: write path fails in make_calendar_request "
    "(Any(None) EncodingError)",
)
def test_write_calendar_dateList_sends_request(cal):
    app = MagicMock()
    cal.this_application = SimpleNamespace(app=app)
    cal.write_calendar_dateList("192.168.1.20", 3, Calendar.datelist_example)
    app.request.assert_called_once()


async def test_read_calendar_dateList_decodes(cal):
    date_list = cal.create_calendar(Calendar.datelist_example)
    calls = []

    async def fake_read(args):
        calls.append(args)
        return date_list

    cal.read = fake_read
    result = await cal.read_calendar_dateList("192.168.1.20", 3)
    assert calls == ["192.168.1.20 calendar:3 dateList"]
    assert result == Calendar.datelist_example


async def test_read_calendar_dateList_error_returns_empty_dict(cal):
    async def failing_read(args):
        await asyncio.sleep(0)
        raise RuntimeError("no response")

    cal.read = failing_read
    assert await cal.read_calendar_dateList("192.168.1.20", 3) == {}
