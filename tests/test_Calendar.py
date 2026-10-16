#!/usr/bin/env python
# -*- coding utf-8 -*-

"""
Tests for BAC0.core.functions.Calendar (holiday calendars / dateList)
"""

import asyncio
import datetime
from typing import AsyncGenerator

import pytest
from bacpypes3.basetypes import CalendarEntry, PropertyIdentifier, WeekNDay
from bacpypes3.errors import EncodingError
from bacpypes3.local.schedule import date_in_calendar_entry
from bacpypes3.object import CalendarObject
from bacpypes3.pdu import Address
from bacpypes3.primitivedata import Date, ObjectIdentifier, ObjectType

from BAC0.core.functions.Calendar import Calendar

MIXED_CALENDAR = {
    "dates": [
        {"date": "2021/3/14", "recurring": False},  # Sunday
        {"date": "2021/3/10", "recurring": True},
        {"date": "2024/2/29", "recurring": False},  # leap day, Thursday
        {"date": "2025/12/25", "recurring": True},
    ],
    "dateRanges": [
        {"startDate": "2021/3/16", "endDate": "2021/3/21"},
        {"startDate": "2023/12/31", "endDate": "2024/1/1"},
    ],
}


@pytest.fixture
def cal():
    return Calendar()


def _dates(entries):
    return [tuple(e.date) for e in entries if e.date is not None]


def _ranges(entries):
    return [
        (tuple(e.dateRange.startDate), tuple(e.dateRange.endDate))
        for e in entries
        if e.dateRange is not None
    ]


# ---------------------------------------------------------------------------
# create_calendar
# ---------------------------------------------------------------------------


def test_create_calendar_example(cal):
    date_list = cal.create_calendar(Calendar.datelist_example)

    assert isinstance(date_list, Calendar.DateList)
    assert len(date_list) == 4
    assert all(isinstance(e, CalendarEntry) for e in date_list)
    assert _dates(date_list) == [(121, 3, 14, 7), (121, 3, 10, 255)]
    assert _ranges(date_list) == [
        ((121, 3, 16, 2), (121, 3, 21, 7)),
        ((121, 3, 5, 5), (121, 3, 7, 7)),
    ]


def test_create_calendar_recurring_sets_weekday_wildcard(cal):
    date_list = cal.create_calendar(
        {"dates": [{"date": "2025/12/25", "recurring": True}]}
    )
    assert tuple(date_list[0].date) == (125, 12, 25, 255)


def test_create_calendar_leap_day(cal):
    date_list = cal.create_calendar(
        {"dates": [{"date": "2024/2/29", "recurring": False}]}
    )
    # 2024-02-29 is a Thursday -> BACnet day of week 4
    assert tuple(date_list[0].date) == (124, 2, 29, 4)
    assert date_in_calendar_entry(Date((124, 2, 29, 4)), date_list[0])


def test_create_calendar_invalid_non_recurring_date_raises(cal):
    with pytest.raises(ValueError):
        cal.create_calendar({"dates": [{"date": "2023/2/29", "recurring": False}]})


@pytest.mark.parametrize(
    "date_str, expected_weekday",
    [
        ("2021/3/15", 1),  # Monday
        ("2021/3/20", 6),  # Saturday
        ("2021/3/21", 7),  # Sunday
    ],
)
def test_create_calendar_weekday_numbering(cal, date_str, expected_weekday):
    # BACnet numbers days Monday=1 .. Sunday=7. Python's weekday()+1 already
    # yields 7 for Sunday, so the `if weekday > 7: weekday = 1` wrap in
    # create_calendar is never reached.
    date_list = cal.create_calendar(
        {
            "dates": [{"date": date_str, "recurring": False}],
            "dateRanges": [{"startDate": date_str, "endDate": date_str}],
        }
    )
    assert date_list[0].date[3] == expected_weekday
    assert date_list[1].dateRange.startDate[3] == expected_weekday
    assert date_list[1].dateRange.endDate[3] == expected_weekday
    year, month, day = (int(x) for x in date_str.split("/"))
    assert datetime.date(year, month, day).isoweekday() == expected_weekday


@pytest.mark.parametrize("year", [1900, 1999, 2000, 2024, 2154])
def test_year_offset_round_trip(cal, year):
    calendar = {
        "dates": [{"date": f"{year}/6/1", "recurring": False}],
        "dateRanges": [{"startDate": f"{year}/6/1", "endDate": f"{year}/6/2"}],
    }
    date_list = cal.create_calendar(calendar)
    assert date_list[0].date[0] == year - 1900
    assert date_list[1].dateRange.startDate[0] == year - 1900
    assert date_list[1].dateRange.endDate[0] == year - 1900
    assert cal.decode_dateList(date_list) == calendar


def test_create_calendar_empty(cal):
    assert len(cal.create_calendar({})) == 0
    assert len(cal.create_calendar({"dates": [], "dateRanges": []})) == 0


# ---------------------------------------------------------------------------
# decode_dateList
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "calendar",
    [Calendar.datelist_example, MIXED_CALENDAR],
    ids=["datelist_example", "mixed"],
)
def test_calendar_round_trip(cal, calendar):
    assert cal.decode_dateList(cal.create_calendar(calendar)) == calendar


def test_decode_empty_dateList(cal):
    assert cal.decode_dateList(Calendar.DateList([])) == {
        "dates": [],
        "dateRanges": [],
    }


def test_decode_recurring_flag_comes_from_weekday(cal):
    entries = Calendar.DateList(
        [CalendarEntry(date=(124, 7, 4, 255)), CalendarEntry(date=(124, 7, 4, 4))]
    )
    assert cal.decode_dateList(entries)["dates"] == [
        {"date": "2024/7/4", "recurring": True},
        {"date": "2024/7/4", "recurring": False},
    ]


@pytest.mark.xfail(
    strict=False,
    reason="Suspected bug: weekNDay calendar entries (e.g. 'every Monday') are "
    "silently dropped by decode_dateList",
)
def test_decode_dateList_keeps_weekNDay_entries(cal):
    entries = Calendar.DateList(
        [
            CalendarEntry(date=(124, 7, 4, 4)),
            CalendarEntry(weekNDay=WeekNDay(b"\xff\xff\x01")),  # every Monday
        ]
    )
    decoded = cal.decode_dateList(entries)
    total = sum(len(v) for v in decoded.values())
    assert total == len(entries)


@pytest.mark.xfail(
    strict=False,
    reason="Suspected bug: recurring dates only wildcard the weekday and keep "
    "the year, so they never match the same day in another year",
)
def test_recurring_date_matches_following_year(cal):
    date_list = cal.create_calendar(
        {"dates": [{"date": "2025/12/25", "recurring": True}]}
    )
    christmas_2026 = Date((126, 12, 25, 5))
    assert date_in_calendar_entry(christmas_2026, date_list[0])


# ---------------------------------------------------------------------------
# make_calendar_request
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    raises=EncodingError,
    reason="Suspected bug: make_calendar_request builds Any(None) and then "
    "cast_in(), which bacpypes3 rejects with EncodingError('Any has no content')",
)
def test_make_calendar_request(cal):
    date_list = cal.create_calendar(Calendar.datelist_example)
    request = cal.make_calendar_request(
        destination="192.168.1.10:47809", object_instance=3, dateList=date_list
    )
    assert request.objectIdentifier == ObjectIdentifier((ObjectType("calendar"), 3))
    assert request.propertyIdentifier == PropertyIdentifier("dateList")
    assert request.pduDestination == Address("192.168.1.10:47809")
    # Unlike make_weeklySchedule_request, no priority (15) is set here;
    # dateList is not commandable, so none is expected.
    assert request.priority is None
    decoded = request.propertyValue.cast_out(Calendar.DateList)
    assert cal.decode_dateList(decoded) == Calendar.datelist_example


# ---------------------------------------------------------------------------
# read_calendar_dateList error handling (no network)
# ---------------------------------------------------------------------------


class _UnreachableCalendar(Calendar):
    async def read(self, *args, **kwargs):
        raise ConnectionError("device unreachable")


@pytest.mark.xfail(
    strict=False,
    reason="Suspected bug: read_calendar_dateList swallows every exception and "
    "returns {}, so a failed read looks like an empty calendar",
)
@pytest.mark.asyncio(loop_scope="session")
async def test_read_calendar_dateList_propagates_read_error():
    with pytest.raises(ConnectionError):
        await _UnreachableCalendar().read_calendar_dateList("192.168.1.10", 1)


# ---------------------------------------------------------------------------
# Integration: read/write against a CalendarObject on device_app
# ---------------------------------------------------------------------------


class _CalendarClient(Calendar):
    """Calendar is commented out of the Lite mixins, so borrow the BAC0
    network client's read() and application to exercise the Calendar methods."""

    def __init__(self, bacnet):
        self._bacnet = bacnet
        self.this_application = bacnet.this_application

    async def read(self, *args, **kwargs):
        return await self._bacnet.read(*args, **kwargs)


def _device_address(device_app):
    return f"{device_app.localIPAddr.addrTuple[0]}:47809"


def _calendar_object(device_app, instance):
    app = device_app.this_application.app
    oid = ObjectIdentifier((ObjectType("calendar"), instance))
    obj = app.get_object_id(oid)
    if obj is None:
        obj = CalendarObject(
            objectIdentifier=oid,
            objectName=f"CAL-{instance}",
            presentValue=False,
            dateList=[],
        )
        app.add_object(obj)
    return obj


@pytest.mark.asyncio(loop_scope="session")
async def test_read_calendar_dateList_network_round_trip(
    network_and_devices: AsyncGenerator,
):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    client = _CalendarClient(bacnet)
    obj = _calendar_object(device_app, 9101)
    obj.dateList = client.create_calendar(MIXED_CALENDAR)

    result = await client.read_calendar_dateList(_device_address(device_app), 9101)

    assert result == MIXED_CALENDAR


@pytest.mark.asyncio(loop_scope="session")
async def test_read_calendar_dateList_empty(network_and_devices: AsyncGenerator):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    _calendar_object(device_app, 9102)
    result = await _CalendarClient(bacnet).read_calendar_dateList(
        _device_address(device_app), 9102
    )
    assert result == {"dates": [], "dateRanges": []}


@pytest.mark.xfail(
    strict=False,
    raises=EncodingError,
    reason="Suspected bug: write_calendar_dateList fails in make_calendar_request "
    "(Any(None) -> EncodingError) before anything is sent",
)
@pytest.mark.asyncio(loop_scope="session")
async def test_write_calendar_dateList_network_round_trip(
    network_and_devices: AsyncGenerator,
):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    client = _CalendarClient(bacnet)
    _calendar_object(device_app, 9103)
    address = _device_address(device_app)

    client.write_calendar_dateList(address, 9103, MIXED_CALENDAR)
    await asyncio.sleep(0.5)

    assert await client.read_calendar_dateList(address, 9103) == MIXED_CALENDAR


@pytest.mark.xfail(
    strict=False,
    reason="Suspected gap: Calendar is commented out of the Lite mixins "
    "(BAC0/scripts/Lite.py), so bacnet.write_calendar_dateList / "
    "read_calendar_dateList are not available on a BAC0 connection",
)
@pytest.mark.asyncio(loop_scope="session")
async def test_lite_exposes_calendar_methods(network_and_devices: AsyncGenerator):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    assert hasattr(bacnet, "write_calendar_dateList")
    assert hasattr(bacnet, "read_calendar_dateList")
