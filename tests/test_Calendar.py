#!/usr/bin/env python
# -*- coding utf-8 -*-

"""
Tests for BAC0/core/functions/Calendar.py (holiday calendars / dateList).

Pure encode/decode tests instantiate Calendar() directly. Only the send/read
integration tests use the session-scoped ``network_and_devices`` fixture.
Suspected library bugs are documented (not fixed) with xfail or
current-behaviour assertions.
"""

import asyncio
import copy
import datetime
import inspect

import pytest
from bacpypes3.apdu import WritePropertyRequest
from bacpypes3.basetypes import CalendarEntry, DateRange, WeekNDay
from bacpypes3.constructeddata import Any
from bacpypes3.errors import EncodingError
from bacpypes3.local.schedule import date_in_calendar_entry
from bacpypes3.object import CalendarObject
from bacpypes3.pdu import Address
from bacpypes3.primitivedata import (
    Date,
    ObjectIdentifier,
    ObjectType,
    PropertyIdentifier,
)

from BAC0.core.functions.Calendar import Calendar

DEST = "127.0.0.1:47809"


def _bacnet_weekday(year, month, day):
    """BACnet day-of-week: Monday=1 ... Sunday=7."""
    return datetime.date(year, month, day).isoweekday()


def _round_trip(dict_calendar):
    c = Calendar()
    return c.decode_dateList(c.create_calendar(dict_calendar))


# ---------------------------------------------------------------------------
# create_calendar / decode_dateList
# ---------------------------------------------------------------------------


def test_create_calendar_returns_datelist():
    dl = Calendar().create_calendar(Calendar.datelist_example)
    assert isinstance(dl, Calendar.DateList)
    assert len(dl) == 4
    assert all(isinstance(e, CalendarEntry) for e in dl)
    assert [e.date is not None for e in dl] == [True, True, False, False]
    assert [e.dateRange is not None for e in dl] == [False, False, True, True]


def test_create_calendar_single_date_encoding():
    dl = Calendar().create_calendar(
        {"dates": [{"date": "2024/7/4", "recurring": False}]}
    )
    (entry,) = dl
    # BACnet dates store year - 1900
    assert tuple(entry.date) == (124, 7, 4, _bacnet_weekday(2024, 7, 4))
    assert entry.dateRange is None


@pytest.mark.parametrize(
    "date_str, expected_year_byte",
    [("1900/1/1", 0), ("1999/12/31", 99), ("2000/1/1", 100), ("2024/2/29", 124)],
)
def test_create_calendar_year_1900_offset(date_str, expected_year_byte):
    dl = Calendar().create_calendar({"dates": [{"date": date_str, "recurring": False}]})
    assert dl[0].date[0] == expected_year_byte
    assert _round_trip({"dates": [{"date": date_str, "recurring": False}]}) == {
        "dates": [{"date": date_str, "recurring": False}],
        "dateRanges": [],
    }


def test_create_calendar_recurring_date_sets_weekday_255():
    dl = Calendar().create_calendar(
        {"dates": [{"date": "2024/12/25", "recurring": True}]}
    )
    assert tuple(dl[0].date) == (124, 12, 25, 255)


@pytest.mark.parametrize(
    "date_str",
    ["2024/3/17", "2021/3/14", "2023/12/31"],  # all Sundays
)
def test_create_calendar_non_recurring_sunday_is_weekday_7(date_str):
    dl = Calendar().create_calendar({"dates": [{"date": date_str, "recurring": False}]})
    assert dl[0].date[3] == 7
    assert _round_trip({"dates": [{"date": date_str, "recurring": False}]})[
        "dates"
    ] == [{"date": date_str, "recurring": False}]


@pytest.mark.parametrize(
    "date_str, weekday",
    [
        ("2024/3/11", 1),  # Monday
        ("2024/3/12", 2),
        ("2024/3/13", 3),
        ("2024/3/14", 4),
        ("2024/3/15", 5),
        ("2024/3/16", 6),
        ("2024/3/17", 7),  # Sunday
    ],
)
def test_create_calendar_weekday_numbering(date_str, weekday):
    dl = Calendar().create_calendar({"dates": [{"date": date_str, "recurring": False}]})
    assert dl[0].date[3] == weekday


def test_create_calendar_date_range_encoding():
    dl = Calendar().create_calendar(
        {"dateRanges": [{"startDate": "2024/12/23", "endDate": "2025/1/5"}]}
    )
    (entry,) = dl
    assert entry.date is None
    assert isinstance(entry.dateRange, DateRange)
    assert tuple(entry.dateRange.startDate) == (124, 12, 23, 1)
    assert tuple(entry.dateRange.endDate) == (125, 1, 5, 7)


def test_round_trip_single_dates():
    cal = {
        "dates": [
            {"date": "2024/1/1", "recurring": False},
            {"date": "2024/7/4", "recurring": False},
        ]
    }
    assert _round_trip(cal) == {"dates": cal["dates"], "dateRanges": []}


def test_round_trip_recurring_dates():
    cal = {
        "dates": [
            {"date": "2024/12/25", "recurring": True},
            {"date": "2024/1/1", "recurring": True},
        ]
    }
    assert _round_trip(cal) == {"dates": cal["dates"], "dateRanges": []}


def test_round_trip_date_ranges():
    cal = {
        "dateRanges": [
            {"startDate": "2024/12/23", "endDate": "2025/1/5"},
            {"startDate": "2024/7/1", "endDate": "2024/7/31"},
        ]
    }
    assert _round_trip(cal) == {"dates": [], "dateRanges": cal["dateRanges"]}


def test_round_trip_mixed_entries_example():
    assert _round_trip(Calendar.datelist_example) == Calendar.datelist_example


def test_round_trip_mixed_entries_dates_listed_before_ranges():
    cal = {
        "dateRanges": [{"startDate": "2024/8/1", "endDate": "2024/8/15"}],
        "dates": [
            {"date": "2024/11/28", "recurring": False},
            {"date": "2024/12/25", "recurring": True},
        ],
    }
    dl = Calendar().create_calendar(cal)
    assert [e.date is not None for e in dl] == [True, True, False]
    assert _round_trip(cal) == cal


@pytest.mark.parametrize(
    "cal", [{}, {"dates": []}, {"dateRanges": []}, {"dates": [], "dateRanges": []}]
)
def test_create_calendar_empty(cal):
    dl = Calendar().create_calendar(cal)
    assert isinstance(dl, Calendar.DateList)
    assert len(dl) == 0
    assert Calendar().decode_dateList(dl) == {"dates": [], "dateRanges": []}


def test_create_calendar_ignores_unknown_keys():
    dl = Calendar().create_calendar({"holidays": [{"date": "2024/1/1"}]})
    assert len(dl) == 0


def test_create_calendar_does_not_mutate_input():
    cal = copy.deepcopy(Calendar.datelist_example)
    Calendar().create_calendar(cal)
    assert cal == Calendar.datelist_example


def test_create_calendar_matches_dates_on_device_side():
    """bacpypes3's own calendar matcher agrees with the encoded entries."""
    dl = Calendar().create_calendar(Calendar.datelist_example)
    single, recurring, range1, range2 = dl
    assert date_in_calendar_entry(Date((121, 3, 14, 7)), single)
    assert not date_in_calendar_entry(Date((121, 3, 15, 1)), single)
    assert date_in_calendar_entry(Date((121, 3, 10, 3)), recurring)
    assert date_in_calendar_entry(Date((121, 3, 18, 4)), range1)
    assert not date_in_calendar_entry(Date((121, 3, 22, 1)), range1)
    assert date_in_calendar_entry(Date((121, 3, 5, 5)), range2)


@pytest.mark.xfail(
    strict=False,
    reason="Suspected bug: recurring dates keep the year (Calendar.py lines 51-57)",
)
def test_recurring_date_should_match_every_year():
    """
    Suspected bug: ``recurring: True`` only sets the day-of-week to 255
    (any weekday) but keeps the specific year (Calendar.py lines 51-57).
    A holiday such as Christmas therefore matches only in the year given,
    not every year. BACnet "every year" requires year=255 as well.
    """
    dl = Calendar().create_calendar(
        {"dates": [{"date": "2024/12/25", "recurring": True}]}
    )
    assert date_in_calendar_entry(Date((124, 12, 25, 3)), dl[0])
    assert date_in_calendar_entry(Date((125, 12, 25, 4)), dl[0])


def test_recurring_date_current_behaviour_matches_one_year_only():
    dl = Calendar().create_calendar(
        {"dates": [{"date": "2024/12/25", "recurring": True}]}
    )
    assert date_in_calendar_entry(Date((124, 12, 25, 3)), dl[0])
    assert not date_in_calendar_entry(Date((125, 12, 25, 4)), dl[0])


def test_decode_dateList_wildcard_year_reads_as_2155():
    """
    Suspected bug (current behaviour asserted): a date written by another
    tool with year=255 ("any year") is decoded as "2155/..." because
    decode_dateList always adds 1900 (Calendar.py lines 143-145). The
    "recurring" flag is derived from the weekday only.
    """
    entry = CalendarEntry(date=(255, 12, 25, 255))
    assert Calendar().decode_dateList([entry]) == {
        "dates": [{"date": "2155/12/25", "recurring": True}],
        "dateRanges": [],
    }


def test_decode_dateList_specific_weekday_is_not_recurring():
    entry = CalendarEntry(date=(124, 12, 25, 3))
    assert Calendar().decode_dateList([entry])["dates"] == [
        {"date": "2024/12/25", "recurring": False}
    ]


def test_decode_dateList_drops_weekNDay_entries():
    """
    Suspected gap (current behaviour asserted): CalendarEntry has a third
    form, weekNDay (e.g. "4th Thursday of November"). decode_dateList
    (Calendar.py lines 136-159) silently drops these entries, so reading and
    re-writing a calendar would delete them.
    """
    entries = [
        CalendarEntry(weekNDay=WeekNDay(bytes([11, 4, 4]))),
        CalendarEntry(date=(124, 1, 1, 1)),
    ]
    assert Calendar().decode_dateList(entries) == {
        "dates": [{"date": "2024/1/1", "recurring": False}],
        "dateRanges": [],
    }


def test_decode_dateList_accepts_plain_list():
    entries = [
        CalendarEntry(
            dateRange=DateRange(startDate=(124, 6, 1, 6), endDate=(124, 6, 30, 7))
        )
    ]
    assert Calendar().decode_dateList(entries) == {
        "dates": [],
        "dateRanges": [{"startDate": "2024/6/1", "endDate": "2024/6/30"}],
    }


def test_create_calendar_reversed_range_is_accepted():
    """Current behaviour: no validation that endDate >= startDate."""
    cal = {"dateRanges": [{"startDate": "2024/12/31", "endDate": "2024/1/1"}]}
    assert _round_trip(cal)["dateRanges"] == cal["dateRanges"]


@pytest.mark.parametrize(
    "cal, exc",
    [
        ({"dates": [{"date": "2024-12-25", "recurring": False}]}, ValueError),
        ({"dates": [{"date": "2023/2/29", "recurring": False}]}, ValueError),
        ({"dates": [{"date": "2024/12/25"}]}, KeyError),
        ({"dateRanges": [{"startDate": "2024/1/1"}]}, KeyError),
        (
            {"dateRanges": [{"startDate": "2024/13/1", "endDate": "2024/1/1"}]},
            ValueError,
        ),
    ],
)
def test_create_calendar_malformed_input_raises(cal, exc):
    with pytest.raises(exc):
        Calendar().create_calendar(cal)


# ---------------------------------------------------------------------------
# make_calendar_request / send_calendar_request / write_calendar_dateList
# ---------------------------------------------------------------------------


def test_make_calendar_request_raises_encoding_error():
    """
    Suspected bug (current behaviour asserted): make_calendar_request builds
    ``Any(None)`` and then calls ``cast_in`` (Calendar.py lines 92-93). With
    bacpypes3 0.0.110 this raises EncodingError("Any has no content"), so
    no calendar write request can be built and write_calendar_dateList always
    fails. ``Any(dateList)`` (as Schedule.py does) works.
    """
    c = Calendar()
    dl = c.create_calendar(Calendar.datelist_example)
    with pytest.raises(EncodingError, match="Any has no content"):
        c.make_calendar_request(DEST, 1, dl)


@pytest.mark.xfail(
    strict=False, reason="Suspected bug: Any(None).cast_in() at Calendar.py lines 92-93"
)
def test_make_calendar_request_fields():
    c = Calendar()
    dl = c.create_calendar(Calendar.datelist_example)
    request = c.make_calendar_request(DEST, 5, dl)
    assert isinstance(request, WritePropertyRequest)
    assert request.objectIdentifier == ObjectIdentifier(("calendar", 5))
    assert request.objectIdentifier[0] == ObjectType("calendar")
    assert request.propertyIdentifier == PropertyIdentifier("dateList")
    assert request.pduDestination == Address(DEST)
    assert (
        c.decode_dateList(request.propertyValue.cast_out(Calendar.DateList))
        == Calendar.datelist_example
    )


def _calendar_request(destination, instance, dict_calendar):
    """Equivalent of make_calendar_request built with Any(dateList)."""
    request = WritePropertyRequest(
        objectIdentifier=("calendar", instance), propertyIdentifier="dateList"
    )
    request.pduDestination = Address(destination)
    request.propertyValue = Any(Calendar().create_calendar(dict_calendar))
    return request


class _FakeApp:
    def __init__(self):
        self.requests = []

    def request(self, request):
        self.requests.append(request)


class _FakeBAC0App:
    def __init__(self):
        self.app = _FakeApp()


def _offline_calendar():
    c = Calendar()
    c.this_application = _FakeBAC0App()
    return c


def test_send_calendar_request_hands_request_to_app():
    c = _offline_calendar()
    request = _calendar_request(DEST, 1, Calendar.datelist_example)
    assert c.send_calendar_request(request) is None
    assert c.this_application.app.requests == [request]


def test_send_calendar_request_timeout_is_unused():
    """
    Suspected bug (current behaviour asserted): ``timeout`` (Calendar.py
    line 96) is accepted but never used; the request is fire-and-forget and
    the device's response/error is never checked.
    """
    assert (
        inspect.signature(Calendar.send_calendar_request).parameters["timeout"].default
        == 10
    )
    c = _offline_calendar()
    result = c.send_calendar_request(
        _calendar_request(DEST, 1, Calendar.datelist_example), timeout=0
    )
    assert result is None and not inspect.isawaitable(result)
    assert len(c.this_application.app.requests) == 1


def test_write_calendar_dateList_offline_current_behaviour():
    """Current behaviour: fails before sending because of make_calendar_request."""
    c = _offline_calendar()
    with pytest.raises(EncodingError):
        c.write_calendar_dateList(DEST, 1, Calendar.datelist_example)
    assert c.this_application.app.requests == []


@pytest.mark.xfail(
    strict=False, reason="Suspected bug: Any(None).cast_in() at Calendar.py lines 92-93"
)
def test_write_calendar_dateList_offline_sends_request():
    c = _offline_calendar()
    c.write_calendar_dateList(DEST, 1, Calendar.datelist_example)
    (request,) = c.this_application.app.requests
    assert request.objectIdentifier == ObjectIdentifier(("calendar", 1))


# ---------------------------------------------------------------------------
# read_calendar_dateList with a fake read primitive (no network)
# ---------------------------------------------------------------------------


class _FakeReader(Calendar):
    def __init__(self, value=None, exc=None):
        self.value = value
        self.exc = exc
        self.calls = []

    async def read(self, args):
        self.calls.append(args)
        if self.exc:
            raise self.exc
        return self.value


async def test_read_calendar_dateList_decodes_value():
    dl = Calendar().create_calendar(Calendar.datelist_example)
    c = _FakeReader(value=dl)
    assert await c.read_calendar_dateList("2:5", 3) == Calendar.datelist_example
    assert c.calls == ["2:5 calendar:3 dateList"]


async def test_read_calendar_dateList_returns_empty_dict_on_error():
    c = _FakeReader(exc=RuntimeError("unknown-object"))
    assert await c.read_calendar_dateList("2:5", 3) == {}


async def test_read_calendar_dateList_returns_empty_dict_on_bad_payload():
    c = _FakeReader(value=None)  # not iterable -> TypeError inside decode
    assert await c.read_calendar_dateList("2:5", 3) == {}


# ---------------------------------------------------------------------------
# Integration with the virtual devices (network_and_devices fixture)
# ---------------------------------------------------------------------------

CALENDAR_INSTANCE = 9201


def _device_address(device_app):
    return f"{device_app.localIPAddr.addrTuple[0]}:47809"


def _get_or_add_calendar(app):
    oid = ObjectIdentifier(("calendar", CALENDAR_INSTANCE))
    obj = app.get_object_id(oid)
    if obj is None:
        obj = CalendarObject(
            objectIdentifier=oid,
            objectName="TEST-HOLIDAYS",
            presentValue=False,
            dateList=[],
        )
        app.add_object(obj)
    return obj


def _calendar_bound_to(bacnet):
    """Calendar is not mixed into Lite on main, so bind one to bacnet's app."""
    c = Calendar()
    c.this_application = bacnet.this_application
    c.read = bacnet.read
    return c


@pytest.mark.xfail(
    strict=False,
    reason="Suspected bug: Calendar mixin is commented out of Lite (BAC0/scripts/Lite.py line 97)",
)
@pytest.mark.asyncio(loop_scope="session")
async def test_bacnet_write_calendar_dateList_does_not_raise(network_and_devices):
    """
    Suspected bug: ``Calendar`` is commented out of the Lite mixins
    (BAC0/scripts/Lite.py line 97), so ``bacnet.write_calendar_dateList``
    and ``bacnet.read_calendar_dateList`` do not exist on a BAC0 app
    (AttributeError). Even when bound manually it fails because of
    make_calendar_request (see test_make_calendar_request_raises_encoding_error).
    """
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    _get_or_add_calendar(device_app.this_application.app)
    bacnet.write_calendar_dateList(
        _device_address(device_app), CALENDAR_INSTANCE, Calendar.datelist_example
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_calendar_methods_missing_from_lite(network_and_devices):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    assert not hasattr(bacnet, "write_calendar_dateList")
    assert not hasattr(bacnet, "read_calendar_dateList")


@pytest.mark.xfail(
    strict=False, reason="Suspected bug: Any(None).cast_in() at Calendar.py lines 92-93"
)
@pytest.mark.asyncio(loop_scope="session")
async def test_write_calendar_dateList_is_applied_on_virtual_device(
    network_and_devices,
):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    obj = _get_or_add_calendar(device_app.this_application.app)
    obj.dateList = []
    c = _calendar_bound_to(bacnet)
    c.write_calendar_dateList(
        _device_address(device_app), CALENDAR_INSTANCE, Calendar.datelist_example
    )
    await asyncio.sleep(1)
    assert c.decode_dateList(obj.dateList) == Calendar.datelist_example


@pytest.mark.asyncio(loop_scope="session")
async def test_send_calendar_request_to_virtual_device(network_and_devices):
    """The send path works when given a correctly encoded request."""
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    obj = _get_or_add_calendar(device_app.this_application.app)
    obj.dateList = []
    c = _calendar_bound_to(bacnet)
    holidays = {
        "dates": [
            {"date": "2024/12/25", "recurring": True},
            {"date": "2024/7/4", "recurring": False},
        ],
        "dateRanges": [{"startDate": "2024/12/23", "endDate": "2025/1/3"}],
    }
    c.send_calendar_request(
        _calendar_request(_device_address(device_app), CALENDAR_INSTANCE, holidays)
    )
    for _ in range(20):
        await asyncio.sleep(0.1)
        if obj.dateList:
            break
    assert c.decode_dateList(obj.dateList) == holidays


@pytest.mark.asyncio(loop_scope="session")
async def test_read_calendar_dateList_from_virtual_device(network_and_devices):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    obj = _get_or_add_calendar(device_app.this_application.app)
    obj.dateList = Calendar().create_calendar(Calendar.datelist_example)
    c = _calendar_bound_to(bacnet)
    assert (
        await c.read_calendar_dateList(_device_address(device_app), CALENDAR_INSTANCE)
        == Calendar.datelist_example
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_read_calendar_dateList_missing_object_returns_empty(
    network_and_devices,
):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    c = _calendar_bound_to(bacnet)
    assert await c.read_calendar_dateList(_device_address(device_app), 9299) == {}
