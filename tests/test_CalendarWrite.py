#!/usr/bin/env python
# -*- coding utf-8 -*-

"""
End-to-end test: write a calendar dateList over BACnet and read it back.

The local ObjectFactory has no calendar helper, so a bacpypes3 CalendarObject
is hosted directly on the test device application.
"""

import pytest
from bacpypes3.apdu import ErrorRejectAbortNack
from bacpypes3.object import CalendarObject

DATES = {
    "dates": [
        {"date": "2024/12/25", "recurring": False},
        {"date": "2024/2/29", "recurring": False},
    ],
    "dateRanges": [
        {"startDate": "2024/12/24", "endDate": "2025/1/2"},
    ],
}


@pytest.mark.asyncio(loop_scope="session")
async def test_write_and_read_calendar_dateList(network_and_devices):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    calendar_object = CalendarObject(
        objectIdentifier=("calendar", 1),
        objectName="TEST-HOLIDAYS",
        presentValue=False,
        dateList=[],
    )
    device_app.this_application.app.add_object(calendar_object)
    address = test_device.properties.address

    await bacnet.write_calendar_dateList(address, 1, DATES)

    assert len(calendar_object.dateList) == 3

    assert await bacnet.read_calendar_dateList(address, 1) == DATES


@pytest.mark.asyncio(loop_scope="session")
async def test_rejected_calendar_write_raises(network_and_devices):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    address = test_device.properties.address

    with pytest.raises(ErrorRejectAbortNack):
        await bacnet.write_calendar_dateList(address, 99, DATES)
