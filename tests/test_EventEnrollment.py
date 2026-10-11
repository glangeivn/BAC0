#!/usr/bin/env python
# -*- coding utf-8 -*-

"""
End-to-end tests: write eventEnrollment properties over BACnet.

The local ObjectFactory has no event enrollment helper, so a bacpypes3
EventEnrollmentObject is hosted directly on the test device application.
"""

import asyncio

import pytest
from bacpypes3.basetypes import (
    DeviceObjectPropertyReference,
    EventParameter,
    EventParameterOutOfRange,
)
from bacpypes3.object import EventEnrollmentObject
from bacpypes3.primitivedata import ObjectIdentifier, PropertyIdentifier

EVENT_ENROLLMENT_INSTANCE = 1


async def _wait_for(predicate, attempts=50):
    for _ in range(attempts):
        if predicate():
            return True
        await asyncio.sleep(0.1)
    return False


def _event_enrollment(device_app):
    app = device_app.this_application.app
    obj = app.get_object_id(
        ObjectIdentifier(("eventEnrollment", EVENT_ENROLLMENT_INSTANCE))
    )
    if obj is None:
        obj = EventEnrollmentObject(
            objectIdentifier=("eventEnrollment", EVENT_ENROLLMENT_INSTANCE),
            objectName="TEST-EVENT-ENROLLMENT",
        )
        app.add_object(obj)
    return obj


@pytest.mark.asyncio(loop_scope="session")
async def test_write_event_parameters(network_and_devices):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    obj = _event_enrollment(device_app)
    event_parameters = EventParameter(
        outOfRange=EventParameterOutOfRange(
            timeDelay=5, lowLimit=10.0, highLimit=30.0, deadband=1.0
        )
    )

    bacnet.write_event_parameters(
        test_device.properties.address, EVENT_ENROLLMENT_INSTANCE, event_parameters
    )

    assert await _wait_for(lambda: obj.eventParameters is not None)
    out_of_range = obj.eventParameters.outOfRange
    assert out_of_range.timeDelay == 5
    assert out_of_range.lowLimit == 10.0
    assert out_of_range.highLimit == 30.0
    assert out_of_range.deadband == 1.0


@pytest.mark.asyncio(loop_scope="session")
async def test_write_obj_prop_ref(network_and_devices):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    obj = _event_enrollment(device_app)
    reference = DeviceObjectPropertyReference(
        objectIdentifier=("analogValue", 0), propertyIdentifier="presentValue"
    )

    bacnet.write_obj_prop_ref(
        test_device.properties.address, EVENT_ENROLLMENT_INSTANCE, reference
    )

    assert await _wait_for(lambda: obj.objectPropertyReference is not None)
    assert obj.objectPropertyReference.objectIdentifier == ObjectIdentifier(
        ("analogValue", 0)
    )
    assert obj.objectPropertyReference.propertyIdentifier == PropertyIdentifier(
        "presentValue"
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_write_event_enrollment_rejects_wrong_types(network_and_devices):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    address = test_device.properties.address

    assert bacnet.write_event_parameters(address, 1, "not-an-event-parameter") is None
    assert bacnet.write_obj_prop_ref(address, 1, ("analogValue", 0)) is None
