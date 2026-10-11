#!/usr/bin/env python
# -*- coding utf-8 -*-

"""
End-to-end tests: write description text to points and devices over BACnet.
"""

import pytest
from bacpypes3.primitivedata import ObjectIdentifier


def _local_object(app, obj_type, instance):
    return app.get_object_id(ObjectIdentifier((obj_type, instance)))


@pytest.mark.asyncio(loop_scope="session")
async def test_point_update_description(network_and_devices):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    point = test_device["AV"]
    server_object = _local_object(
        device_app.this_application.app, "analogValue", int(point.properties.address)
    )

    await point.update_description("Updated by update_description")

    assert server_object.description == "Updated by update_description"
    assert point.properties.description == "Updated by update_description"


@pytest.mark.asyncio(loop_scope="session")
async def test_point_write_description(network_and_devices):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    point = test_device["AV"]
    server_object = _local_object(
        device_app.this_application.app, "analogValue", int(point.properties.address)
    )

    await point.write("Updated by write", prop="description")

    assert server_object.description == "Updated by write"
    assert point.properties.description == "Updated by write"


@pytest.mark.asyncio(loop_scope="session")
async def test_device_update_description(network_and_devices):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    device_object = device_app.this_application.app.device_object

    await test_device.update_description("Device description")

    assert device_object.description == "Device description"
    assert test_device.properties.description == "Device description"

    await test_device.write_property("description", "Device description 2")

    assert device_object.description == "Device description 2"


@pytest.mark.asyncio(loop_scope="session")
async def test_send_text_write_request_returns_awaitable(network_and_devices):
    loop, bacnet, device_app, device30_app, test_device, test_device_30 = (
        network_and_devices
    )
    point = test_device["AV"]
    server_object = _local_object(
        device_app.this_application.app, "analogValue", int(point.properties.address)
    )

    await bacnet.send_text_write_request(
        addr=test_device.properties.address,
        obj_type="analogValue",
        obj_inst=int(point.properties.address),
        value="Written directly",
    )

    assert server_object.description == "Written directly"
