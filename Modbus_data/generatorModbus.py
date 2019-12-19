#!/usr/bin/env python
# -*- coding:utf-8 -*-


import random
from DCGAN_utils import files_number
# --------------------------------------------------------------------------#
# import all available messages
# --------------------------------------------------------------------------#
from pymodbus.bit_read_message import *
from pymodbus.bit_write_message import *
from pymodbus.diag_message import *
from pymodbus.file_message import *
from pymodbus.other_message import *
from pymodbus.register_read_message import *
from pymodbus.register_write_message import *
# --------------------------------------------------------------------------#
# import all the available framers
# --------------------------------------------------------------------------#
from pymodbus.transaction import ModbusSocketFramer

"""
!!!!!!!   Run in python 2.7x    !!!!!!!   Can run on self computer.
"""

def random_address(start=0x0000, end=0xFFFF):
    ret = random.randint(start, end)
    return ret


def random_integer(min, max):
    ret = random.randint(min, max)
    return ret


# 0x01 read coil
def read_coils(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 1 byte(字节数 N='count'/8) + N bytes
    '''
    _arguments['address'] = random_address()  # 随机 0x0000-0xFFFF
#    print hex(_arguments['address'])
    _arguments['count'] = random_integer(1000, 2000)  # 随机 1-2000(0x07D0)
#    print hex(_arguments['count'])

    request = ReadCoilsRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    _arguments['values'] = []  # 随机

    for i in range(0, _arguments['count']):
        _arguments['values'].append(random_integer(0, 1))

    response = ReadCoilsResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x02 read discrete input
def read_discrete_inputs(framer, _arguments):
    '''
    :param _arguments:
    :param framer:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 1 byte(字节数 N='count'/8) + N bytes
    '''
    _arguments['address'] = random_address()  # 随机 0x0000-0xFFFF
    _arguments['count'] = random_integer(1000, 2000)  # 随机 1-2000(0x07D0)

    request = ReadDiscreteInputsRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    _arguments['values'] = []  # 随机
    for i in range(0, _arguments['count']):
        _arguments['values'].append(random_integer(0, 1))

    response = ReadDiscreteInputsResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x03 read holding register
def read_holding_registers(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 1 byte(字节数 N='count') + 2*N bytes
    '''
    _arguments['address'] = random_address()  # 随机 0x0000-0xFFFF
    _arguments['count'] = random_address(1, 125)  # 随机 1-125(0x007D)

    request = ReadHoldingRegistersRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    _arguments['values'] = []  # 随机
    for i in range(0, _arguments['count']):
        _arguments['values'].append(random_integer(0, 0xFFFF))

    response = ReadHoldingRegistersResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x04 read input registers
def read_input_registers(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 1 byte(字节数 N='count') + 2*N bytes
    '''
    _arguments['address'] = random_address()  # 随机 0x0000-0xFFFF
    _arguments['count'] = random_integer(100, 125)  # 随机 1-125(0x007D)

    request = ReadInputRegistersRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    _arguments['values'] = []  # 随机
    for i in range(0, _arguments['count']):
        _arguments['values'].append(random_integer(0, 0xFFFF))

    response = ReadInputRegistersResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x05 write single coil
def write_single_coil(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 2 bytes + 2 bytes
    '''
    _arguments['address'] = random_address()  # 随机 0x0000-0xFFFF
    tmp = random_integer(0, 1)
    _arguments['value'] = 0xFF00 if tmp == 1 else 0x0000  # ON or 0x0000 OFF

    request = WriteSingleCoilRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    response = WriteSingleCoilResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x06 write single register
def write_single_register(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 2 bytes + 2 bytes
    '''
    _arguments['address'] = random_address()  # 随机 0x0000-0xFFFF
    _arguments['value'] = random_integer(0, 0xFFFF)  # 随机 0x0000-0xFFFF

    request = WriteSingleRegisterRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    response = WriteSingleRegisterResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x0F write multiple coils
def write_multiple_coils(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes + 1 byte(字节数 N='count'/8) + N bytes
    response 2 bytes + 2 bytes
    '''
    _arguments['address'] = random_address()  # 随机 0x0000-0xFFFF
    _arguments['count'] = random_integer(1000, 0x07B0)  # 随机 0x0001-0x07B0
    _arguments['values'] = []  # 随机
    for i in range(0, _arguments['count']):
        _arguments['values'].append(random_integer(0, 1))

    request = WriteMultipleCoilsRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    response = WriteMultipleCoilsResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x10 write multiple registers
def write_multiple_registers(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes + 1 byte(字节数 N='count') + 2*N bytes
    response 2 bytes + 2 bytes
    '''
    _arguments['address'] = random_address()  # 随机 0x0000-0xFFFF
    _arguments['count'] = random_integer(100, 120)  # 1-120个 0x0001-0x0078
    _arguments['values'] = []  # 每个寄存器最大为0xFFFF
    for i in range(0, _arguments['count']):
        _arguments['values'].append(random_integer(0, 0xFFFF))

    request = WriteMultipleRegistersRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    response = WriteMultipleRegistersResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x14


# 0x15


# 0x16 mask write register
def mask_write_register(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes + 2 bytes
    response 2 bytes + 2 bytes + 2 bytes
    '''
    _arguments['address'] = random_address()  # 随机 0x0000-0xFFFF

    # and_mask, or_mask 随机 最大0xFFFF
    and_mask = random_address()  # 随机 0x0000-0xFFFF
    or_mask = random_address()  # 随机 0x0000-0xFFFF

    request = MaskWriteRegisterRequest(and_mask=and_mask, or_mask=or_mask, **_arguments)
    request_packet = framer.buildPacket(request)

    response = MaskWriteRegisterResponse(and_mask=and_mask, or_mask=or_mask, **_arguments)
    response_packet = framer.buildPacket(response)
    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x17 read write multiple registers
def read_write_multiple_registers(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes + 2 bytes + 2 bytes(N=写的数量) + 1 byte + 2*N bytes
    response 1 byte + 2*N bytes
    '''
    _arguments['read_address'] = random_address()  # 随机 0x0000-0xFFFF
    _arguments['read_count'] = random_integer(100, 118)  # 随机 1-118 0x0001-0x0076
    _arguments['write_address'] = random_address()  # 随机 0x0000-0xFFFF
    _arguments['write_count'] = random_integer(100, 118)  # 随机 1-118 0x0001-0x0076
    _arguments['write_registers'] = []  # 随机
    for i in range(0, _arguments['write_count']):
        _arguments['write_registers'].append(random_integer(0, 0xFFFF))

    request = ReadWriteMultipleRegistersRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    _arguments['values'] = []  # 随机
    for i in range(0, _arguments['read_count']):
        _arguments['values'].append(random_integer(0, 0xFFFF))

    response = ReadWriteMultipleRegistersResponse(**_arguments)
    response_packet = framer.buildPacket(response)
    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x2B read device information 待补充
def read_device_information(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 1 byte + 1 byte + 1 byte
    response 1 byte + 1 byte + 1 byte + 1 byte + 1 byte + 1 byte + 1 byte + 1 byte + 1 byte
    '''
    request = WriteMultipleRegistersRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    response = WriteMultipleRegistersResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x07 read exception status
def read_exception_status(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 0 byte
    response 1 byte
    '''
    request = ReadExceptionStatusRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    status = random_integer(0, 0xFF)  # status 随机, 每位表示八个异常中的一个
    response = ReadExceptionStatusResponse(status, **_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x0B get common event counter
def get_communications_event_counter(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 0 byte
    response 2 bytes(0x0000 represents ready, 0xFFFF means waiting) + 2 bytes
    '''
    request = GetCommEventCounterRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    _arguments['count'] = random_integer(0, 0xFFFF)  # 随机 0x0000-0xFFFF

    response = GetCommEventCounterResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x0C get common event log
def get_communications_event_log(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 0 byte
    response 1 byte + 2 bytes + 2 bytes + 2 bytes(N=事件数量) + N bytes
    '''
    request = GetCommEventLogRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    tmp = random_integer(0, 1)
    _arguments['status'] = True if tmp == 1 else False  # 0000 or 0xFFFF False
    _arguments['message_count'] = random_integer(100, 0xF9)  # 随机
    _arguments['event_count'] = random_integer(100, 0xF9)  # 随机
    _arguments['events'] = []  # 随机
    for i in range(0, _arguments['event_count']):
        _arguments['events'].append(random_integer(0, 0xFF))

    response = GetCommEventLogResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x11 report slave id
def report_slave_id(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 0 byte
    response k bytes(k与identifier有关) byte + 1 byte
    '''
    request = ReportSlaveIdRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    identifier = hex(random_integer(0, 0xFFFF))  # 随机
    tmp = random_integer(0, 1)
    status = True if tmp == 1 else False  # 0xFF or 0x00 False

    response = ReportSlaveIdResponse(identifier, status, **_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x18 read fifo queue
def read_fifo_queue(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes
    response 2 bytes + 2 bytes(N) + N bytes
    '''
    _arguments['address'] = random_address()  # 随机 0x0000-0xFFFF
    request = ReadFifoQueueRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    tmp_cnt = random_integer(100, 0xFF)
    _arguments['values'] = []  # 随机
    for i in range(0, tmp_cnt):
        _arguments['values'].append(random_integer(0, 0xFFFF))

    response = ReadFifoQueueResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x08(0x0000) return query data
def return_query_data(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 2 bytes + 2 bytes
    '''

    message = random_integer(0, 0xFFFF)  # 随机

    request = ReturnQueryDataRequest(message, **_arguments)
    request_packet = framer.buildPacket(request)

    response = ReturnQueryDataResponse(message, **_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x08(0x0001) restart communication option
def restart_communication_option(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 2 bytes + 2 bytes
    '''
    tmp = random_integer(0, 1)
    toggle = True if tmp == 1 else False  # 0xFF00 or 0x0000 False
    request = RestartCommunicationsOptionRequest(toggle, **_arguments)
    request_packet = framer.buildPacket(request)

    response = RestartCommunicationsOptionResponse(toggle, **_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x08(0x0002) Return diagnostic register
def return_diagnostic_register(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 2 bytes + 2 bytes
    '''
    request = ReturnDiagnosticRegisterRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    response = ReturnDiagnosticRegisterResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x08(0x0003) change ascii input delimiter
def change_ascii_input_delimiter(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 2 bytes + 2 bytes
    '''
    request = ChangeAsciiInputDelimiterRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    response = ChangeAsciiInputDelimiterResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x08(0x0004) force listen only mode
def force_listen_only_mode(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 2 bytes + 2 bytes
    '''
    request = ForceListenOnlyModeRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    response = ForceListenOnlyModeResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x08(0x000a) clear counter
def clear_counter(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 2 bytes + 2 bytes
    '''
    request = ClearCountersRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    response = ClearCountersResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x08(0x000b) return bus message count
def return_bus_message_count(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 2 bytes + 2 bytes
    '''
    request = ReturnBusMessageCountRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    response = ReturnBusMessageCountResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x08(0x000c) return bus communication error count
def return_bus_communication_error_count(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 2 bytes + 2 bytes
    '''
    request = ReturnBusCommunicationErrorCountRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    response = ReturnBusCommunicationErrorCountResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x08(0x000d) return bus exception error count
def return_bus_exception_error_count(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 2 bytes + 2 bytes
    '''
    request = ReturnBusExceptionErrorCountRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    response = ReturnBusExceptionErrorCountResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x08(0x000e) return slave message count
def return_slave_message_count(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 2 bytes + 2 bytes
    '''
    request = ReturnSlaveMessageCountRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    response = ReturnSlaveMessageCountResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x08(0x000f) return slave no response count
def return_slave_no_response_count(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 2 bytes + 2 bytes
    '''
    request = ReturnSlaveNoResponseCountRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    response = ReturnSlaveNoReponseCountResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x08(0x0010) return slave nak count
def return_slave_nak_count(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 2 bytes + 2 bytes
    '''
    request = ReturnSlaveNAKCountRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    response = ReturnSlaveNAKCountResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x08(0x0011) return slave busy count
def return_slave_busy_count(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 2 bytes + 2 bytes
    '''
    request = ReturnSlaveBusyCountRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    response = ReturnSlaveBusyCountResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x08(0x0012) return slave bus character overrun count
def return_slave_bus_character_overrun_count(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 2 bytes + 2 bytes
    '''
    request = ReturnSlaveBusCharacterOverrunCountRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    response = ReturnSlaveBusCharacterOverrunCountResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x08(0x0013) return iop overrun count
def return_iop_overrun_count(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 2 bytes + 2 bytes
    '''
    request = ReturnIopOverrunCountRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    response = ReturnIopOverrunCountResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x08(0x0014) clear overrun count
def clear_overrun_count(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 2 bytes + 2 bytes
    '''
    request = ClearOverrunCountRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    response = ClearOverrunCountResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]


# 0x08(0x0015) get clear modbus plus
def get_clear_modbus_plus(framer, _arguments):
    '''
    :param framer:
    :param _arguments:
    :return: [request_packet, response_packet]
    request 2 bytes + 2 bytes
    response 2 bytes + 2 bytes
    '''
    request = GetClearModbusPlusRequest(**_arguments)
    request_packet = framer.buildPacket(request)

    response = GetClearModbusPlusResponse(**_arguments)
    response_packet = framer.buildPacket(response)

    return [request_packet.encode('hex'), response_packet.encode('hex')]

functions = [
    read_coils,
    read_discrete_inputs,
    read_holding_registers,
    read_input_registers,
    write_single_coil,
    write_single_register,
    write_multiple_coils,
    write_multiple_registers,
    mask_write_register,
    read_write_multiple_registers,
    read_exception_status,
    get_communications_event_counter,
    get_communications_event_log,
    report_slave_id,
    read_fifo_queue,
    return_query_data,
    restart_communication_option,
    return_diagnostic_register,
    change_ascii_input_delimiter,
    force_listen_only_mode,
    clear_counter,
    return_bus_message_count,
    return_bus_communication_error_count,
    return_bus_exception_error_count,
    return_slave_message_count,
    return_slave_no_response_count,
    return_slave_nak_count,
    return_slave_busy_count,
    return_slave_bus_character_overrun_count,
    return_iop_overrun_count,
    clear_overrun_count,
    get_clear_modbus_plus
]

choosed_function = [
        write_multiple_coils,
        write_multiple_registers,
        write_single_coil,
        write_single_register,
        read_coils,
        read_holding_registers,
        read_input_registers,
        read_write_multiple_registers
    ]

def generator(max_num):
    framer = ModbusSocketFramer(None)

    for index in range(0,len(choosed_function)):
        path = 'Modbus_data\Origin_generated_data\Modbus_' + choosed_function[index].func_name + '_' \
               + str(files_number('Modbus_data\Origin_generated_data', 'Modbus_' + choosed_function[index].func_name)+1) + '.txt'
        f_txt = open(path, 'w')

        for i in range(0, max_num):
            _arguments = dict()
            _arguments['protocol'] = 0x0000
            _arguments['transaction'] = random_integer(1, pow(2,16-1))
            _arguments['unit'] = random_integer(1, 247)

            result = choosed_function[index](framer, _arguments)

            f_txt.write(str(result[0]) + '\n')  # Just save request packets
        f_txt.close()


def iterate_generator(max_num):
    f_txt = open('Modbus_data\Origin_generated_data\modbus.txt', 'w')
    #    field_names = ['request', 'response']
    #    writer = csv.DictWriter(f, fieldnames=field_names)
    #    writer.writeheader()

    framer = ModbusSocketFramer(None)
    for i in range(0, max_num):
        index = random_integer(0, len(functions) - 1)
        _arguments = dict()
        _arguments['protocol'] = 0x0000
        _arguments['transaction'] = random_integer(1, pow(2,16-1))
        _arguments['unit'] = random_integer(1, 247)

        result = functions[index](framer, _arguments)
        #        writer.writerow({'request':result[0], 'response': result[1]})
        f_txt.write(str(result[0]) + '\n')
    f_txt.close()
if __name__ == '__main__':
    generator(1000)
    iterate_generator(100000)

