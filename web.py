# import soaplib
# from soaplib.core.service import rpc, DefinitionBase
# from soaplib.core.model.primitive import String, Integer, Boolean
# from soaplib.core.server import wsgi
# from soaplib.core.model.clazz import Array
# from soaplib.core.service import soap
# from soaplib.core.model.clazz import ClassModel
#
#
# class Rules(ClassModel):
#     __namespace__ = "Rules"
#     username = String
#     emotion = String
#
#
# class HelloWorldService(DefinitionBase):
#     @soap(String, Integer, _returns=Array(String))
#     def say_hello(self, name, times):
#         results = []
#         for i in range(0, times):
#             results.append('Hello, %s' % name)
#         return results
#
#     @soap(Rules, _returns=Boolean)
#     def get_recommend(self, rules):
#         print rules.username
#         print 111
#         print rules.emotion
#
#         return 1
#
#
# if __name__ == '__main__':
#     try:
#         from wsgiref.simple_server import make_server
#
#         soap_application = soaplib.core.Application([HelloWorldService], 'tns')
#         wsgi_application = wsgi.Application(soap_application)
#         server = make_server('localhost', 7789, wsgi_application)
#         server.serve_forever()
#     except ImportError:
#         print "Error: example server code requires Python >= 2.5"
# test_rpc.py
# coding=utf-8
from SimpleXMLRPCServer import SimpleXMLRPCServer
from SocketServer import ThreadingMixIn
from xmlrpclib import ServerProxy
import thread
import json


class ThreadXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    pass


class RPCServer:
    def __init__(self, ip='0.0.0.0', port='8000'):
        self.ip = ip
        self.port = int(port)
        self.svr = None

    def start(self, func_lst):
        thread.start_new_thread(self.service, (func_lst, 0,))

    def resume_service(self):
        self.svr.serve_forever(poll_interval=0.001)

    def service(self, func_lst):
        self.svr = ThreadXMLRPCServer((self.ip, self.port), allow_none=True)
        for func in func_lst:
            self.svr.register_function(func)
        self.svr.serve_forever(poll_interval=0.001)

    def activate(self):
        thread.start_new_thread(self.resume_service, (0, 0,))

    def shutdown(self):
        try:
            self.svr.shutdown()
        except Exception, e:
            print 'rpc_server shutdown:', str(e)


def get_ti():
    with open('ti.json', 'r') as f:
        ti = json.load(f)
    print 'get time interval'
    json_ti = json.dumps(ti)
    return json_ti


def get_sp():
    with open('sp.json', 'r') as f:
        sp = json.load(f)
    print 'get set point'
    json_sp = json.dumps(sp)
    return json_sp


def get_crc():
    with open('crc.json', 'r') as f:
        crc = json.load(f)
    print 'get crc rate'
    json_crc = json.dumps(crc)
    return json_crc


def get_pm():
    with open('pm.json', 'r') as f:
        pm = json.load(f)
    print 'get pressure measurement'
    json_pm = json.dumps(pm)
    return json_pm


def get_pid():
    with open('pid.json', 'r') as f:
        pid = json.load(f)
    print 'get pid'
    json_pid = json.dumps(pid)
    return json_pid


if __name__ == "__main__":
    r = RPCServer('0.0.0.0', '8061')
    r.service([get_ti, get_crc, get_sp, get_pm, get_pid])
