"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""


from io import BytesIO
import struct

class twist_t(object):

    __slots__ = ["timestamp", "x_vel", "y_vel", "omega_vel"]

    __typenames__ = ["int64_t", "double", "double", "double"]

    __dimensions__ = [None, [2], [2], [2]]

    def __init__(self):
        self.timestamp = 0
        """ LCM Type: int64_t """
        self.x_vel = [ 0.0 for dim0 in range(2) ]
        """ LCM Type: double[2] """
        self.y_vel = [ 0.0 for dim0 in range(2) ]
        """ LCM Type: double[2] """
        self.omega_vel = [ 0.0 for dim0 in range(2) ]
        """ LCM Type: double[2] """

    def encode(self):
        buf = BytesIO()
        buf.write(twist_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack(">q", self.timestamp))
        buf.write(struct.pack('>2d', *self.x_vel[:2]))
        buf.write(struct.pack('>2d', *self.y_vel[:2]))
        buf.write(struct.pack('>2d', *self.omega_vel[:2]))

    @staticmethod
    def decode(data: bytes):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != twist_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return twist_t._decode_one(buf)

    @staticmethod
    def _decode_one(buf):
        self = twist_t()
        self.timestamp = struct.unpack(">q", buf.read(8))[0]
        self.x_vel = struct.unpack('>2d', buf.read(16))
        self.y_vel = struct.unpack('>2d', buf.read(16))
        self.omega_vel = struct.unpack('>2d', buf.read(16))
        return self

    @staticmethod
    def _get_hash_recursive(parents):
        if twist_t in parents: return 0
        tmphash = (0xc9afc5e8385b4a7) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff) + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _packed_fingerprint = None

    @staticmethod
    def _get_packed_fingerprint():
        if twist_t._packed_fingerprint is None:
            twist_t._packed_fingerprint = struct.pack(">Q", twist_t._get_hash_recursive([]))
        return twist_t._packed_fingerprint

    def get_hash(self):
        """Get the LCM hash of the struct"""
        return struct.unpack(">Q", twist_t._get_packed_fingerprint())[0]

