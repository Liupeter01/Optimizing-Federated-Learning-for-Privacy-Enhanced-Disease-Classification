# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: ml_vector.proto
# Protobuf Python Version: 5.29.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    0,
    '',
    'ml_vector.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0fml_vector.proto\x12\x08mlvector\"I\n\rVectorRequest\x12\x11\n\tclient_id\x18\x01 \x01(\t\x12\x0e\n\x06vector\x18\x02 \x03(\x02\x12\x15\n\rmodel_version\x18\x03 \x01(\t\"S\n\x0eVectorResponse\x12\x15\n\rmerged_vector\x18\x01 \x03(\x02\x12\x13\n\x0bstatus_code\x18\x02 \x01(\x05\x12\x15\n\rerror_message\x18\x03 \x01(\t2X\n\tMLService\x12K\n\x12\x46\x65\x64\x65ratedAveraging\x12\x17.mlvector.VectorRequest\x1a\x18.mlvector.VectorResponse(\x01\x30\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'ml_vector_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_VECTORREQUEST']._serialized_start=29
  _globals['_VECTORREQUEST']._serialized_end=102
  _globals['_VECTORRESPONSE']._serialized_start=104
  _globals['_VECTORRESPONSE']._serialized_end=187
  _globals['_MLSERVICE']._serialized_start=189
  _globals['_MLSERVICE']._serialized_end=277
# @@protoc_insertion_point(module_scope)
