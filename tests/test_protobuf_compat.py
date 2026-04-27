from edsvfh.droid_convert import DroidPreparedTFDSSource


def test_protobuf_label_shim_exists_after_patch():
    DroidPreparedTFDSSource._ensure_protobuf_descriptor_compat()
    from google.protobuf import descriptor as pb_descriptor

    assert hasattr(pb_descriptor.FieldDescriptor, "label")
    assert hasattr(pb_descriptor.FieldDescriptor, "LABEL_REPEATED")
