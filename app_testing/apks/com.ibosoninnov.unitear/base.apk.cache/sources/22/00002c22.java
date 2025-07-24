package com.google.protobuf;

/* loaded from: classes2.dex */
public abstract class UnknownFieldSchema<T, B> {
    public abstract void addFixed32(B b2, int i, int i2);

    public abstract void addFixed64(B b2, int i, long j);

    public abstract void addGroup(B b2, int i, T t);

    public abstract void addLengthDelimited(B b2, int i, ByteString byteString);

    public abstract void addVarint(B b2, int i, long j);

    public abstract B getBuilderFromMessage(Object obj);

    public abstract T getFromMessage(Object obj);

    public abstract int getSerializedSize(T t);

    public abstract int getSerializedSizeAsMessageSet(T t);

    public abstract void makeImmutable(Object obj);

    public abstract T merge(T t, T t2);

    public final void mergeFrom(B b2, Reader reader) {
        while (reader.getFieldNumber() != Integer.MAX_VALUE && mergeOneFieldFrom(b2, reader)) {
        }
    }

    public final boolean mergeOneFieldFrom(B b2, Reader reader) {
        int tag = reader.getTag();
        int tagFieldNumber = WireFormat.getTagFieldNumber(tag);
        int tagWireType = WireFormat.getTagWireType(tag);
        if (tagWireType == 0) {
            addVarint(b2, tagFieldNumber, reader.readInt64());
            return true;
        } else if (tagWireType == 1) {
            addFixed64(b2, tagFieldNumber, reader.readFixed64());
            return true;
        } else if (tagWireType == 2) {
            addLengthDelimited(b2, tagFieldNumber, reader.readBytes());
            return true;
        } else if (tagWireType != 3) {
            if (tagWireType != 4) {
                if (tagWireType == 5) {
                    addFixed32(b2, tagFieldNumber, reader.readFixed32());
                    return true;
                }
                throw InvalidProtocolBufferException.invalidWireType();
            }
            return false;
        } else {
            B newBuilder = newBuilder();
            int makeTag = WireFormat.makeTag(tagFieldNumber, 4);
            mergeFrom(newBuilder, reader);
            if (makeTag == reader.getTag()) {
                addGroup(b2, tagFieldNumber, toImmutable(newBuilder));
                return true;
            }
            throw InvalidProtocolBufferException.invalidEndTag();
        }
    }

    public abstract B newBuilder();

    public abstract void setBuilderToMessage(Object obj, B b2);

    public abstract void setToMessage(Object obj, T t);

    public abstract boolean shouldDiscardUnknownFields(Reader reader);

    public abstract T toImmutable(B b2);

    public abstract void writeAsMessageSetTo(T t, Writer writer);

    public abstract void writeTo(T t, Writer writer);
}