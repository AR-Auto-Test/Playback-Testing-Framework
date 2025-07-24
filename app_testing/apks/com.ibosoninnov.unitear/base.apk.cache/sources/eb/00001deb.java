package com.google.ar.schemas.sceneform;

import c.b.a.a.a;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class ParameterDef extends Table {
    public static void addId(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static void addInitialValue(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(1, i, 0);
    }

    public static int createParameterDef(FlatBufferBuilder flatBufferBuilder, int i, int i2) {
        flatBufferBuilder.startObject(2);
        addInitialValue(flatBufferBuilder, i2);
        addId(flatBufferBuilder, i);
        return endParameterDef(flatBufferBuilder);
    }

    public static int endParameterDef(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static ParameterDef getRootAsParameterDef(ByteBuffer byteBuffer) {
        return getRootAsParameterDef(byteBuffer, new ParameterDef());
    }

    public static void startParameterDef(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(2);
    }

    public ParameterDef __assign(int i, ByteBuffer byteBuffer) {
        __init(i, byteBuffer);
        return this;
    }

    public void __init(int i, ByteBuffer byteBuffer) {
        this.bb_pos = i;
        this.bb = byteBuffer;
        int i2 = i - byteBuffer.getInt(i);
        this.vtable_start = i2;
        this.vtable_size = this.bb.getShort(i2);
    }

    public String id() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return __string(__offset + this.bb_pos);
        }
        return null;
    }

    public ByteBuffer idAsByteBuffer() {
        return __vector_as_bytebuffer(4, 1);
    }

    public ByteBuffer idInByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 4, 1);
    }

    public ParameterInitDef initialValue() {
        return initialValue(new ParameterInitDef());
    }

    public static ParameterDef getRootAsParameterDef(ByteBuffer byteBuffer, ParameterDef parameterDef) {
        return parameterDef.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public ParameterInitDef initialValue(ParameterInitDef parameterInitDef) {
        int __offset = __offset(6);
        if (__offset != 0) {
            return parameterInitDef.__assign(__indirect(__offset + this.bb_pos), this.bb);
        }
        return null;
    }
}