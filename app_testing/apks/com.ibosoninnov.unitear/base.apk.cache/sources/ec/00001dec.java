package com.google.ar.schemas.sceneform;

import c.b.a.a.a;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class ParameterInitDef extends Table {
    public static void addInit(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(1, i, 0);
    }

    public static void addInitType(FlatBufferBuilder flatBufferBuilder, byte b2) {
        flatBufferBuilder.addByte(0, b2, 0);
    }

    public static int createParameterInitDef(FlatBufferBuilder flatBufferBuilder, byte b2, int i) {
        flatBufferBuilder.startObject(2);
        addInit(flatBufferBuilder, i);
        addInitType(flatBufferBuilder, b2);
        return endParameterInitDef(flatBufferBuilder);
    }

    public static int endParameterInitDef(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static ParameterInitDef getRootAsParameterInitDef(ByteBuffer byteBuffer) {
        return getRootAsParameterInitDef(byteBuffer, new ParameterInitDef());
    }

    public static void startParameterInitDef(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(2);
    }

    public ParameterInitDef __assign(int i, ByteBuffer byteBuffer) {
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

    public Table init(Table table) {
        int __offset = __offset(6);
        if (__offset != 0) {
            return __union(table, __offset);
        }
        return null;
    }

    public byte initType() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return this.bb.get(__offset + this.bb_pos);
        }
        return (byte) 0;
    }

    public static ParameterInitDef getRootAsParameterInitDef(ByteBuffer byteBuffer, ParameterInitDef parameterInitDef) {
        return parameterInitDef.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }
}