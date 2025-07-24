package com.google.ar.sceneform.lullmodel;

import c.b.a.a.a;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class DataQuat extends Table {
    public static void addValue(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addStruct(0, i, 0);
    }

    public static int endDataQuat(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static DataQuat getRootAsDataQuat(ByteBuffer byteBuffer) {
        return getRootAsDataQuat(byteBuffer, new DataQuat());
    }

    public static void startDataQuat(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(1);
    }

    public DataQuat __assign(int i, ByteBuffer byteBuffer) {
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

    public Quat value() {
        return value(new Quat());
    }

    public static DataQuat getRootAsDataQuat(ByteBuffer byteBuffer, DataQuat dataQuat) {
        return dataQuat.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public Quat value(Quat quat) {
        int __offset = __offset(4);
        if (__offset != 0) {
            return quat.__assign(__offset + this.bb_pos, this.bb);
        }
        return null;
    }
}