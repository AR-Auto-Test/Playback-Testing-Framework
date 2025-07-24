package com.google.ar.sceneform.lullmodel;

import c.b.a.a.a;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class DataVec3 extends Table {
    public static void addValue(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addStruct(0, i, 0);
    }

    public static int endDataVec3(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static DataVec3 getRootAsDataVec3(ByteBuffer byteBuffer) {
        return getRootAsDataVec3(byteBuffer, new DataVec3());
    }

    public static void startDataVec3(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(1);
    }

    public DataVec3 __assign(int i, ByteBuffer byteBuffer) {
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

    public Vec3 value() {
        return value(new Vec3());
    }

    public static DataVec3 getRootAsDataVec3(ByteBuffer byteBuffer, DataVec3 dataVec3) {
        return dataVec3.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public Vec3 value(Vec3 vec3) {
        int __offset = __offset(4);
        if (__offset != 0) {
            return vec3.__assign(__offset + this.bb_pos, this.bb);
        }
        return null;
    }
}