package com.google.ar.sceneform.lullmodel;

import c.b.a.a.a;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class DataVec2 extends Table {
    public static void addValue(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addStruct(0, i, 0);
    }

    public static int endDataVec2(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static DataVec2 getRootAsDataVec2(ByteBuffer byteBuffer) {
        return getRootAsDataVec2(byteBuffer, new DataVec2());
    }

    public static void startDataVec2(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(1);
    }

    public DataVec2 __assign(int i, ByteBuffer byteBuffer) {
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

    public Vec2 value() {
        return value(new Vec2());
    }

    public static DataVec2 getRootAsDataVec2(ByteBuffer byteBuffer, DataVec2 dataVec2) {
        return dataVec2.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public Vec2 value(Vec2 vec2) {
        int __offset = __offset(4);
        if (__offset != 0) {
            return vec2.__assign(__offset + this.bb_pos, this.bb);
        }
        return null;
    }
}