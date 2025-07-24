package com.google.ar.sceneform.lullmodel;

import c.b.a.a.a;
import com.google.common.primitives.UnsignedInts;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class DataHashValue extends Table {
    public static void addValue(FlatBufferBuilder flatBufferBuilder, long j) {
        flatBufferBuilder.addInt(0, (int) j, 0);
    }

    public static int createDataHashValue(FlatBufferBuilder flatBufferBuilder, long j) {
        flatBufferBuilder.startObject(1);
        addValue(flatBufferBuilder, j);
        return endDataHashValue(flatBufferBuilder);
    }

    public static int endDataHashValue(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static DataHashValue getRootAsDataHashValue(ByteBuffer byteBuffer) {
        return getRootAsDataHashValue(byteBuffer, new DataHashValue());
    }

    public static void startDataHashValue(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(1);
    }

    public DataHashValue __assign(int i, ByteBuffer byteBuffer) {
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

    public long value() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return this.bb.getInt(__offset + this.bb_pos) & UnsignedInts.INT_MASK;
        }
        return 0L;
    }

    public static DataHashValue getRootAsDataHashValue(ByteBuffer byteBuffer, DataHashValue dataHashValue) {
        return dataHashValue.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }
}