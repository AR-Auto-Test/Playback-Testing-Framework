package com.google.ar.schemas.sceneform;

import c.b.a.a.a;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class NullInit extends Table {
    public static int endNullInit(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static NullInit getRootAsNullInit(ByteBuffer byteBuffer) {
        return getRootAsNullInit(byteBuffer, new NullInit());
    }

    public static void startNullInit(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(0);
    }

    public NullInit __assign(int i, ByteBuffer byteBuffer) {
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

    public static NullInit getRootAsNullInit(ByteBuffer byteBuffer, NullInit nullInit) {
        return nullInit.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }
}