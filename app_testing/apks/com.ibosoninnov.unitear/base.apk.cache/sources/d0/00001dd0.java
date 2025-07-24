package com.google.ar.schemas.motive;

import c.b.a.a.a;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class SplineParameters extends Table {
    public static void addBase(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static int createSplineParameters(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startObject(1);
        addBase(flatBufferBuilder, i);
        return endSplineParameters(flatBufferBuilder);
    }

    public static int endSplineParameters(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static SplineParameters getRootAsSplineParameters(ByteBuffer byteBuffer) {
        return getRootAsSplineParameters(byteBuffer, new SplineParameters());
    }

    public static void startSplineParameters(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(1);
    }

    public SplineParameters __assign(int i, ByteBuffer byteBuffer) {
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

    public ModularParameters base() {
        return base(new ModularParameters());
    }

    public static SplineParameters getRootAsSplineParameters(ByteBuffer byteBuffer, SplineParameters splineParameters) {
        return splineParameters.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public ModularParameters base(ModularParameters modularParameters) {
        int __offset = __offset(4);
        if (__offset != 0) {
            return modularParameters.__assign(__indirect(__offset + this.bb_pos), this.bb);
        }
        return null;
    }
}