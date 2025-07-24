package com.google.ar.schemas.motive;

import c.b.a.a.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.shadow.ShadowDrawableWrapper;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class Settled1fParameters extends Table {
    public static void addMaxDifference(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.addFloat(0, f2, ShadowDrawableWrapper.COS_45);
    }

    public static void addMaxVelocity(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.addFloat(1, f2, ShadowDrawableWrapper.COS_45);
    }

    public static int createSettled1fParameters(FlatBufferBuilder flatBufferBuilder, float f2, float f3) {
        flatBufferBuilder.startObject(2);
        addMaxVelocity(flatBufferBuilder, f3);
        addMaxDifference(flatBufferBuilder, f2);
        return endSettled1fParameters(flatBufferBuilder);
    }

    public static int endSettled1fParameters(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static Settled1fParameters getRootAsSettled1fParameters(ByteBuffer byteBuffer) {
        return getRootAsSettled1fParameters(byteBuffer, new Settled1fParameters());
    }

    public static void startSettled1fParameters(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(2);
    }

    public Settled1fParameters __assign(int i, ByteBuffer byteBuffer) {
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

    public float maxDifference() {
        int __offset = __offset(4);
        return __offset != 0 ? this.bb.getFloat(__offset + this.bb_pos) : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public float maxVelocity() {
        int __offset = __offset(6);
        return __offset != 0 ? this.bb.getFloat(__offset + this.bb_pos) : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public static Settled1fParameters getRootAsSettled1fParameters(ByteBuffer byteBuffer, Settled1fParameters settled1fParameters) {
        return settled1fParameters.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }
}