package com.google.ar.schemas.sceneform;

import c.b.a.a.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.shadow.ShadowDrawableWrapper;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class ScalarInit extends Table {
    public static void addValue(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.addFloat(0, f2, ShadowDrawableWrapper.COS_45);
    }

    public static int createScalarInit(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.startObject(1);
        addValue(flatBufferBuilder, f2);
        return endScalarInit(flatBufferBuilder);
    }

    public static int endScalarInit(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static ScalarInit getRootAsScalarInit(ByteBuffer byteBuffer) {
        return getRootAsScalarInit(byteBuffer, new ScalarInit());
    }

    public static void startScalarInit(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(1);
    }

    public ScalarInit __assign(int i, ByteBuffer byteBuffer) {
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

    public float value() {
        int __offset = __offset(4);
        return __offset != 0 ? this.bb.getFloat(__offset + this.bb_pos) : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public static ScalarInit getRootAsScalarInit(ByteBuffer byteBuffer, ScalarInit scalarInit) {
        return scalarInit.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }
}