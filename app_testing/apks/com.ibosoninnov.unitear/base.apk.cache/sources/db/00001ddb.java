package com.google.ar.schemas.sceneform;

import c.b.a.a.a;
import com.google.android.material.shadow.ShadowDrawableWrapper;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class DoubleInit extends Table {
    public static void addValue(FlatBufferBuilder flatBufferBuilder, double d2) {
        flatBufferBuilder.addDouble(0, d2, ShadowDrawableWrapper.COS_45);
    }

    public static int createDoubleInit(FlatBufferBuilder flatBufferBuilder, double d2) {
        flatBufferBuilder.startObject(1);
        addValue(flatBufferBuilder, d2);
        return endDoubleInit(flatBufferBuilder);
    }

    public static int endDoubleInit(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static DoubleInit getRootAsDoubleInit(ByteBuffer byteBuffer) {
        return getRootAsDoubleInit(byteBuffer, new DoubleInit());
    }

    public static void startDoubleInit(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(1);
    }

    public DoubleInit __assign(int i, ByteBuffer byteBuffer) {
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

    public double value() {
        int __offset = __offset(4);
        return __offset != 0 ? this.bb.getDouble(__offset + this.bb_pos) : ShadowDrawableWrapper.COS_45;
    }

    public static DoubleInit getRootAsDoubleInit(ByteBuffer byteBuffer, DoubleInit doubleInit) {
        return doubleInit.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }
}