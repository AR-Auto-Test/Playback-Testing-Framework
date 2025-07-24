package com.google.ar.schemas.motive;

import c.b.a.a.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.shadow.ShadowDrawableWrapper;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class ConstantOpFb extends Table {
    public static void addYConst(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.addFloat(0, f2, ShadowDrawableWrapper.COS_45);
    }

    public static int createConstantOpFb(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.startObject(1);
        addYConst(flatBufferBuilder, f2);
        return endConstantOpFb(flatBufferBuilder);
    }

    public static int endConstantOpFb(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static ConstantOpFb getRootAsConstantOpFb(ByteBuffer byteBuffer) {
        return getRootAsConstantOpFb(byteBuffer, new ConstantOpFb());
    }

    public static void startConstantOpFb(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(1);
    }

    public ConstantOpFb __assign(int i, ByteBuffer byteBuffer) {
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

    public float yConst() {
        int __offset = __offset(4);
        return __offset != 0 ? this.bb.getFloat(__offset + this.bb_pos) : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public static ConstantOpFb getRootAsConstantOpFb(ByteBuffer byteBuffer, ConstantOpFb constantOpFb) {
        return constantOpFb.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }
}