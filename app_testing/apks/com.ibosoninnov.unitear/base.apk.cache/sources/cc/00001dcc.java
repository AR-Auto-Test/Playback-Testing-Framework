package com.google.ar.schemas.motive;

import c.b.a.a.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.shadow.ShadowDrawableWrapper;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class ModularParameters extends Table {
    public static void addMax(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.addFloat(2, f2, ShadowDrawableWrapper.COS_45);
    }

    public static void addMin(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.addFloat(1, f2, ShadowDrawableWrapper.COS_45);
    }

    public static void addModular(FlatBufferBuilder flatBufferBuilder, boolean z) {
        flatBufferBuilder.addBoolean(0, z, false);
    }

    public static int createModularParameters(FlatBufferBuilder flatBufferBuilder, boolean z, float f2, float f3) {
        flatBufferBuilder.startObject(3);
        addMax(flatBufferBuilder, f3);
        addMin(flatBufferBuilder, f2);
        addModular(flatBufferBuilder, z);
        return endModularParameters(flatBufferBuilder);
    }

    public static int endModularParameters(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static ModularParameters getRootAsModularParameters(ByteBuffer byteBuffer) {
        return getRootAsModularParameters(byteBuffer, new ModularParameters());
    }

    public static void startModularParameters(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(3);
    }

    public ModularParameters __assign(int i, ByteBuffer byteBuffer) {
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

    public float max() {
        int __offset = __offset(8);
        return __offset != 0 ? this.bb.getFloat(__offset + this.bb_pos) : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public float min() {
        int __offset = __offset(6);
        return __offset != 0 ? this.bb.getFloat(__offset + this.bb_pos) : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public boolean modular() {
        int __offset = __offset(4);
        return (__offset == 0 || this.bb.get(__offset + this.bb_pos) == 0) ? false : true;
    }

    public static ModularParameters getRootAsModularParameters(ByteBuffer byteBuffer, ModularParameters modularParameters) {
        return modularParameters.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }
}