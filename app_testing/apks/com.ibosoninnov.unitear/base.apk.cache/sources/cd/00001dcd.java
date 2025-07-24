package com.google.ar.schemas.motive;

import c.b.a.a.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.shadow.ShadowDrawableWrapper;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class OvershootParameters extends Table {
    public static void addAccelerationPerDifference(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.addFloat(4, f2, ShadowDrawableWrapper.COS_45);
    }

    public static void addAtTarget(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(3, i, 0);
    }

    public static void addBase(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static void addMaxDelta(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.addFloat(2, f2, ShadowDrawableWrapper.COS_45);
    }

    public static void addMaxDeltaTime(FlatBufferBuilder flatBufferBuilder, short s) {
        flatBufferBuilder.addShort(6, s, 32767);
    }

    public static void addMaxVelocity(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.addFloat(1, f2, ShadowDrawableWrapper.COS_45);
    }

    public static void addWrongDirectionAccelerationMultiplier(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.addFloat(5, f2, ShadowDrawableWrapper.COS_45);
    }

    public static int createOvershootParameters(FlatBufferBuilder flatBufferBuilder, int i, float f2, float f3, int i2, float f4, float f5, short s) {
        flatBufferBuilder.startObject(7);
        addWrongDirectionAccelerationMultiplier(flatBufferBuilder, f5);
        addAccelerationPerDifference(flatBufferBuilder, f4);
        addAtTarget(flatBufferBuilder, i2);
        addMaxDelta(flatBufferBuilder, f3);
        addMaxVelocity(flatBufferBuilder, f2);
        addBase(flatBufferBuilder, i);
        addMaxDeltaTime(flatBufferBuilder, s);
        return endOvershootParameters(flatBufferBuilder);
    }

    public static int endOvershootParameters(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static OvershootParameters getRootAsOvershootParameters(ByteBuffer byteBuffer) {
        return getRootAsOvershootParameters(byteBuffer, new OvershootParameters());
    }

    public static void startOvershootParameters(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(7);
    }

    public OvershootParameters __assign(int i, ByteBuffer byteBuffer) {
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

    public float accelerationPerDifference() {
        int __offset = __offset(12);
        return __offset != 0 ? this.bb.getFloat(__offset + this.bb_pos) : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public Settled1fParameters atTarget() {
        return atTarget(new Settled1fParameters());
    }

    public ModularParameters base() {
        return base(new ModularParameters());
    }

    public float maxDelta() {
        int __offset = __offset(8);
        return __offset != 0 ? this.bb.getFloat(__offset + this.bb_pos) : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public short maxDeltaTime() {
        int __offset = __offset(16);
        if (__offset != 0) {
            return this.bb.getShort(__offset + this.bb_pos);
        }
        return Short.MAX_VALUE;
    }

    public float maxVelocity() {
        int __offset = __offset(6);
        return __offset != 0 ? this.bb.getFloat(__offset + this.bb_pos) : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public float wrongDirectionAccelerationMultiplier() {
        int __offset = __offset(14);
        return __offset != 0 ? this.bb.getFloat(__offset + this.bb_pos) : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public static OvershootParameters getRootAsOvershootParameters(ByteBuffer byteBuffer, OvershootParameters overshootParameters) {
        return overshootParameters.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public Settled1fParameters atTarget(Settled1fParameters settled1fParameters) {
        int __offset = __offset(10);
        if (__offset != 0) {
            return settled1fParameters.__assign(__indirect(__offset + this.bb_pos), this.bb);
        }
        return null;
    }

    public ModularParameters base(ModularParameters modularParameters) {
        int __offset = __offset(4);
        if (__offset != 0) {
            return modularParameters.__assign(__indirect(__offset + this.bb_pos), this.bb);
        }
        return null;
    }
}