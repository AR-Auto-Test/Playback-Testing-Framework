package com.google.ar.schemas.sceneform;

import c.b.a.a.a;
import com.google.common.primitives.UnsignedBytes;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class SamplerParamsDef extends Table {
    public static void addAnisotropyLog2(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addByte(6, (byte) i, 0);
    }

    public static void addCompareFunc(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addShort(8, (short) i, 0);
    }

    public static void addCompareMode(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addShort(7, (short) i, 0);
    }

    public static void addMagFilter(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addShort(1, (short) i, 0);
    }

    public static void addMinFilter(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addShort(2, (short) i, 0);
    }

    public static void addUsageType(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addShort(0, (short) i, 0);
    }

    public static void addWrapR(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addShort(5, (short) i, 0);
    }

    public static void addWrapS(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addShort(3, (short) i, 0);
    }

    public static void addWrapT(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addShort(4, (short) i, 0);
    }

    public static int createSamplerParamsDef(FlatBufferBuilder flatBufferBuilder, int i, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9) {
        flatBufferBuilder.startObject(9);
        addCompareFunc(flatBufferBuilder, i9);
        addCompareMode(flatBufferBuilder, i8);
        addWrapR(flatBufferBuilder, i6);
        addWrapT(flatBufferBuilder, i5);
        addWrapS(flatBufferBuilder, i4);
        addMinFilter(flatBufferBuilder, i3);
        addMagFilter(flatBufferBuilder, i2);
        addUsageType(flatBufferBuilder, i);
        addAnisotropyLog2(flatBufferBuilder, i7);
        return endSamplerParamsDef(flatBufferBuilder);
    }

    public static int endSamplerParamsDef(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static SamplerParamsDef getRootAsSamplerParamsDef(ByteBuffer byteBuffer) {
        return getRootAsSamplerParamsDef(byteBuffer, new SamplerParamsDef());
    }

    public static void startSamplerParamsDef(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(9);
    }

    public SamplerParamsDef __assign(int i, ByteBuffer byteBuffer) {
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

    public int anisotropyLog2() {
        int __offset = __offset(16);
        if (__offset != 0) {
            return this.bb.get(__offset + this.bb_pos) & UnsignedBytes.MAX_VALUE;
        }
        return 0;
    }

    public int compareFunc() {
        int __offset = __offset(20);
        if (__offset != 0) {
            return this.bb.getShort(__offset + this.bb_pos) & 65535;
        }
        return 0;
    }

    public int compareMode() {
        int __offset = __offset(18);
        if (__offset != 0) {
            return this.bb.getShort(__offset + this.bb_pos) & 65535;
        }
        return 0;
    }

    public int magFilter() {
        int __offset = __offset(6);
        if (__offset != 0) {
            return this.bb.getShort(__offset + this.bb_pos) & 65535;
        }
        return 0;
    }

    public int minFilter() {
        int __offset = __offset(8);
        if (__offset != 0) {
            return this.bb.getShort(__offset + this.bb_pos) & 65535;
        }
        return 0;
    }

    public int usageType() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return this.bb.getShort(__offset + this.bb_pos) & 65535;
        }
        return 0;
    }

    public int wrapR() {
        int __offset = __offset(14);
        if (__offset != 0) {
            return this.bb.getShort(__offset + this.bb_pos) & 65535;
        }
        return 0;
    }

    public int wrapS() {
        int __offset = __offset(10);
        if (__offset != 0) {
            return this.bb.getShort(__offset + this.bb_pos) & 65535;
        }
        return 0;
    }

    public int wrapT() {
        int __offset = __offset(12);
        if (__offset != 0) {
            return this.bb.getShort(__offset + this.bb_pos) & 65535;
        }
        return 0;
    }

    public static SamplerParamsDef getRootAsSamplerParamsDef(ByteBuffer byteBuffer, SamplerParamsDef samplerParamsDef) {
        return samplerParamsDef.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }
}