package com.google.ar.schemas.sceneform;

import c.b.a.a.a;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.android.material.shadow.ShadowDrawableWrapper;
import com.google.ar.schemas.lull.Vec3;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class LightingDef extends Table {
    public static void addCubeLevels(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(2, i, 0);
    }

    public static void addName(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static void addScale(FlatBufferBuilder flatBufferBuilder, float f2) {
        flatBufferBuilder.addFloat(1, f2, ShadowDrawableWrapper.COS_45);
    }

    public static void addShCoefficients(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(3, i, 0);
    }

    public static int createCubeLevelsVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int createLightingDef(FlatBufferBuilder flatBufferBuilder, int i, float f2, int i2, int i3) {
        flatBufferBuilder.startObject(4);
        addShCoefficients(flatBufferBuilder, i3);
        addCubeLevels(flatBufferBuilder, i2);
        addScale(flatBufferBuilder, f2);
        addName(flatBufferBuilder, i);
        return endLightingDef(flatBufferBuilder);
    }

    public static int endLightingDef(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static LightingDef getRootAsLightingDef(ByteBuffer byteBuffer) {
        return getRootAsLightingDef(byteBuffer, new LightingDef());
    }

    public static void startCubeLevelsVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public static void startLightingDef(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(4);
    }

    public static void startShCoefficientsVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(12, i, 4);
    }

    public LightingDef __assign(int i, ByteBuffer byteBuffer) {
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

    public LightingCubeDef cubeLevels(int i) {
        return cubeLevels(new LightingCubeDef(), i);
    }

    public int cubeLevelsLength() {
        int __offset = __offset(8);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public String name() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return __string(__offset + this.bb_pos);
        }
        return null;
    }

    public ByteBuffer nameAsByteBuffer() {
        return __vector_as_bytebuffer(4, 1);
    }

    public ByteBuffer nameInByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 4, 1);
    }

    public float scale() {
        int __offset = __offset(6);
        return __offset != 0 ? this.bb.getFloat(__offset + this.bb_pos) : StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
    }

    public Vec3 shCoefficients(int i) {
        return shCoefficients(new Vec3(), i);
    }

    public int shCoefficientsLength() {
        int __offset = __offset(10);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public static LightingDef getRootAsLightingDef(ByteBuffer byteBuffer, LightingDef lightingDef) {
        return lightingDef.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public LightingCubeDef cubeLevels(LightingCubeDef lightingCubeDef, int i) {
        int __offset = __offset(8);
        if (__offset != 0) {
            return lightingCubeDef.__assign(__indirect((i * 4) + __vector(__offset)), this.bb);
        }
        return null;
    }

    public Vec3 shCoefficients(Vec3 vec3, int i) {
        int __offset = __offset(10);
        if (__offset != 0) {
            return vec3.__assign((i * 12) + __vector(__offset), this.bb);
        }
        return null;
    }
}