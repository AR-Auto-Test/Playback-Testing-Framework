package com.google.ar.schemas.sceneform;

import c.b.a.a.a;
import com.google.common.primitives.UnsignedBytes;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class LightingCubeFaceDef extends Table {
    public static void addData(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static int createDataVector(FlatBufferBuilder flatBufferBuilder, byte[] bArr) {
        return flatBufferBuilder.createByteVector(bArr);
    }

    public static int createLightingCubeFaceDef(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startObject(1);
        addData(flatBufferBuilder, i);
        return endLightingCubeFaceDef(flatBufferBuilder);
    }

    public static int endLightingCubeFaceDef(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static LightingCubeFaceDef getRootAsLightingCubeFaceDef(ByteBuffer byteBuffer) {
        return getRootAsLightingCubeFaceDef(byteBuffer, new LightingCubeFaceDef());
    }

    public static void startDataVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(1, i, 1);
    }

    public static void startLightingCubeFaceDef(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(1);
    }

    public LightingCubeFaceDef __assign(int i, ByteBuffer byteBuffer) {
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

    public int data(int i) {
        int __offset = __offset(4);
        if (__offset != 0) {
            return this.bb.get((i * 1) + __vector(__offset)) & UnsignedBytes.MAX_VALUE;
        }
        return 0;
    }

    public ByteBuffer dataAsByteBuffer() {
        return __vector_as_bytebuffer(4, 1);
    }

    public ByteBuffer dataInByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 4, 1);
    }

    public int dataLength() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public static int createDataVector(FlatBufferBuilder flatBufferBuilder, ByteBuffer byteBuffer) {
        return flatBufferBuilder.createByteVector(byteBuffer);
    }

    public static LightingCubeFaceDef getRootAsLightingCubeFaceDef(ByteBuffer byteBuffer, LightingCubeFaceDef lightingCubeFaceDef) {
        return lightingCubeFaceDef.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }
}