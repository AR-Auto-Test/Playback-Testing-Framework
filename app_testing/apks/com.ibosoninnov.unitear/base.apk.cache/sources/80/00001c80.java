package com.google.ar.sceneform.lullmodel;

import c.b.a.a.a;
import com.google.common.primitives.UnsignedBytes;
import com.google.common.primitives.UnsignedInts;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class BlendShape extends Table {
    public static void addName(FlatBufferBuilder flatBufferBuilder, long j) {
        flatBufferBuilder.addInt(0, (int) j, 0);
    }

    public static void addTangentData(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(4, i, 0);
    }

    public static void addTangentIndices16(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(6, i, 0);
    }

    public static void addTangentIndices32(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(5, i, 0);
    }

    public static void addVertexData(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(1, i, 0);
    }

    public static void addVertexIndices16(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(3, i, 0);
    }

    public static void addVertexIndices32(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(2, i, 0);
    }

    public static int createBlendShape(FlatBufferBuilder flatBufferBuilder, long j, int i, int i2, int i3, int i4, int i5, int i6) {
        flatBufferBuilder.startObject(7);
        addTangentIndices16(flatBufferBuilder, i6);
        addTangentIndices32(flatBufferBuilder, i5);
        addTangentData(flatBufferBuilder, i4);
        addVertexIndices16(flatBufferBuilder, i3);
        addVertexIndices32(flatBufferBuilder, i2);
        addVertexData(flatBufferBuilder, i);
        addName(flatBufferBuilder, j);
        return endBlendShape(flatBufferBuilder);
    }

    public static int createTangentDataVector(FlatBufferBuilder flatBufferBuilder, byte[] bArr) {
        return flatBufferBuilder.createByteVector(bArr);
    }

    public static int createTangentIndices16Vector(FlatBufferBuilder flatBufferBuilder, short[] sArr) {
        flatBufferBuilder.startVector(2, sArr.length, 2);
        for (int length = sArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addShort(sArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int createTangentIndices32Vector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addInt(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int createVertexDataVector(FlatBufferBuilder flatBufferBuilder, byte[] bArr) {
        return flatBufferBuilder.createByteVector(bArr);
    }

    public static int createVertexIndices16Vector(FlatBufferBuilder flatBufferBuilder, short[] sArr) {
        flatBufferBuilder.startVector(2, sArr.length, 2);
        for (int length = sArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addShort(sArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int createVertexIndices32Vector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addInt(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int endBlendShape(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static BlendShape getRootAsBlendShape(ByteBuffer byteBuffer) {
        return getRootAsBlendShape(byteBuffer, new BlendShape());
    }

    public static void startBlendShape(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(7);
    }

    public static void startTangentDataVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(1, i, 1);
    }

    public static void startTangentIndices16Vector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(2, i, 2);
    }

    public static void startTangentIndices32Vector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public static void startVertexDataVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(1, i, 1);
    }

    public static void startVertexIndices16Vector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(2, i, 2);
    }

    public static void startVertexIndices32Vector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public BlendShape __assign(int i, ByteBuffer byteBuffer) {
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

    public long name() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return this.bb.getInt(__offset + this.bb_pos) & UnsignedInts.INT_MASK;
        }
        return 0L;
    }

    public int tangentData(int i) {
        int __offset = __offset(12);
        if (__offset != 0) {
            return this.bb.get((i * 1) + __vector(__offset)) & UnsignedBytes.MAX_VALUE;
        }
        return 0;
    }

    public ByteBuffer tangentDataAsByteBuffer() {
        return __vector_as_bytebuffer(12, 1);
    }

    public ByteBuffer tangentDataInByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 12, 1);
    }

    public int tangentDataLength() {
        int __offset = __offset(12);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public int tangentIndices16(int i) {
        int __offset = __offset(16);
        if (__offset != 0) {
            return this.bb.getShort((i * 2) + __vector(__offset)) & 65535;
        }
        return 0;
    }

    public ByteBuffer tangentIndices16AsByteBuffer() {
        return __vector_as_bytebuffer(16, 2);
    }

    public ByteBuffer tangentIndices16InByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 16, 2);
    }

    public int tangentIndices16Length() {
        int __offset = __offset(16);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public long tangentIndices32(int i) {
        int __offset = __offset(14);
        if (__offset != 0) {
            return this.bb.getInt((i * 4) + __vector(__offset)) & UnsignedInts.INT_MASK;
        }
        return 0L;
    }

    public ByteBuffer tangentIndices32AsByteBuffer() {
        return __vector_as_bytebuffer(14, 4);
    }

    public ByteBuffer tangentIndices32InByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 14, 4);
    }

    public int tangentIndices32Length() {
        int __offset = __offset(14);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public int vertexData(int i) {
        int __offset = __offset(6);
        if (__offset != 0) {
            return this.bb.get((i * 1) + __vector(__offset)) & UnsignedBytes.MAX_VALUE;
        }
        return 0;
    }

    public ByteBuffer vertexDataAsByteBuffer() {
        return __vector_as_bytebuffer(6, 1);
    }

    public ByteBuffer vertexDataInByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 6, 1);
    }

    public int vertexDataLength() {
        int __offset = __offset(6);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public int vertexIndices16(int i) {
        int __offset = __offset(10);
        if (__offset != 0) {
            return this.bb.getShort((i * 2) + __vector(__offset)) & 65535;
        }
        return 0;
    }

    public ByteBuffer vertexIndices16AsByteBuffer() {
        return __vector_as_bytebuffer(10, 2);
    }

    public ByteBuffer vertexIndices16InByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 10, 2);
    }

    public int vertexIndices16Length() {
        int __offset = __offset(10);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public long vertexIndices32(int i) {
        int __offset = __offset(8);
        if (__offset != 0) {
            return this.bb.getInt((i * 4) + __vector(__offset)) & UnsignedInts.INT_MASK;
        }
        return 0L;
    }

    public ByteBuffer vertexIndices32AsByteBuffer() {
        return __vector_as_bytebuffer(8, 4);
    }

    public ByteBuffer vertexIndices32InByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 8, 4);
    }

    public int vertexIndices32Length() {
        int __offset = __offset(8);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public static int createTangentDataVector(FlatBufferBuilder flatBufferBuilder, ByteBuffer byteBuffer) {
        return flatBufferBuilder.createByteVector(byteBuffer);
    }

    public static int createVertexDataVector(FlatBufferBuilder flatBufferBuilder, ByteBuffer byteBuffer) {
        return flatBufferBuilder.createByteVector(byteBuffer);
    }

    public static BlendShape getRootAsBlendShape(ByteBuffer byteBuffer, BlendShape blendShape) {
        return blendShape.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }
}