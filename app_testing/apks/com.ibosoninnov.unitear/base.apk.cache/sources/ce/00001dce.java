package com.google.ar.schemas.motive;

import c.b.a.a.a;
import com.google.common.primitives.UnsignedBytes;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class RigAnimFb extends Table {
    public static boolean RigAnimFbBufferHasIdentifier(ByteBuffer byteBuffer) {
        return Table.__has_identifier(byteBuffer, "ANIM");
    }

    public static void addBoneNames(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(2, i, 0);
    }

    public static void addBoneParents(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(1, i, 0);
    }

    public static void addMatrixAnims(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static void addName(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(4, i, 0);
    }

    public static void addRepeat(FlatBufferBuilder flatBufferBuilder, boolean z) {
        flatBufferBuilder.addBoolean(3, z, false);
    }

    public static int createBoneNamesVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int createBoneParentsVector(FlatBufferBuilder flatBufferBuilder, byte[] bArr) {
        return flatBufferBuilder.createByteVector(bArr);
    }

    public static int createMatrixAnimsVector(FlatBufferBuilder flatBufferBuilder, int[] iArr) {
        flatBufferBuilder.startVector(4, iArr.length, 4);
        for (int length = iArr.length - 1; length >= 0; length--) {
            flatBufferBuilder.addOffset(iArr[length]);
        }
        return flatBufferBuilder.endVector();
    }

    public static int createRigAnimFb(FlatBufferBuilder flatBufferBuilder, int i, int i2, int i3, boolean z, int i4) {
        flatBufferBuilder.startObject(5);
        addName(flatBufferBuilder, i4);
        addBoneNames(flatBufferBuilder, i3);
        addBoneParents(flatBufferBuilder, i2);
        addMatrixAnims(flatBufferBuilder, i);
        addRepeat(flatBufferBuilder, z);
        return endRigAnimFb(flatBufferBuilder);
    }

    public static int endRigAnimFb(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static void finishRigAnimFbBuffer(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.finish(i, "ANIM");
    }

    public static void finishSizePrefixedRigAnimFbBuffer(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.finishSizePrefixed(i, "ANIM");
    }

    public static RigAnimFb getRootAsRigAnimFb(ByteBuffer byteBuffer) {
        return getRootAsRigAnimFb(byteBuffer, new RigAnimFb());
    }

    public static void startBoneNamesVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public static void startBoneParentsVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(1, i, 1);
    }

    public static void startMatrixAnimsVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public static void startRigAnimFb(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(5);
    }

    public RigAnimFb __assign(int i, ByteBuffer byteBuffer) {
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

    public String boneNames(int i) {
        int __offset = __offset(8);
        if (__offset != 0) {
            return __string((i * 4) + __vector(__offset));
        }
        return null;
    }

    public int boneNamesLength() {
        int __offset = __offset(8);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public int boneParents(int i) {
        int __offset = __offset(6);
        if (__offset != 0) {
            return this.bb.get((i * 1) + __vector(__offset)) & UnsignedBytes.MAX_VALUE;
        }
        return 0;
    }

    public ByteBuffer boneParentsAsByteBuffer() {
        return __vector_as_bytebuffer(6, 1);
    }

    public ByteBuffer boneParentsInByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 6, 1);
    }

    public int boneParentsLength() {
        int __offset = __offset(6);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public MatrixAnimFb matrixAnims(int i) {
        return matrixAnims(new MatrixAnimFb(), i);
    }

    public int matrixAnimsLength() {
        int __offset = __offset(4);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public String name() {
        int __offset = __offset(12);
        if (__offset != 0) {
            return __string(__offset + this.bb_pos);
        }
        return null;
    }

    public ByteBuffer nameAsByteBuffer() {
        return __vector_as_bytebuffer(12, 1);
    }

    public ByteBuffer nameInByteBuffer(ByteBuffer byteBuffer) {
        return __vector_in_bytebuffer(byteBuffer, 12, 1);
    }

    public boolean repeat() {
        int __offset = __offset(10);
        return (__offset == 0 || this.bb.get(__offset + this.bb_pos) == 0) ? false : true;
    }

    public static int createBoneParentsVector(FlatBufferBuilder flatBufferBuilder, ByteBuffer byteBuffer) {
        return flatBufferBuilder.createByteVector(byteBuffer);
    }

    public static RigAnimFb getRootAsRigAnimFb(ByteBuffer byteBuffer, RigAnimFb rigAnimFb) {
        return rigAnimFb.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public MatrixAnimFb matrixAnims(MatrixAnimFb matrixAnimFb, int i) {
        int __offset = __offset(4);
        if (__offset != 0) {
            return matrixAnimFb.__assign(__indirect((i * 4) + __vector(__offset)), this.bb);
        }
        return null;
    }
}