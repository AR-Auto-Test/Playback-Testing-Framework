package com.google.ar.sceneform.lullmodel;

import c.b.a.a.a;
import com.google.common.primitives.UnsignedBytes;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/* loaded from: classes.dex */
public final class SkeletonDef extends Table {
    public static void addBoneNames(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(0, i, 0);
    }

    public static void addBoneParents(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(1, i, 0);
    }

    public static void addBoneTransforms(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.addOffset(2, i, 0);
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

    public static int createSkeletonDef(FlatBufferBuilder flatBufferBuilder, int i, int i2, int i3) {
        flatBufferBuilder.startObject(5);
        addBoneTransforms(flatBufferBuilder, i3);
        addBoneParents(flatBufferBuilder, i2);
        addBoneNames(flatBufferBuilder, i);
        return endSkeletonDef(flatBufferBuilder);
    }

    public static int endSkeletonDef(FlatBufferBuilder flatBufferBuilder) {
        return flatBufferBuilder.endObject();
    }

    public static SkeletonDef getRootAsSkeletonDef(ByteBuffer byteBuffer) {
        return getRootAsSkeletonDef(byteBuffer, new SkeletonDef());
    }

    public static void startBoneNamesVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(4, i, 4);
    }

    public static void startBoneParentsVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(1, i, 1);
    }

    public static void startBoneTransformsVector(FlatBufferBuilder flatBufferBuilder, int i) {
        flatBufferBuilder.startVector(48, i, 4);
    }

    public static void startSkeletonDef(FlatBufferBuilder flatBufferBuilder) {
        flatBufferBuilder.startObject(5);
    }

    public SkeletonDef __assign(int i, ByteBuffer byteBuffer) {
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
        int __offset = __offset(4);
        if (__offset != 0) {
            return __string((i * 4) + __vector(__offset));
        }
        return null;
    }

    public int boneNamesLength() {
        int __offset = __offset(4);
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

    public Mat4x3 boneTransforms(int i) {
        return boneTransforms(new Mat4x3(), i);
    }

    public int boneTransformsLength() {
        int __offset = __offset(8);
        if (__offset != 0) {
            return __vector_len(__offset);
        }
        return 0;
    }

    public static int createBoneParentsVector(FlatBufferBuilder flatBufferBuilder, ByteBuffer byteBuffer) {
        return flatBufferBuilder.createByteVector(byteBuffer);
    }

    public static SkeletonDef getRootAsSkeletonDef(ByteBuffer byteBuffer, SkeletonDef skeletonDef) {
        return skeletonDef.__assign(byteBuffer.position() + a.w(byteBuffer, ByteOrder.LITTLE_ENDIAN), byteBuffer);
    }

    public Mat4x3 boneTransforms(Mat4x3 mat4x3, int i) {
        int __offset = __offset(8);
        if (__offset != 0) {
            return mat4x3.__assign((i * 48) + __vector(__offset), this.bb);
        }
        return null;
    }
}